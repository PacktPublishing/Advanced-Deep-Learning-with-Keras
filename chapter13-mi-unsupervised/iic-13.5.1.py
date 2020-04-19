"""Build, train and evaluate an IIC Model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

import numpy as np
import os
import argparse
import vgg

from data_generator import DataGenerator
from utils import unsupervised_labels, center_crop
from utils import AccuracyCallback, lr_schedule


class IIC:
    def __init__(self,
                 args,
                 backbone):
        """Contains the encoder model, the loss function,
            loading of datasets, train and evaluation routines
            to implement IIC unsupervised clustering via mutual
            information maximization

        Arguments:
            args : Command line arguments to indicate choice
                of batch size, number of heads, folder to save
                weights file, weights file name, etc
            backbone (Model): IIC Encoder backbone (eg VGG)
        """
        self.args = args
        self.backbone = backbone
        self._model = None
        self.train_gen = DataGenerator(args, siamese=True)
        self.n_labels = self.train_gen.n_labels
        self.build_model()
        self.load_eval_dataset()
        self.accuracy = 0

    def build_model(self):
        """Build the n_heads of the IIC model
        """
        inputs = Input(shape=self.train_gen.input_shape, name='x')
        x = self.backbone(inputs)
        x = Flatten()(x)
        # number of output heads
        outputs = []
        for i in range(self.args.heads):
            name = "z_head%d" % i
            outputs.append(Dense(self.n_labels,
                                 activation='softmax',
                                 name=name)(x))
        self._model = Model(inputs, outputs, name='encoder')
        optimizer = Adam(lr=1e-3)
        self._model.compile(optimizer=optimizer, loss=self.mi_loss)
        self._model.summary()

    def mi_loss(self, y_true, y_pred):
        """Mutual information loss computed from the joint
           distribution matrix and the marginals

        Arguments:
            y_true (tensor): Not used since this is
                unsupervised learning
            y_pred (tensor): stack of softmax predictions for
                the Siamese latent vectors (Z and Zbar)
        """
        size = self.args.batch_size
        n_labels = y_pred.shape[-1]
        # lower half is Z
        Z = y_pred[0: size, :]
        Z = K.expand_dims(Z, axis=2)
        # upper half is Zbar
        Zbar = y_pred[size: y_pred.shape[0], :]
        Zbar = K.expand_dims(Zbar, axis=1)
        # compute joint distribution (Eq 10.3.2 & .3)
        P = K.batch_dot(Z, Zbar)
        P = K.sum(P, axis=0)
        # enforce symmetric joint distribution (Eq 10.3.4)
        P = (P + K.transpose(P)) / 2.0
        # normalization of total probability to 1.0
        P = P / K.sum(P)
        # marginal distributions (Eq 10.3.5 & .6)
        Pi = K.expand_dims(K.sum(P, axis=1), axis=1)
        Pj = K.expand_dims(K.sum(P, axis=0), axis=0)
        Pi = K.repeat_elements(Pi, rep=n_labels, axis=1)
        Pj = K.repeat_elements(Pj, rep=n_labels, axis=0)
        P = K.clip(P, K.epsilon(), np.finfo(float).max)
        Pi = K.clip(Pi, K.epsilon(), np.finfo(float).max)
        Pj = K.clip(Pj, K.epsilon(), np.finfo(float).max)
        # negative MI loss (Eq 10.3.7)
        neg_mi = K.sum((P * (K.log(Pi) + K.log(Pj) - K.log(P))))
        # each head contribute 1/n_heads to the total loss
        return neg_mi/self.args.heads


    def train(self):
        """Train function uses the data generator,
            accuracy computation, and learning rate
            scheduler callbacks
        """
        accuracy = AccuracyCallback(self)
        lr_scheduler = LearningRateScheduler(lr_schedule,
                                             verbose=1)
        callbacks = [accuracy, lr_scheduler]
        self._model.fit(x=self.train_gen,
                        use_multiprocessing=False,
                        epochs=self.args.epochs,
                        callbacks=callbacks,
                        shuffle=True)


    def load_eval_dataset(self):
        """Pre-load test data for evaluation
        """
        (_, _), (x_test, self.y_test) = self.args.dataset.load_data()
        image_size = x_test.shape[1]
        x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
        x_test = x_test.astype('float32') / 255
        x_eval = np.zeros([x_test.shape[0], *self.train_gen.input_shape])
        for i in range(x_eval.shape[0]):
            x_eval[i] = center_crop(x_test[i])

        self.x_test = x_eval


    def load_weights(self):
        """Reload model weights for evaluation
        """
        if self.args.restore_weights is None:
            raise ValueError("Must load model weights for evaluation")

        if self.args.restore_weights:
            folder = "weights"
            os.makedirs(folder, exist_ok=True) 
            path = os.path.join(folder, self.args.restore_weights)
            print("Loading weights... ", path)
            self._model.load_weights(path)


    def eval(self):
        """Evaluate the accuracy of the current model weights
        """
        y_pred = self._model.predict(self.x_test)
        print("")
        # accuracy per head
        for head in range(self.args.heads):
            if self.args.heads == 1:
                y_head = y_pred
            else:
                y_head = y_pred[head]
            y_head = np.argmax(y_head, axis=1)

            accuracy = unsupervised_labels(list(self.y_test),
                                           list(y_head),
                                           self.n_labels,
                                           self.n_labels)
            info = "Head %d accuracy: %0.2f%%"
            if self.accuracy > 0:
                info += ", Old best accuracy: %0.2f%%" 
                data = (head, accuracy, self.accuracy)
            else:
                data = (head, accuracy)
            print(info % data)
            # if accuracy improves during training, 
            # save the model weights on a file
            if accuracy > self.accuracy \
                    and self.args.save_weights is not None:
                self.accuracy = accuracy
                folder = self.args.save_dir
                os.makedirs(folder, exist_ok=True) 
                path = os.path.join(folder, self.args.save_weights)
                print("Saving weights... ", path)
                self._model.save_weights(path)


    @property
    def model(self):
        return self._model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IIC Keras')
    parser.add_argument('--save-dir',
                       default="weights",
                       help='Folder for storing model weights (h5)')
    parser.add_argument('--save-weights',
                       default=None,
                       help='Folder for storing model weights (h5)')
    parser.add_argument('--dataset',
                       default=mnist,
                       help='Dataset to use')
    parser.add_argument('--epochs',
                        type=int,
                        default=1200,
                        metavar='N',
                        help='Number of epochs to train')
    parser.add_argument('--batch-size',
                        type=int,
                        default=512,
                        metavar='N',
                        help='Train batch size')
    parser.add_argument('--heads',
                        type=int,
                        default=1,
                        metavar='N',
                        help='Number of heads')
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train the model')
    parser.add_argument('--restore-weights',
                        default=None,
                        help='Restore saved model weights')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Evaluate a pre trained model. Must indicate weights file.')
    parser.add_argument('--crop',
                        type=int,
                        default=4,
                        help='Pixels to crop from the image')
    parser.add_argument('--plot-model',
                        default=False,
                        action='store_true',
                        help='Plot all network models')

    args = parser.parse_args()

    # build backbone
    backbone = vgg.VGG(vgg.cfg['F'])
    backbone.model.summary()
    # instantiate IIC object
    iic = IIC(args, backbone.model)
    if args.plot_model:
        plot_model(backbone.model,
                   to_file="model-vgg.png",
                   show_shapes=True)
        plot_model(iic.model,
                   to_file="model-iic.png",
                   show_shapes=True)
    if args.eval:
        iic.load_weights()
        iic.eval()
    elif args.train:
        iic.train()
