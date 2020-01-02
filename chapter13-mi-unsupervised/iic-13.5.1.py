"""Build, train and evaluate an IIC Model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

import numpy as np
import os
import argparse
import vgg

from data_generator import DataGenerator
from utils import unsupervised_labels, center_crop, AccuracyCallback, lr_schedule

from loss import iic_mi_loss
from vgg import VGGBackbone
from models import IICEncoder


class IIC:
    def __init__(self, args):
        self.args = args
        self.train_gen = DataGenerator(args, siamese=True)
        self.n_labels = self.train_gen.n_labels
        self._encoder = IICEncoder(backbone=VGGBackbone(),
                                   n_heads=args.heads,
                                   n_labels=self.n_labels)
        self.init_encoder()
        self.load_eval_dataset()
        self.accuracy = 0


    def init_encoder(self):
        # inputs = Input(shape=self.train_gen.input_shape, name='x')
        optimizer = Adam(lr=1e-3)
        loss = iic_mi_loss(self.args.batch_size, self.args.heads)
        self._encoder.compile(optimizer=optimizer, loss=loss)


    # train the model
    def train(self):
        accuracy = AccuracyCallback(self)
        lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
        callbacks = [accuracy, lr_scheduler]
        self._encoder.fit_generator(generator=self.train_gen,
                                    use_multiprocessing=True,
                                    epochs=self.args.epochs,
                                    callbacks=callbacks,
                                    workers=4,
                                    shuffle=True)


    # pre-load test data for evaluation
    def load_eval_dataset(self):
        (_, _), (x_test, self.y_test) = self.args.dataset.load_data()
        image_size = x_test.shape[1]
        x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
        x_test = x_test.astype('float32') / 255
        x_eval = np.zeros([x_test.shape[0], *self.train_gen.input_shape])
        for i in range(x_eval.shape[0]):
            x_eval[i] = center_crop(x_test[i])

        self.x_test = x_eval


    # reload model weights for evaluation
    def load_weights(self):
        if self.args.restore_weights is None:
            raise ValueError("Must load model weights for evaluation")

        if self.args.restore_weights:
            folder = "weights"
            os.makedirs(folder, exist_ok=True) 
            path = os.path.join(folder, self.args.restore_weights)
            print("Loading weights... ", path)
            self._encoder.load_weights(path)


    # evaluate the accuracy of the current model weights
    def eval(self):
        y_pred = self._encoder.predict(self.x_test)
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
                self._encoder.save_weights(path)


    @property
    def encoder(self):
        return self._encoder


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IIC Keras')
    parser.add_argument('--save-dir',
                       default="weights",
                       help='Folder for storing model weights (h5)')
    parser.add_argument('--save-weights',
                       default=None,
                       help='File name for storing model weights (h5)')
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
                        help='File name to restore saved model weights')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Evaluate a pre trained model.\
                              Must indicate weights file.')
    parser.add_argument('--crop',
                        type=int,
                        default=4,
                        help='Pixels to crop from the image')
    parser.add_argument('--plot-model',
                        default=False,
                        action='store_true',
                        help='Plot all network models')

    args = parser.parse_args()

    # instantiate IIC object
    iic = IIC(args)
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
