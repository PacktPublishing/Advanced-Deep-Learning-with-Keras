"""Build, train and evaluate a MINE Model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras.layers import Input, Dense, Add, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
import os
import argparse
import vgg

import matplotlib.pyplot as plt
from scipy.stats.contingency import margins
from data_generator import DataGenerator
from utils import unsupervised_labels, center_crop, AccuracyCallback, lr_schedule

def sample(joint=True,
           mean=[0, 0],
           cov=[[1, 0.5], [0.5, 1]],
           n_data=1000000):
    """Helper function to obtain samples 
        fr a bivariate Gaussian distribution

    Arguments:
        joint (Bool): If joint distribution is desired
        mean (list): The mean values of the 2D Gaussian
        cov (list): The covariance matrix of the 2D Gaussian
        n_data (int): Number of samples fr 2D Gaussian
    """
    xy = np.random.multivariate_normal(mean=mean,
                                       cov=cov,
                                       size=n_data)
    # samples fr joint distribution
    if joint:
        return xy 
    y = np.random.multivariate_normal(mean=mean,
                                      cov=cov,
                                      size=n_data)

    # samples fr marginal distribution
    x = xy[:,0].reshape(-1,1)
    y = y[:,1].reshape(-1,1)
   
    xy = np.concatenate([x, y], axis=1)
    return xy

def compute_mi(cov_xy=0.5, n_bins=100):
    """Analytic computation of MI using binned 
        2D Gaussian

    Arguments:
        cov_xy (list): Off-diagonal elements of covariance
            matrix
        n_bins (int): Number of bins to "quantize" the
            continuous 2D Gaussian
    """
    cov=[[1, cov_xy], [cov_xy, 1]]
    data = sample(cov=cov)
    # get joint distribution samples
    # perform histogram binning
    joint, edge = np.histogramdd(data, bins=n_bins)
    joint /= joint.sum()
    eps = np.finfo(float).eps
    joint[joint<eps] = eps
    # compute marginal distributions
    x, y = margins(joint)

    xy = x*y
    xy[xy<eps] = eps
    # MI is P(X,Y)*log(P(X,Y)/P(X)*P(Y))
    mi = joint*np.log(joint/xy)
    mi = mi.sum()
    print("Computed MI: %0.6f" % mi)
    return mi

class SimpleMINE:
    def __init__(self,
                 args,
                 input_dim=1,
                 hidden_units=16,
                 output_dim=1):
        """Learn to compute MI using MINE (Algorithm 13.7.1)

        Arguments:
            args : User-defined arguments such as off-diagonal
                elements of covariance matrix, batch size, 
                epochs, etc
            input_dim (int): Input size dimension
            hidden_units (int): Number of hidden units of the 
                MINE MLP network
            output_dim (int): Output size dimension
        """
        self.args = args
        self._model = None
        self.build_model(input_dim,
                         hidden_units,
                         output_dim)


    def build_model(self,
                    input_dim,
                    hidden_units,
                    output_dim):
        """Build a simple MINE model
        
        Arguments:
            See class arguments.
        """
        inputs1 = Input(shape=(input_dim), name="x")
        inputs2 = Input(shape=(input_dim), name="y")
        x1 = Dense(hidden_units)(inputs1)
        x2 = Dense(hidden_units)(inputs2)
        x = Add()([x1, x2])
        x = Activation('relu', name="ReLU")(x)
        outputs = Dense(output_dim, name="MI")(x)
        inputs = [inputs1, inputs2]
        self._model = Model(inputs,
                            outputs,
                            name='MINE')
        self._model.summary()


    def mi_loss(self, y_true, y_pred):
        """ MINE loss function

        Arguments:
            y_true (tensor): Not used since this is
                unsupervised learning
            y_pred (tensor): stack of predictions for
                joint T(x,y) and marginal T(x,y)
        """
        size = self.args.batch_size
        # lower half is pred for joint dist
        pred_xy = y_pred[0: size, :]

        # upper half is pred for marginal dist
        pred_x_y = y_pred[size : y_pred.shape[0], :]
        # implentation of MINE loss (Eq 13.7.3)
        loss = K.mean(pred_xy) \
               - K.log(K.mean(K.exp(pred_x_y)))
        return -loss


    def train(self):
        """Train MINE to estimate MI between 
            X and Y of a 2D Gaussian
        """
        optimizer = Adam(lr=0.01)
        self._model.compile(optimizer=optimizer,
                            loss=self.mi_loss)
        plot_loss = []
        cov=[[1, self.args.cov_xy], [self.args.cov_xy, 1]]
        loss = 0.
        for epoch in range(self.args.epochs):
            # joint dist samples
            xy = sample(n_data=self.args.batch_size,
                        cov=cov)
            x1 = xy[:,0].reshape(-1,1)
            y1 = xy[:,1].reshape(-1,1)
            # marginal dist samples
            xy = sample(joint=False,
                        n_data=self.args.batch_size,
                        cov=cov)
            x2 = xy[:,0].reshape(-1,1)
            y2 = xy[:,1].reshape(-1,1)
    
            # train on batch of joint & marginal samples
            x =  np.concatenate((x1, x2))
            y =  np.concatenate((y1, y2))
            loss_item = self._model.train_on_batch([x, y],
                                                   np.zeros(x.shape))
            loss += loss_item
            plot_loss.append(-loss_item)
            if (epoch + 1) % 100 == 0:
                fmt = "Epoch %d MINE MI: %0.6f" 
                print(fmt % ((epoch+1), -loss/100))
                loss = 0.

        plt.plot(plot_loss, color='black')
        plt.xlabel('epoch')
        plt.ylabel('MI')
        plt.savefig("simple_mine_mi.png", dpi=300, color='black')
        plt.show()


    @property
    def model(self):
        return self._model


class LinearClassifier:
    def __init__(self,
                 latent_dim=10,
                 n_classes=10):
        """A simple MLP-based linear classifier. 
            A linear classifier is an MLP network
            without non-linear activation like ReLU.
            This can be used as a substitute to linear
            assignment algorithm.

        Arguments:
            latent_dim (int): Latent vector dimensionality
            n_classes (int): Number of classes the latent
                dim will be converted to.
        """
        self.build_model(latent_dim, n_classes)


    def build_model(self, latent_dim, n_classes):
        """Linear classifier model builder.

        Arguments: (see class arguments)
        """
        inputs = Input(shape=(latent_dim,), name="cluster")
        x = Dense(256)(inputs)
        outputs = Dense(n_classes,
                        activation='softmax',
                        name="class")(x)
        name = "classifier"
        self._model = Model(inputs, outputs, name=name)
        self._model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
        self._model.summary()


    def train(self, x_test, y_test):
        """Linear classifier training. 

        Arguments:
            x_test (tensor): Image fr test dataset
            y_test (tensor): Corresponding image label
                fr test dataset
        """
        self._model.fit(x_test,
                        y_test,
                        epochs=10,
                        batch_size=128)


    def eval(self, x_test, y_test):
        """Linear classifier evaluation. 

        Arguments:
            x_test (tensor): Image fr test dataset
            y_test (tensor): Corresponding image label
                fr test dataset
        """
        self._model.fit(x_test,
                        y_test,
                        epochs=10,
                        batch_size=128)

        score = self._model.evaluate(x_test,
                                     y_test,
                                     batch_size=128,
                                     verbose=0)
        accuracy = score[1] * 100
        return accuracy


    @property
    def model(self):
        return self._model



class MINE:
    def __init__(self,
                 args,
                 backbone):
        """Contains the encoder, SimpleMINE, and linear 
            classifier models, the loss function,
            loading of datasets, train and evaluation routines
            to implement MINE unsupervised clustering via mutual
            information maximization

        Arguments:
            args : Command line arguments to indicate choice
                of batch size, folder to save
                weights file, weights file name, etc
            backbone (Model): MINE Encoder backbone (eg VGG)
        """
        self.args = args
        self.latent_dim = args.latent_dim
        self.backbone = backbone
        self._model = None
        self._encoder = None
        self.train_gen = DataGenerator(args, 
                                       siamese=True,
                                       mine=True)
        self.n_labels = self.train_gen.n_labels
        self.build_model()
        self.accuracy = 0


    def build_model(self):
        """Build the MINE model unsupervised classifier
        """
        inputs = Input(shape=self.train_gen.input_shape,
                       name="x")
        x = self.backbone(inputs)
        x = Flatten()(x)
        y = Dense(self.latent_dim,
                  activation='linear',
                  name="encoded_x")(x)
        # encoder is based on backbone (eg VGG)
        # feature extractor
        self._encoder = Model(inputs, y, name="encoder")
        # the SimpleMINE in bivariate Gaussian is used 
        # as T(x,y) function in MINE (Algorithm 13.7.1)
        self._mine = SimpleMINE(self.args,
                                input_dim=self.latent_dim,
                                hidden_units=1024,
                                output_dim=1)
        inputs1 = Input(shape=self.train_gen.input_shape,
                        name="x")
        inputs2 = Input(shape=self.train_gen.input_shape,
                        name="y")
        x1 = self._encoder(inputs1)
        x2 = self._encoder(inputs2)
        outputs = self._mine.model([x1, x2])
        # the model computes the MI between 
        # inputs1 and 2 (x and y)
        self._model = Model([inputs1, inputs2],
                            outputs, 
                            name='encoder')
        optimizer = Adam(lr=1e-3)
        self._model.compile(optimizer=optimizer, 
                            loss=self.mi_loss)
        self._model.summary()
        self.load_eval_dataset()
        self._classifier = LinearClassifier(\
                            latent_dim=self.latent_dim)


    def mi_loss(self, y_true, y_pred):
        """ MINE loss function

        Arguments:
            y_true (tensor): Not used since this is
                unsupervised learning
            y_pred (tensor): stack of predictions for
                joint T(x,y) and marginal T(x,y)
        """
        size = self.args.batch_size
        # lower half is pred for joint dist
        pred_xy = y_pred[0: size, :]

        # upper half is pred for marginal dist
        pred_x_y = y_pred[size : y_pred.shape[0], :]
        loss = K.mean(K.exp(pred_x_y))
        loss = K.clip(loss, K.epsilon(), np.finfo(float).max)
        loss = K.mean(pred_xy) - K.log(loss)
        return -loss


    def train(self):
        """Train MINE to estimate MI between 
            X and Y (eg MNIST image and its transformed
            version)
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
        (_, _), (x_test, self.y_test) = \
                self.args.dataset.load_data()
        image_size = x_test.shape[1]
        x_test = np.reshape(x_test,
                            [-1, image_size, image_size, 1])
        x_test = x_test.astype('float32') / 255
        x_eval = np.zeros([x_test.shape[0],
                          *self.train_gen.input_shape])
        for i in range(x_eval.shape[0]):
            x_eval[i] = center_crop(x_test[i])

        self.y_test = to_categorical(self.y_test)
        self.x_test = x_eval


    def load_weights(self):
        """Reload model weights for evaluation
        """
        if self.args.restore_weights is None:
            error_msg = "Must load model weights for evaluation"
            raise ValueError(error_msg)

        if self.args.restore_weights:
            folder = "weights"
            os.makedirs(folder, exist_ok=True) 
            path = os.path.join(folder, self.args.restore_weights)
            print("Loading weights... ", path)
            self._model.load_weights(path)


    def eval(self):
        """Evaluate the accuracy of the current model weights
        """
        # generate clustering predictions fr test data
        y_pred = self._encoder.predict(self.x_test)
        # train a linear classifier
        # input: clustered data
        # output: ground truth labels
        self._classifier.train(y_pred, self.y_test)
        accuracy = self._classifier.eval(y_pred, self.y_test)

        info = "Accuracy: %0.2f%%"
        if self.accuracy > 0:
            info += ", Old best accuracy: %0.2f%%" 
            data = (accuracy, self.accuracy)
        else:
            data = (accuracy)
        print(info % data)
        # if accuracy improves during training, 
        # save the model weights on a file
        if accuracy > self.accuracy \
            and self.args.save_weights is not None:
            folder = self.args.save_dir
            os.makedirs(folder, exist_ok=True) 
            args = (self.latent_dim, self.args.save_weights)
            filename = "%d-dim-%s" % args
            path = os.path.join(folder, filename)
            print("Saving weights... ", path)
            self._model.save_weights(path)

        if accuracy > self.accuracy: 
            self.accuracy = accuracy


    @property
    def model(self):
        return self._model


    @property
    def encoder(self):
        return self._encoder


    @property
    def classifier(self):
        return self._classifier



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MI on 2D Gaussian')
    parser.add_argument('--cov_xy',
                        type=float,
                        default=0.5,
                        help='Gaussian off diagonal element')
    parser.add_argument('--save-dir',
                       default="weights",
                       help='Folder for storing model weights')
    parser.add_argument('--save-weights',
                       default=None,
                       help='Filename (dim added) of model weights (h5).')
    parser.add_argument('--dataset',
                       default=mnist,
                       help='Dataset to use')
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='Number of epochs to train')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='Train batch size')
    parser.add_argument('--gaussian',
                        default=False,
                        action='store_true',
                        help='Compute MI of 2D Gaussian')
    parser.add_argument('--plot-model',
                        default=False,
                        action='store_true',
                        help='Plot all network models')
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train the model')
    parser.add_argument('--latent-dim',
                        type=int,
                        default=10,
                        metavar='N',
                        help='MNIST encoder latent dim')
    parser.add_argument('--restore-weights',
                        default=None,
                        help='Restore saved model weights')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Evaluate a pre trained model. Must indicate weights file.')

    args = parser.parse_args()
    if args.gaussian:
        print("Covariace off diagonal:", args.cov_xy)
        simple_mine = SimpleMINE(args)
        simple_mine.train()
        compute_mi(cov_xy=args.cov_xy)
        if args.plot_model:
            plot_model(simple_mine.model,
                       to_file="simple_mine.png",
                       show_shapes=True)
    else:
        # build backbone
        backbone = vgg.VGG(vgg.cfg['F'])
        backbone.model.summary()
        # instantiate MINE object
        mine = MINE(args, backbone.model)
        if args.plot_model:
            plot_model(mine.classifier.model,
                       to_file="classifier.png",
                       show_shapes=True)
            plot_model(mine.encoder,
                       to_file="encoder.png",
                       show_shapes=True)
            plot_model(mine.model,
                       to_file="model-mine.png",
                       show_shapes=True)
        if args.train:
            mine.train()
    
        if args.eval:
            mine.load_weights()
            mine.eval()
