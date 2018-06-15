'''Trains LSGAN on MNIST using Keras

LSGAN is similar to DCGAN except for the MSE loss used by the 
Discriminator and Adversarial networks.
  
[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Mao, Xudong, et al. "Least squares generative adversarial networks." 
2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model

import numpy as np
import argparse

import sys
sys.path.append("..")
from lib import gan


def build_and_train_models():
    # load MNIST dataset
    (x_train, _), (_, _) = mnist.load_data()

    # reshape data for CNN as (28, 28, 1) and normalize
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    model_name = "lsgan_mnist"
    # network parameters
    # the latent or z vector is 100-dim
    latent_size = 100
    input_shape = (image_size, image_size, 1)
    batch_size = 64
    lr = 2e-4
    decay = 6e-8
    train_steps = 40000

    # build discriminator model
    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = gan.discriminator(inputs, activation=None)
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # LSGAN uses MSE loss [2]
    discriminator.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # build generator model
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = gan.generator(inputs, image_size)
    generator.summary()

    # build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    # freeze the weights of discriminator during adversarial training
    discriminator.trainable = False
    adversarial = Model(inputs,
                        discriminator(generator(inputs)),
                        name=model_name)
    # LSGAN uses MSE loss [2]
    adversarial.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_size, train_steps, model_name)
    gan.train(models, x_train, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        gan.test_generator(generator)
    else:
        build_and_train_models()
