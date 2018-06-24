"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import cifar10
from keras.models import load_model
from keras.layers.merge import concatenate

from keras_contrib.layers.normalization import InstanceNormalization

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import cifar10_utils


def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  normalize=True):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if normalize:
        x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2):

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')

    x = InstanceNormalization()(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3):

    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs,
                       32,
                       kernel_size=kernel_size,
                       strides=1) 
    e2 = encoder_layer(e1,
                        64,
                        kernel_size=kernel_size) 
    e3 = encoder_layer(e2,
                       128,
                       kernel_size=kernel_size) 
    e4 = encoder_layer(e3,
                       256,
                       kernel_size=kernel_size) 

    d1 = decoder_layer(e4,
                       e3,
                       128,
                       kernel_size=kernel_size)
    d2 = decoder_layer(d1,
                       e2,
                       64,
                       kernel_size=kernel_size)
    d3 = decoder_layer(d2,
                       e1,
                       32,
                       kernel_size=kernel_size)
    outputs = Conv2DTranspose(channels,
                              kernel_size=kernel_size,
                              strides=1,
                              activation='sigmoid',
                              padding='same')(d3)

    generator = Model(inputs, outputs)

    return generator


def build_discriminator(input_shape,
                        kernel_size=3,
                        patchgan=True):

    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs,
                      32,
                      kernel_size=kernel_size,
                      normalize=False)
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      normalize=False)
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      normalize=False)
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      strides=1,
                      normalize=False)
    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(1,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('linear')(x)


    discriminator = Model(inputs, outputs)

    return discriminator



def colorize_cifar10():
    model_name = 'cyclegan_cifar10'
    batch_size = 32
    train_steps = 100000
    lr = 2e-4
    decay = 6e-8

    data, img_shape = cifar10_utils.load_data()
    rows, cols, _ = img_shape 
    color_shape = img_shape
    gray_shape = (rows, cols, 1)

    gen_color = build_generator(gray_shape, color_shape)
    gen_gray = build_generator(color_shape, gray_shape)
    print('---- COLOR GENERATOR---')
    gen_color.summary()
    print('---- GRAY GENERATOR---')
    gen_gray.summary()

    dis_color = build_discriminator(color_shape, patchgan=False)
    dis_gray = build_discriminator(gray_shape, patchgan=False)
    print('---- COLOR DISCRIMINATOR---')
    dis_color.summary()
    print('---- GRAY DISCRIMINATOR---')
    dis_gray.summary()

    optimizer = RMSprop(lr=lr, decay=decay)
    dis_color.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    dis_gray.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    dis_color.trainable = False
    dis_gray.trainable = False

    img_gray = Input(shape=(rows, cols, 1))
    fake_color = gen_color(img_gray)
    preal_color = dis_color(fake_color)
    reco_gray = gen_gray(fake_color)

    img_color = Input(shape=img_shape)
    fake_gray = gen_gray(img_color)
    preal_gray = dis_gray(fake_gray)
    reco_color = gen_color(fake_gray)

    iden_gray = gen_gray(img_color)
    iden_color = gen_color(img_gray)

    inputs = [img_gray, img_color]
    outputs = [preal_gray,
               preal_color,
               reco_gray,
               reco_color,
               iden_gray,
               iden_color]

    adv = Model(inputs, outputs)
    loss = ['mse', 'mse', 'mae', 'mae', 'mse', 'mse']
    loss_weights = [1., 1., 10., 10., 1., 1.]
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    adv.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    adv.summary()

    # Calculate output shape of D (PatchGAN)
    patch = int(rows / 2**3)
    dis_patch = (patch, patch, 1)

    models = (gen_gray, gen_color, dis_gray, dis_color, adv)
    params = (batch_size, train_steps, dis_patch, model_name)
    cifar10_utils.train(models, data, params)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Train cifar10 colorization"
    parser.add_argument("-c",
                        "--cifar10",
                        action='store_true',
                        help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    else:
        colorize_cifar10()
