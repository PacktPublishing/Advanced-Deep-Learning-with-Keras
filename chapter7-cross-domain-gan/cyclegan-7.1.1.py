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
                  activate=True,
                  normalize=True):

    conv = Conv2D(filters=filters,
                  kernel_size=5,
                  strides=2,
                  padding='same')

    x = inputs
    if normalize:
        x = InstanceNormalization()(x)
    if activate:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  strides=2,
                  activation='relu'):

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=5,
                           strides=strides,
                           activation=activation,
                           padding='same')

    x = conv(inputs)
    x = InstanceNormalization()(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape, output_shape=None):

    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs, 16) 
    e2 = encoder_layer(e1, 32) 
    e3 = encoder_layer(e2, 64) 
    e4 = encoder_layer(e3, 128) 

    d1 = decoder_layer(e4, e3, 64)
    d2 = decoder_layer(d1, e2, 32)
    d3 = decoder_layer(d2, e1, 16)
    outputs = Conv2DTranspose(channels,
                              kernel_size=5,
                              strides=2,
                              activation='tanh',
                              padding='same')(d3)



    generator = Model(inputs, outputs)

    return generator


def build_discriminator(input_shape):

    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs, 32, normalize=False)
    x = encoder_layer(x, 64)
    x = encoder_layer(x, 128)
    x = encoder_layer(x, 256)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2D(1,
                     kernel_size=5,
                     strides=1,
                     activation='linear',
                     padding='same')(x)

    discriminator = Model(inputs, outputs)

    return discriminator



def train_cifar10(models, data, params):
    # the models
    gen_gray, gen_color, dis_gray, dis_color, adv = models
    # network parameters
    batch_size, train_steps, dis_patch, model_name = params
    # train dataset
    x_train, x_test, x_train_gray, x_test_gray = data
    # the generator image is saved every 500 steps
    save_interval = 500
    # number of elements in train dataset
    train_size = x_train.shape[0]
    # test_size = x_test.shape[0]

    valid = np.ones((batch_size,) + dis_patch)
    fake = np.zeros((batch_size,) + dis_patch)
    y = np.concatenate((valid, fake))


    for i in range(train_steps):
        # train the discriminator1 for 1 batch
        # 1 batch of real (label=1.0) and fake feature1 (label=0.0)
        # randomly pick real images from dataset
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_color = x_train[rand_indexes]

        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_gray = x_train_gray[rand_indexes]
        fake_color = gen_color.predict(real_gray)

        metrics1 = dis_color.train_on_batch(real_color, valid)
        metrics2 = dis_color.train_on_batch(fake_color, fake)
        loss = 0.5 * (metrics1[0] + metrics2[1])
        log = "%d: [dis_color loss: %f]" % (i, loss)

        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_gray = x_train_gray[rand_indexes]

        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_color = x_train[rand_indexes]
        fake_gray = gen_gray.predict(real_color)

        metrics1 = dis_gray.train_on_batch(real_gray, valid)
        metrics2 = dis_gray.train_on_batch(fake_gray, fake)
        loss = 0.5 * (metrics1[0] + metrics2[1])
        log = "%s [dis_gray loss: %f]" % (log, loss)

        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_gray = x_train_gray[rand_indexes]
        fake_color = gen_color.predict(real_gray)

        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_color = x_train[rand_indexes]
        fake_gray = gen_gray.predict(real_color)

        x = [real_gray, real_color]
        y = [valid, valid, fake_gray, fake_color]
        metrics = adv.train_on_batch(x, y)
        log = "%s [adv loss: %f]" % (log, metrics[0])
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False

            cifar10_utils.test_generator(gen_color,
                                         x_test_gray,
                                         step=i+1,
                                         show=show)


def colorize_cifar10():

    model_name = 'cyclegan_cifar10'
    batch_size = 32
    train_steps = 10000
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

    dis_color = build_discriminator(color_shape)
    dis_gray = build_discriminator(gray_shape)
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

    # iden_gray = gen_gray(fake_gray)
    # iden_color = gen_color(fake_color)

    inputs = [img_gray, img_color]
    outputs = [preal_gray,
               preal_color,
               reco_gray,
               reco_color]

    adv = Model(inputs, outputs)
    loss = ['mse', 'mse', 'mae', 'mae']
    loss_weights = [1., 1., 10., 10.]
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    adv.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    adv.summary()

    # Calculate output shape of D (PatchGAN)
    patch = int(rows / 2**4)
    dis_patch = (patch, patch, 1)

    models = (gen_gray, gen_color, dis_gray, dis_color, adv)
    params = (batch_size, train_steps, dis_patch, model_name)
    train_cifar10(models, data, params)


def test_generator(generator):
    # predict the autoencoder output from test data
    x_decoded = autoencoder.predict(x_test_gray)

    # display the 1st 100 colorized images
    imgs = x_decoded[:100]
    title = 'Colorized test images (Predicted)'
    filename = '%s/colorized.png' % imgs_dir
    display_images(imgs,
                   img_shape=img_shape,
                   filename=filename,
                   title=title)


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
