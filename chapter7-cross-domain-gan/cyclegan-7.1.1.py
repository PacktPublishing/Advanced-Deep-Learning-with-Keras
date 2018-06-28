"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop
from keras.models import Model
from keras.models import load_model
from keras.layers.merge import concatenate

from keras_contrib.layers.normalization import InstanceNormalization

import numpy as np
import argparse
import cifar10_utils
import mnist_svhn_utils
import other_utils
import datetime


def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  instance_norm=True):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  instance_norm=True):

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name=None):

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

    generator = Model(inputs, outputs, name=name)

    return generator


def build_discriminator(input_shape,
                        kernel_size=3,
                        patchgan=True,
                        name=None):

    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs,
                      32,
                      kernel_size=kernel_size,
                      instance_norm=False)
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      instance_norm=False)
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      instance_norm=False)
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      strides=1,
                      instance_norm=False)
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


    discriminator = Model(inputs, outputs, name=name)

    return discriminator


def train_cyclegan(models, data, params, test_params, test_generator):
    # the models
    g_source, g_target, d_source, d_target, adv = models
    # network parameters
    batch_size, train_steps, dis_patch, model_name = params
    # train dataset
    source_data, target_data, test_source_data, test_target_data = data

    titles, dirs = test_params

    # the generator image is saved every 500 steps
    save_interval = 500
    # number of elements in train dataset
    target_size = target_data.shape[0]
    source_size = source_data.shape[0]

    # valid = np.ones((batch_size,) + dis_patch)
    # fake = np.zeros((batch_size,) + dis_patch)
    valid = np.ones([batch_size, 1])
    fake = np.zeros([batch_size, 1])
    valid_fake = np.concatenate((valid, fake))

    start_time = datetime.datetime.now()

    for step in range(train_steps):
        rand_indexes = np.random.randint(0, target_size, size=batch_size)
        real_target = target_data[rand_indexes]

        rand_indexes = np.random.randint(0, source_size, size=batch_size)
        real_source = source_data[rand_indexes]
        fake_target = g_target.predict(real_source)
        
        x = np.concatenate((real_target, fake_target))
        metrics = d_target.train_on_batch(x, valid_fake)
        log = "%d: [d_target loss: %f]" % (step, metrics[0])

        fake_source = g_source.predict(real_target)
        x = np.concatenate((real_source, fake_source))
        metrics = d_source.train_on_batch(x, valid_fake)
        log = "%s [d_source loss: %f]" % (log, metrics[0])

        x = [real_source, real_target]
        y = [valid, valid, real_source, real_target]
        metrics = adv.train_on_batch(x, y)
        # print(adv.metrics_names)
        elapsed_time = datetime.datetime.now() - start_time
        fmt = "%s [adv loss: %f] [time: %s]"
        log = fmt % (log, metrics[0], elapsed_time)
        print(log)
        if (step + 1) % save_interval == 0:
            if (step + 1) == train_steps:
                show = True
            else:
                show = False

            test_generator((g_source, g_target),
                           (test_source_data, test_target_data),
                           step=step+1,
                           titles=titles,
                           dirs=dirs,
                           show=show)

    # save the model after training the generator
    g_source.save(model_name + "-g_source.h5")
    g_target.save(model_name + "-g_target.h5")



def build_cyclegan(shapes,
                   source_name='source',
                   target_name='target',
                   kernel_size=3,
                   patchgan=False
                   ):

    source_shape, target_shape = shapes
    lr = 2e-4
    decay = 6e-8
    gt_name = "gen_" + target_name
    gs_name = "gen_" + source_name
    dt_name = "dis_" + target_name
    ds_name = "dis_" + source_name

    g_target = build_generator(source_shape,
                               target_shape,
                               kernel_size=kernel_size,
                               name=gt_name)
    g_source = build_generator(target_shape,
                               source_shape,
                               kernel_size=kernel_size,
                               name=gs_name)
    print('---- TARGET GENERATOR ----')
    g_target.summary()
    print('---- SOURCE GENERATOR ----')
    g_source.summary()

    d_target = build_discriminator(target_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=dt_name)
    d_source = build_discriminator(source_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=ds_name)
    print('---- TARGET DISCRIMINATOR ----')
    d_target.summary()
    print('---- SOURCE DISCRIMINATOR ----')
    d_source.summary()

    optimizer = RMSprop(lr=lr, decay=decay)
    d_target.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    d_source.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])

    d_target.trainable = False
    d_source.trainable = False

    source_input = Input(shape=source_shape)
    fake_target = g_target(source_input)
    preal_target = d_target(fake_target)
    reco_source = g_source(fake_target)

    target_input = Input(shape=target_shape)
    fake_source = g_source(target_input)
    preal_source = d_source(fake_source)
    reco_target = g_target(fake_source)

    # iden_gray = g_source(img_color)
    # iden_color = g_target(img_gray)

    inputs = [source_input, target_input]
    outputs = [preal_source,
               preal_target,
               reco_source,
               reco_target]
               # iden_gray,
               # iden_color]

    adv = Model(inputs, outputs, name='adversarial')
    loss = ['mse', 'mse', 'mae', 'mae']
    loss_weights = [1., 1., 10., 10.]
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    adv.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    print('---- ADVERSARIAL NETWORK ----')
    adv.summary()

    return g_source, g_target, d_source, d_target, adv


def graycifar10_cross_colorcifar10():
    model_name = 'cyclegan_cifar10'
    batch_size = 32
    train_steps = 100000

    data, shapes = cifar10_utils.load_data()
    models = build_cyclegan(shapes, "gray", "color", kernel_size=3)
    params = (batch_size, train_steps, 1, model_name)
    titles = ('CIFAR10 predicted source images.', 'CIFAR10 predicted target images.')
    dirs = ('cifar10_source', 'cifar10_target')
    test_params = (titles, dirs)
    train_cyclegan(models,
                   data,
                   params,
                   test_params,
                   other_utils.test_generator)


def mnist_cross_svhn():
    model_name = 'cyclegan_mnist_svhn'
    batch_size = 32
    train_steps = 100000

    data, shapes = mnist_svhn_utils.load_data()
    models = build_cyclegan(shapes, "mnist", "svhn", kernel_size=3)
    params = (batch_size, train_steps, 1, model_name)
    titles = ('MNIST predicted source images.', 'SVHN predicted target images.')
    dirs = ('mnist_source', 'svhn_target')
    test_params = (titles, dirs)
    train_cyclegan(models,
                   data,
                   params,
                   test_params,
                   other_utils.test_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Train cifar10 colorization"
    parser.add_argument("-c",
                        "--cifar10",
                        action='store_true',
                        help=help_)
    help_ = "Train mnist-svhn cross domain cyclegan"
    parser.add_argument("-m",
                        "--mnist-svhn",
                        action='store_true',
                        help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    elif args.cifar10:
        graycifar10_cross_colorcifar10()
    else:
        mnist_cross_svhn()
