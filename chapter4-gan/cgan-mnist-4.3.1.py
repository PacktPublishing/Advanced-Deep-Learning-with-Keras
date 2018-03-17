'''Trains CGAN on MNIST using Keras

CGAN is Conditional Generative Adversarial Nets.
This version of CGAN is similar to DCGAN. The difference mainly
is that the z-vector of geneerator is conditioned by a one-hot label
to produce specific fake images. The discriminator is trained to
discriminate real from fake images that are conditioned on
specific one-hot labels.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Mirza, Mehdi, and Simon Osindero. "Conditional generative
adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse


def build_generator(inputs, y_labels, image_size):
    """Build a Generator Model

    Inputs are concatenated before Dense layer.
    Stacks of BN-ReLU-Conv2DTranpose to generate fake images
    Output activation is sigmoid instead of tanh in orig DCGAN.
    Sigmoid converges easily.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        y_labels (Layer): Input layer for one-hot vector to condition
            the inputs
        image_size: Target size of one side (assuming square image)

    # Returns
        Model: Generator Model
    """
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = keras.layers.concatenate([inputs, y_labels], axis=1)
    x = Dense(image_resize * image_resize * layer_filters[0])(x)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    generator = Model([inputs, y_labels], x, name='generator')
    return generator


def build_discriminator(inputs, y_labels, image_size):
    """Build a Discriminator Model

    Inputs are concatenated after Dense layer.
    Stacks of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in DCGAN paper.

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        y_labels (Layer): Input layer for one-hot vector to condition
            the inputs
        image_size: Target size of one side (assuming square image)

    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs

    y = Dense(image_size * image_size)(y_labels)
    y = Reshape((image_size, image_size, 1))(y)
    x = keras.layers.concatenate([x, y])

    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model([inputs, y_labels], x, name='discriminator')
    return discriminator


def train(models, data, params):
    """Train the Discriminator and Adversarial Networks

    Alternately train Discriminator and Adversarial networks by batch
    Discriminator is trained first with properly labelled real and fake images
    Adversarial is trained next with fake images pretending to be real
    Discriminator inputs are conditioned by train labels for real images,
    and random labels for fake images
    Adversarial inputs are conditioned by random labels
    Generate sample images per save_interval

    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        data (list): x_train, y_train data
        params (list): Network parameters

    """
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    for i in range(train_steps):
        # Random real images and their labels
        rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        # Generate fake images and their labels
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                           batch_size)]

        fake_images = generator.predict([noise, fake_labels])
        x = np.concatenate((real_images, fake_images))

        y_labels = np.concatenate((real_labels, fake_labels))

        # Label real and fake images
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        # Train the discriminator network
        metrics = discriminator.train_on_batch([x, y_labels], y)
        loss = metrics[0]
        acc = metrics[1]
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        # Generate random noise and fake labels
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                           batch_size)]
        # Label fake images as real
        y = np.ones([batch_size, 1])
        # Train the adversarial network
        metrics = adversarial.train_on_batch([noise, fake_labels], y)
        loss = metrics[0]
        acc = metrics[1]
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images(generator,
                        noise_input=noise_input,
                        noise_class=noise_class,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)
    
    generator.save(model_name + ".h5")


def plot_images(generator,
                noise_input,
                noise_class,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    # Arguments
        generator (Model): The Generator Model for fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name

    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict([noise_input, noise_class])
    print(model_name , " labels for generated images: ", np.argmax(noise_class, axis=1))
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


def build_and_train_models():
    # MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)

    # Network parameters
    model_name = "cgan_mnist"
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 0.0002
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )

    # Build discriminator model
    inputs = Input(shape=input_shape, name='discriminator_input')
    y_labels = Input(shape=label_shape, name='class_labels')

    discriminator = build_discriminator(inputs, y_labels, image_size)
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # Build generator model
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, y_labels, image_size)
    generator.summary()

    # Build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False
    outputs = discriminator([generator([inputs, y_labels]), y_labels])
    adversarial = Model([inputs, y_labels],
                        outputs,
                        name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # Train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    plot_images(generator,
                noise_input=noise_input,
                noise_class=noise_class,
                show=True,
                step=step,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        class_label = None
        if args.digit is not None:
            class_label = args.digit
        test_generator(generator, class_label)
    else:
        build_and_train_models()
