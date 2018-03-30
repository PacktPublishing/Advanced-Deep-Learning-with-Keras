'''Trains infoGAN on MNIST using Keras

This version of infoGAN is similar to DCGAN. The difference mainly
is that the z-vector of geneerator is conditioned by a one-hot label
to produce specific fake images. The discriminator is trained to
discriminate real from fake images and predict the corresponding
one-hot labels.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).


[2] Chen, Xi, et al. "Infogan: Interpretable representation learning by
information maximizing generative adversarial nets." 
Advances in Neural Information Processing Systems. 2016.
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
from keras import backend as K


import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse


def build_generator(inputs, latent_codes, image_size):
    """Build a Generator Model

    Inputs are concatenated before Dense layer.
    Stacks of BN-ReLU-Conv2DTranpose to generate fake images.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        latent_codes (tuple): dicrete code (labels), and continuous codes
            the inputs
        image_size: Target size of one side (assuming square image)

    # Returns
        Model: Generator Model
    """

    y_labels, y_code1, y_code2 = latent_codes
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    generator_inputs = [inputs, y_labels, y_code1, y_code2]
    x = keras.layers.concatenate(generator_inputs, axis=1)
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

    x = Activation('tanh')(x)
    generator = Model(generator_inputs, x, name='generator')
    return generator


def build_discriminator(inputs, num_labels, image_size):
    """Build a Discriminator Model

    Stacks of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in DCGAN paper.

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        num_labels (int): Dimension of one-hot vector output
        image_size (int): Target size of one side (assuming square image)

    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
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

    # First output is probability that the image is real
    y_source = Dense(1)(x)
    y_source = Activation('sigmoid', name='source')(y_source)

    # 2nd output is 10-dim one-hot vector of label (discrete Q of c given x)
    y_q = Dense(layer_filters[-2])(x)
    y_class = Dense(num_labels)(y_q)
    y_class = Activation('softmax', name='label')(y_class)

    # 3rd output is 1-dim continous Q of 1st c given x
    y_code1 = Dense(1)(y_q)
    y_code1 = Activation('sigmoid', name='code1')(y_code1)

    # 4th output is 1-dim continuous Q of 2nd c given x
    y_code2 = Dense(1)(y_q)
    y_code2 = Activation('sigmoid', name='code2')(y_code2)

    discriminator_outputs = [y_source, y_class, y_code1, y_code2]
    discriminator = Model(inputs, discriminator_outputs, name='discriminator')
    return discriminator


def train(models, data, params):
    """Train the discriminator and adversarial Networks

    Alternately train discriminator and adversarial networks by batch.
    Discriminator is trained first with real and fake images,
    corresponding one-hot labels and continuous codes
    Adversarial is trained next with fake images pretending to be real,
    corresponding one-hot labels and continous codes
    Generate sample images per save_interval.

    # Arguments
        models (tuple): Generator, Discriminator, Adversarial models
        data (tuple): x_train, y_train data
        params (tuple): Network parameters

    """
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    noise_code1 = np.random.normal(scale=0.5, size=[16, 1])
    noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    noise_params = [noise_input, noise_class, noise_code1, noise_code2]
    for i in range(train_steps):
        # Random real images, labels and codes
        rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        real_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        real_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])

        # Generate fake images, labels and codes
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])

        generator_inputs = [noise, fake_labels, fake_code1, fake_code2]
        fake_images = generator.predict(generator_inputs)

        x = np.concatenate((real_images, fake_images))
        y_labels = np.concatenate((real_labels, fake_labels))
        y_codes1 = np.concatenate((real_code1, fake_code1))
        y_codes2 = np.concatenate((real_code2, fake_code2))

        # Label real and fake images
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        # Train the discriminator network
        discriminator_outputs = [y, y_labels, y_codes1, y_codes2]
        metrics = discriminator.train_on_batch(x, discriminator_outputs)
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, accuracy)

        # Generate fake images, labels and codes
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])
        # Label fake images as real
        y = np.ones([batch_size, 1])

        # Train the adversarial network
        advers_inputs = [noise, fake_labels, fake_code1, fake_code2]
        advers_outputs = [y, fake_labels, fake_code1, fake_code2]
        metrics = adversarial.train_on_batch(advers_inputs, advers_outputs)

        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, accuracy)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                generator.save(model_name + ".h5")
                show = True
            else:
                show = False
            plot_images(generator,
                        noise_params=noise_params,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)
    

def mi_loss(c, q_of_c_given_x):
    """ Mutual information, Equation 5 in [2] , assuming H(c) is constant"""
    # lambda is 0.5
    conditional_entropy = K.mean(-K.sum(K.log(q_of_c_given_x + K.epsilon()) * c, axis=1))
    return 0.5 * conditional_entropy

def plot_images(generator,
                noise_params,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    # Arguments
        generator (Model): The Generator Model for fake images generation
        noise_params (list): noise parameters (noise, label, codes)
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name

    """
    noise_input, noise_class, _, _ = noise_params
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = (generator.predict(noise_params) + 1.0) * 0.5 
    print(model_name,
          " labels for generated images: ",
          np.argmax(noise_class, axis=1))

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


def build_and_train_models(latent_size=100):
    # MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = (x_train.astype('float32') - 127.5) / 127.5

    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)

    model_name = "infogan_mnist"
    # Network parameters
    batch_size = 64
    train_steps = 20000
    lr = 0.0002
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )
    code_shape = (1, )

    # Build discriminator Model
    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = build_discriminator(inputs, num_labels, image_size)
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # Loss fuctions: 1) Probability image is real
    # 2) Class label of the image, 3) and 4) Mutual Information Loss
    loss = ['binary_crossentropy', 'categorical_crossentropy', mi_loss, mi_loss]
    discriminator.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # Build generator model
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    y_labels = Input(shape=label_shape, name='y_labels')
    y_code1 = Input(shape=code_shape, name="continuous_latent_code1")
    y_code2 = Input(shape=code_shape, name="continuous_latent_code2")
    latent_codes = (y_labels, y_code1, y_code2)
    generator = build_generator(inputs, latent_codes, image_size)
    generator.summary()

    # Build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False
    generator_inputs = [inputs, y_labels, y_code1, y_code2]
    adversarial = Model(generator_inputs,
                        discriminator(generator(generator_inputs)),
                        name=model_name)
    adversarial.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # Train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, params, latent_size=100):
    class_label, latent_code1, latent_code2 = params
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    if latent_code1 is None:
        noise_code1 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        noise_code1 = np.ones((16, 1)) * latent_code1

    if latent_code2 is None:
        noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        noise_code2 = np.ones((16, 1)) * latent_code2
        # a = np.linspace(-2, 2, 16)
        # a = np.reshape(a, [16, 1])
        # noise_code2 = np.ones((16, 1)) * a
        # print(noise_code2)

    noise_params = [noise_input, noise_class, noise_code1, noise_code2]

    plot_images(generator,
                noise_params=noise_params,
                show=True,
                step=step,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    help_ = "Specify latent code 1"
    parser.add_argument("-a", "--code1", type=float, help=help_)
    help_ = "Specify latent code 2"
    parser.add_argument("-b", "--code2", type=float, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        class_label = None
        latent_code1 = None
        latent_code2 = None
        if args.digit is not None:
            class_label = args.digit
        if args.code1 is not None:
            latent_code1 = args.code1
        if args.code2 is not None:
            latent_code2 = args.code2
        params = (class_label, latent_code1, latent_code2)
        test_generator(generator, params, latent_size=62)
    else:
        build_and_train_models(latent_size=62)
