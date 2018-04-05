'''Trains infoGAN on MNIST using Keras

This version of infoGAN is similar to DCGAN. The difference mainly
is that the z-vector of geneerator is conditioned by a one-hot label
to produce specific fake images. The discriminator is trained to
discriminate real from fake images and predict the corresponding
one-hot labels.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Huang, Xun, et al. "Stacked generative adversarial networks." 
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 
Vol. 2. 2017.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, MaxPooling2D
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

def build_encoder(inputs, num_labels=10, z_dim=146):
    kernel_size = 3
    filters = 64
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu')(inputs)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu')(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)
    fc3_outputs = Dense(z_dim)(y)
    y = Dense(num_labels)(fc3_outputs)
    outputs = Activation('softmax')(y)

    # Build encoder model 
    e0 = Model(inputs=inputs, outputs=fc3_outputs, name="e0")
    e1 = Model(inputs=inputs, outputs=outputs, name="e1")
    return e0, e1


def build_generator(latent_codes, image_size, fc3_dim=146):
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

    y_labels, z0, z1, fc3 = latent_codes
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    g1_inputs = [y_labels, z1] # 10 + 50
    x = keras.layers.concatenate(g1_inputs, axis=1)
    fc3_outputs = Dense(146)(x)
    # Build generator 0 model
    g1 = Model(g1_inputs, fc3_outputs, name='g1')

    g0_inputs = [fc3, z0] # 146 + 50
    x = keras.layers.concatenate(g0_inputs, axis=1)
    x = Reshape((image_resize, image_resize, 4))(x)

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
    # Build generator 0 model
    g0 = Model(g0_inputs, x, name="g0")
    return g0, g1


def build_discriminator0(inputs, z_dim=50):
    """Build a Discriminator 0 Model

    Stacks of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in DCGAN paper.

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        num_labels (int): Dimension of one-hot vector output

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
    y_source = Activation('sigmoid', name='image_source')(y_source)

    # z0 reonstruction (adjust noise to 0 to 1.0 in the inputs)
    z0_recon = Dense(z_dim)(x) 
    z0_recon = Activation('sigmoid', name='z0')(z0_recon)
    
    discriminator_outputs = [y_source, z0_recon]
    d0 = Model(inputs, discriminator_outputs, name='d0')
    return d0


def build_discriminator1(inputs, fc3_dim=146, z_dim=50):
    """Build a Discriminator 1 Model

    # Arguments
        fc3_dim (int): Dimension of one-hot vector output

    # Returns
        Model: Discriminator Model
    """

    x = Dense(256)(inputs)
    x = Dense(256)(x)

    # First output is probability that fc3 is real
    y_source = Dense(1)(x)
    y_source = Activation('sigmoid', name='fc3_source')(y_source)

    # z1 reonstruction (adjust noise to 0 to 1.0 in the inputs)
    z1_recon = Dense(z_dim)(x) 
    z1_recon = Activation('sigmoid', name='z1')(z1_recon)
    
    discriminator_outputs = [y_source, z1_recon]
    d1 = Model(inputs, discriminator_outputs, name='d1')
    return d1


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
    e0, e1, g0, g1, d0, d1, a0, a1 = models
    batch_size, train_steps, num_labels, z_dim, model_name = params
    (x_train, y_train), (x_test, y_test) = data
    save_interval = 500
    z0 = np.random.uniform(0.0, 1.0, size=[16, z_dim])
    z1 = np.random.uniform(0.0, 1.0, size=[16, z_dim])
    # noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    noise_params = [z0, z1]
    for i in range(train_steps):
        # Random real images, labels 
        rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        real_images = x_train[rand_indexes]
        real_fc3 = e0.predict(real_images)
        real_z1 = np.random.uniform(0.0, 1.0, size=[batch_size, z_dim])
        real_labels = y_train[rand_indexes]

        # Generate fake images, labels
        fake_z1 = np.random.uniform(0.0, 1.0, size=[batch_size, z_dim])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]

        generator_inputs = [fake_labels, fake_z1]
        fake_fc3 = g1.predict(generator_inputs)

        fc3 = np.concatenate((real_fc3, fake_fc3))
        z1 = np.concatenate((real_z1, fake_z1))

        # Label real and fake fc3
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        metrics = d1.train_on_batch(fc3, [y, z1])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%d: [d1 loss: %f, acc: %f]" % (i, loss, accuracy)

        # ----- 1 -----
        real_z0 = np.random.uniform(0.0, 1.0, size=[batch_size, z_dim])

        fake_z0 = np.random.uniform(0.0, 1.0, size=[batch_size, z_dim])
        fake_images = g0.predict([fake_fc3, fake_z0])
        
        x = np.concatenate((real_images, fake_images))
        z0 = np.concatenate((real_z0, fake_z0))

        metrics = d0.train_on_batch(x, [y, z0])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s [d0 loss: %f, acc: %f]" % (log, loss, accuracy)

        print(log)
        continue
        




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

def train_encoder(model, data, batch_size=64):
    (x_train, y_train), (x_test, y_test) = data
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              epochs=1,
              batch_size=batch_size)

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))

def build_and_train_models():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = (x_train.astype('float32') - 127.5) / 127.5

    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_test = (x_test.astype('float32') - 127.5) / 127.5

    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model_name = "stackedgan_mnist"
    # Network parameters
    batch_size = 64
    train_steps = 40000
    lr = 0.0002
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )
    z_dim = 50
    z_shape = (z_dim, )
    fc3_dim = 146
    fc3_shape = (fc3_dim, )

    # Build discriminator zero model 
    inputs = Input(shape=input_shape, name='discriminator0_input')
    d0 = build_discriminator0(inputs, z_dim=z_dim)
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # Loss fuctions: 1) Probability image is real 2) MSE z0 recon loss
    loss = ['binary_crossentropy', 'mse']
    d0.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    d0.summary() # image discriminator, z0 discriminator

    # Build discriminator one model
    input_shape = (fc3_dim, )
    inputs = Input(shape=input_shape, name='discriminator1_input')
    d1 = build_discriminator1(inputs, z_dim=z_dim )
    # Loss fuctions: 1) Probability fc3 is real 2) MSE z1 recon loss
    loss = ['binary_crossentropy', 'mse']
    d1.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    d1.summary() # fc3 discriminator, z1 discriminator

    # Build generator models
    fc3 = Input(shape=fc3_shape, name='fc3_input')
    y_labels = Input(shape=label_shape, name='y_labels')
    z1 = Input(shape=z_shape, name="z1_input")
    z0 = Input(shape=z_shape, name="z0_input")
    latent_codes = (y_labels, z0, z1, fc3)
    g0, g1 = build_generator(latent_codes, image_size)
    g0.summary() # image generator 
    g1.summary() # fc3 generator


    # Build encoder models
    input_shape = (image_size, image_size, 1)
    inputs = Input(shape=input_shape, name='encoder_input')
    e0, e1 = build_encoder(inputs, num_labels)
    e0.summary() # fc3 encoder
    e1.summary() # image encoder (classifier)

    # Build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    d0.trainable = False
    generator_inputs = [fc3, z0]
    a0 = Model(generator_inputs,
               d0(g0(generator_inputs)),
               name=model_name)
    a0.compile(loss=loss,
               optimizer=optimizer,
               metrics=['accuracy'])
    a0.summary()

    # g1 = Model(g1_inputs, fc3_outputs, name='g1')
    # g0 = Model([y_labels, z0, z1], x, name="g0")
    d1.trainable = False
    generator_inputs = [y_labels, z1]
    a1 = Model(generator_inputs,
               d1(g1(generator_inputs)),
               name=model_name)
    a1.compile(loss=loss,
               optimizer=optimizer,
               metrics=['accuracy'])
    a1.summary()

    data = (x_train, y_train), (x_test, y_test)
    train_encoder(e1, data)
    e1.trainable = False
    
    # Train discriminator and adversarial networks
    models = (e0, e1, g0, g1, d0, d1, a0, a1)
    params = (batch_size, train_steps, num_labels, z_dim, model_name)
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
        # a = np.linspace(-2, 2, 16)
        # a = np.reshape(a, [16, 1])
        # noise_code1 = np.ones((16, 1)) * a
        # print(noise_code1)

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
        test_generator(generator, params)
    else:
        build_and_train_models()
