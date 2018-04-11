'''Trains StackedGAN on MNIST using Keras

Stacked GAN uses Encoder, Generator and Discriminator.
The encoder is a CNN MNIST classifier. The encoder provides latent
features (fc3) and labels that the generator learns by inverting the 
process. The generator uses conditioning labels and latent codes
(z0 and z1) to synthesize images by fooling the discriminator.
The labels, z0 and z1 are disentangled codes used to control 
the attributes of synthesized images. The discriminator determines 
if the image and fc3 features are real or fake. At the same time,
it estimates the latent codes that generated the image and fc3 features.

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

def build_encoder(inputs, num_labels=10, fc3_dim=256):
    """ Build the Classifier (Encoder) Model sub networks

    Two sub networks: 1) Image to fc3 (intermediate latent feature)
    2) fc3 to classes

    # Arguments
        inputs (Layers): x - images, fc3 - fc3 layer output
        num_labels (int): number of classes
        fc3_dim (int): fc3 dimensionality

    # Returns
        e0, e1 (Models): Description below 
    """
    kernel_size = 3
    filters = 64

    x, fc3 = inputs
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu')(x)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu')(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)
    fc3_output = Dense(fc3_dim, activation='relu')(y)

    y = Dense(num_labels)(fc3)
    labels = Activation('softmax')(y)

    # Build encoder models 
    # e0: image to fc3 
    e0 = Model(inputs=x, outputs=fc3_output, name="e0")
    # e1: fc3 to classes
    e1 = Model(inputs=fc3, outputs=labels, name="e1")
    return e0, e1


def build_generator(latent_codes, image_size, fc3_dim=256):
    """Build Generator Model sub networks

    Two sub networks: 1) Class and noise to fc3 (intermediate feature)
    2) fc3 to image

    # Arguments
        latent_codes (Layers): dicrete code (labels), noise and fc3 features
        image_size (int): Target size of one side (assuming square image)
        fc3_dim (int): fc3 dimensionality

    # Returns
        g0, g1 (Models): Description below
    """

    # Latent codes and network parameters
    y_labels, z0, z1, fc3 = latent_codes
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    # Inputs to g1
    g1_inputs = [y_labels, z1]      # 10 + 50
    x = keras.layers.concatenate(g1_inputs, axis=1)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    fc3_outputs = Dense(fc3_dim, activation='relu')(x)
    # g1: classes and noise to fc3
    g1 = Model(g1_inputs, fc3_outputs, name='g1')

    # Inputs to g0
    g0_inputs = [fc3, z0]           # 256 + 50
    x = keras.layers.concatenate(g0_inputs, axis=1)
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
    # g0: fc3 features and noise to image
    g0 = Model(g0_inputs, x, name="g0")

    return g0, g1


def build_discriminator0(inputs, z_dim=50):
    """Build a Discriminator 0 Model

    Classifies x (image) as real/fake image and recovers
    the input noise or latent code (by minimizing entropy loss)

    # Arguments
        inputs (Layer): image
        z_dim (int): noise dimensionality

    # Returns
        d0 (Model): image x as real/fake and recovered latent code
    """

    # Network parameters
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

    # z0 reonstruction (Q0 network)
    z0_recon = Dense(z_dim)(x) 
    z0_recon = Activation('tanh', name='z0')(z0_recon)
    
    discriminator_outputs = [y_source, z0_recon]
    d0 = Model(inputs, discriminator_outputs, name='d0')
    return d0


def build_discriminator1(inputs, z_dim=50):
    """Build a Discriminator 1 Model

    Classifies fc3 (features) as real/fake image and recovers
    the input noise or latent code (by minimizing entropy loss)

    # Arguments
        inputs (Layer): fc3 features
        z_dim (int): noise dimensionality

    # Returns
        d1 (Model): fc3 as real/fake and recovered latent code
    """

    x = Dense(256, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)

    # First output is probability that fc3 is real
    y_source = Dense(1)(x)
    y_source = Activation('sigmoid', name='fc3_source')(y_source)

    # z1 reonstruction (Q1 network)
    z1_recon = Dense(z_dim)(x) 
    z1_recon = Activation('tanh', name='z1')(z1_recon)
    
    discriminator_outputs = [y_source, z1_recon]
    d1 = Model(inputs, discriminator_outputs, name='d1')
    return d1


def train(models, data, params):
    """Train the discriminator and adversarial Networks

    Alternately train discriminator and adversarial networks by batch.
    Discriminator is trained first with real and fake images,
    corresponding one-hot labels and latent codes.
    Adversarial is trained next with fake images pretending to be real,
    corresponding one-hot labels and latent codes.
    Generate sample images per save_interval.

    # Arguments
        models (Models): Encoder, Generator, Discriminator, Adversarial models
        data (tuple): x_train, y_train data
        params (tuple): Network parameters

    """
    e0, e1, g0, g1, d0, d1, a0, a1 = models
    batch_size, train_steps, num_labels, z_dim, model_name = params
    (x_train, y_train), (_, _) = data
    save_interval = 500

    # label and noise codes for generator testing
    z0 = np.random.normal(scale=0.5, size=[16, z_dim])
    z1 = np.random.normal(scale=0.5, size=[16, z_dim])
    noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    noise_params = [noise_class, z0, z1]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    for i in range(train_steps):
        # Stack 1
        # Random real data
        rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        real_images = x_train[rand_indexes]
        real_fc3 = e0.predict(real_images)
        real_z1 = np.random.normal(0.5, size=[batch_size, z_dim])
        real_labels = y_train[rand_indexes]

        # Generate fake data
        # Joint
        fake_z1 = np.random.normal(0.5, size=[batch_size, z_dim])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_fc3 = g1.predict([fake_labels, fake_z1])

        # real + fake data
        fc3 = np.concatenate((real_fc3, fake_fc3))
        z1 = np.concatenate((real_z1, fake_z1))

        # Label 1st half as real and 2nd half as fake
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        # Train discriminator 1 to classify fc3 as real/fake and recover
        # latent code. real = from encoder, fake = from g1 
        metrics = d1.train_on_batch(fc3, [y, z1])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%d: [d1 loss: %f, acc: %f]" % (i, loss, accuracy)

        # Stack 0
        real_z0 = np.random.normal(0.5, size=[batch_size, z_dim])
        fake_z0 = np.random.normal(0.5, size=[batch_size, z_dim])
        # Joint 
        fake_images = g0.predict([fake_fc3, fake_z0])
       
        # real + fake data
        x = np.concatenate((real_images, fake_images))
        z0 = np.concatenate((real_z0, fake_z0))

        # Train discriminator 0 to classify image as real/fake and recover
        # latent code
        metrics = d0.train_on_batch(x, [y, z0])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s [d0 loss: %f, acc: %f]" % (log, loss, accuracy)

        # Adversarial training 
        # Generate fake z1, labels
        fake_z1 = np.random.normal(0.5, size=[batch_size, z_dim])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        g1_inputs = [fake_labels, fake_z1]

        # Label fake fc3 as real
        y = np.ones([batch_size, 1])
    
        # Train generator 1 (thru adversarial) by fooling the discriminator
        # and approximating encoder 1 fc3 features generator
        metrics = a1.train_on_batch(g1_inputs, [y, fake_z1, fake_labels])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s [a1 loss: %f, acc: %f]" % (log, loss, accuracy)

        # Generate fake fc3 and noise
        fake_fc3 = g1.predict([fake_labels, fake_z1])
        fake_z0 = np.random.normal(0.5, size=[batch_size, z_dim])
        g0_inputs = [fake_fc3, fake_z0]

        # Train generator 0 (thru adversarial) by fooling the discriminator
        # and approximating encoder 1 image source generator
        metrics = a0.train_on_batch(g0_inputs, [y, fake_z0, fake_fc3])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s [a0 loss: %f, acc: %f]" % (log, loss, accuracy)

        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                g1.save(model_name + "-g1.h5")
                g0.save(model_name + "-g0.h5")
                show = True
            else:
                show = False
            generators = (g0, g1)
            plot_images(generators,
                        noise_params=noise_params,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)
    

def plot_images(generators,
                noise_params,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    # Arguments
        generators (Models): g0 and g1 models for fake images generation
        noise_params (list): noise parameters (label, z0 and z1 codes)
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name
    """

    g0, g1 = generators
    noise_class, z0, z1 = noise_params
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    fc3 = g1.predict([noise_class, z1])
    images = g0.predict([fc3, z0])
    images = (images + 1.0) * 0.5 
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_class.shape[0]))
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


def train_encoder(model, data, model_name="stackedgan_mnist", batch_size=64):
    """ Train the Encoder Model (e0 and e1)

    # Arguments
        model (Model): Encoder
        data (tensor): Train and test data
        model_name (string): model name
        batch_size (int): Train batch size
    """

    (x_train, y_train), (x_test, y_test) = data
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              epochs=10,
              batch_size=batch_size)

    model.save(model_name + "-encoder.h5")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))


def build_and_train_models(encoder_saved_model):
    """ Build and train Stacked GAN
    
    # Arguments
        encoder_saved_model (h5): use trained model if available
    """

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
    fc3_dim = 256
    fc3_shape = (fc3_dim, )

    # Build discriminator zero model 
    inputs = Input(shape=input_shape, name='discriminator0_input')
    d0 = build_discriminator0(inputs, z_dim=z_dim)
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # Loss fuctions: 1) Probability image is real 2) MSE z0 recon loss
    loss = ['binary_crossentropy', 'mse']
    d0.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    d0.summary() # image discriminator, z0 estimator 

    # Build discriminator one model
    input_shape = (fc3_dim, )
    inputs = Input(shape=input_shape, name='discriminator1_input')
    d1 = build_discriminator1(inputs, z_dim=z_dim )
    # Loss fuctions: 1) Probability fc3 is real 2) MSE z1 recon loss
    loss = ['binary_crossentropy', 'mse']
    d1.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    d1.summary() # fc3 discriminator, z1 estimator

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
    e0, e1 = build_encoder((inputs, fc3), num_labels)
    e0.summary() # image to fc3 encoder
    e1.summary() # fc3 to labels encoder (classifier)
    encoder = Model(inputs, e1(e0(inputs)))
    encoder.summary() # image to labels encoder (classifier)

    data = (x_train, y_train), (x_test, y_test)
    # Train or load encoder saved model
    if encoder_saved_model is not None:
        encoder = load_model(encoder_saved_model)
    else:
        train_encoder(encoder, data, model_name=model_name)

    # Build adversarial0 model = generator0 + discriminator0 + encoder0
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    e0.trainable = False
    d0.trainable = False
    g0_inputs = [fc3, z0]
    g0_outputs = g0(g0_inputs)
    a0_outputs = d0(g0_outputs) + [e0(g0_outputs)]
    a0 = Model(g0_inputs, a0_outputs, name="a0")
    loss = ['binary_crossentropy', 'mse', 'mse']
    a0.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    a0.summary()

    # Build adversarial1 model = generator1 + discriminator1 + encoder1
    e1.trainable = False
    d1.trainable = False
    g1_inputs = [y_labels, z1]
    g1_outputs = g1(g1_inputs)
    a1_outputs = d1(g1_outputs) + [e1(g1_outputs)]
    a1 = Model(g1_inputs, a1_outputs, name="a1")
    loss = ['binary_crossentropy', 'mse', 'categorical_crossentropy']
    a1.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    a1.summary()

    # Train discriminator and adversarial networks
    models = (e0, e1, g0, g1, d0, d1, a0, a1)
    params = (batch_size, train_steps, num_labels, z_dim, model_name)
    train(models, data, params)


def test_generator(generators, params, z_dim=50):
    class_label, z0, z1 = params
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    if z0 is None:
        z0 = np.random.normal(0.5, size=[16, z_dim])
    else:
        z0 = np.ones((16, z_dim)) * z0
        # a = np.linspace(-2, 2, 16)
        # a = np.reshape(a, [16, 1])
        # z0 = np.ones((16, z_dim)) * a
        print("z0: ", z0[:,0])

    if z1 is None:
        z1 = np.random.normal(0.5, size=[16, z_dim])
    else:
        z1 = np.ones((16, z_dim)) * z1
        # a = np.linspace(-2, 2, 16)
        # a = np.reshape(a, [16, 1])
        # z1 = np.ones((16, z_dim)) * a
        print("z1: ", z1[:,0])

    noise_params = [noise_class, z0, z1]

    plot_images(generators,
                noise_params=noise_params,
                show=True,
                step=step,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator 0 h5 model with trained weights"
    parser.add_argument("-g", "--generator0", help=help_)
    help_ = "Load generator 1 h5 model with trained weights"
    parser.add_argument("-k", "--generator1", help=help_)
    help_ = "Load encoder h5 model with trained weights"
    parser.add_argument("-e", "--encoder", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    help_ = "Specify z0 noise code (as a 50-dim with z0 constant)"
    parser.add_argument("-z", "--z0", type=float, help=help_)
    help_ = "Specify z1 noise code (as a 50-dim with z1 constant)"
    parser.add_argument("-x", "--z1", type=float, help=help_)
    args = parser.parse_args()
    if args.encoder:
        encoder = args.encoder
    else:
        encoder = None
    if args.generator0:
        g0 = load_model(args.generator0)
        if args.generator1:
            g1 = load_model(args.generator1)
        else:
            print("Must specify both generators 0 and 1 models")
            exit(0)
        class_label = None
        z0 = None
        z1 = None
        if args.digit is not None:
            class_label = args.digit
        if args.z0 is not None:
            z0 = args.z0
        if args.z1 is not None:
            z1 = args.z1
        params = (class_label, z0, z1)
        test_generator((g0, g1), params)
    else:
        build_and_train_models(encoder)
