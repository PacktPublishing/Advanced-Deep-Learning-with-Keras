'''Example of VAE on MNIST dataset

This autoencoder has modular design. The encoder, decoder and autoencoder
are 3 models that share weights. For example, after training the
autoencoder, the encoder can be used to  generate latent vectors
of input data for low-dim visualization like PCA or TSNE.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras
from keras.layers import Activation, Dense, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import losses
from keras.utils import plot_model
from keras import backend as K

import matplotlib.pyplot as plt
from scipy.stats import norm


# MNIST dataset
(x_train, x_test), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 50

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + eps*sqrt(var) 
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.std(K.exp(z_log_var)) * epsilon


# Build the VAE model
# First build the encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of BN-ReLU-Conv2D-MaxPooling2D blocks
for i in range(2):
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same')(x)
    x = MaxPooling2D()(x)

# Shape info needed to build decoder model
shape = K.int_shape(x)


# Generate latent vector (Q(z|X))
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='encoder.png', show_shapes=True)

# Build the decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of BN-ReLU-Transposed Conv2D-UpSampling2D blocks
for i in range(2):
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        padding='same')(x)
    x = UpSampling2D()(x)
    filters //= 2

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='decoder.png', show_shapes=True)

# vae = Model(inputs, outputs, name='vae')
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# Compute VAE loss
xent_loss = losses.mse(
            K.flatten(inputs),
            K.flatten(outputs))
xent_loss *= image_size * image_size 

# print("xent_loss shape: ", K.int_shape(xent_loss))
# print("x shape: ", K.int_shape(x))
# print("x decoded shape: ", K.int_shape(x_decoded_mean))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')


vae.summary()
plot_model(vae, to_file='vae.png', show_shapes=True)

# Train the autoencoder
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))


# display a 2D plot of the digit classes in the latent space
x_test_encoded, _, _ = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# display a 2D manifold of the digits
n = 30 # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
