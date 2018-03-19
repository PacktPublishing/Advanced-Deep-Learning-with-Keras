'''
Implementing a Y-Network using Functional API

Project: https://github.com/roatienza/dl-keras
Dependencies: keras 
Usage: python3 <this file>
'''

import numpy as np
import keras
from keras.layers import Activation, Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels = np.amax(y_train) + 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
dropout = 0.4
n_filters = 32

# Left branch of Y network
inputs1 = Input(shape=input_shape)
x = inputs1
filters = n_filters
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters = 2 * filters

# Right branch of Y network
inputs2 = Input(shape=input_shape)
y = inputs2
filters = n_filters
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               dilation_rate=2)(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters = 2 * filters

# Merge left and right branches outputs
y = keras.layers.concatenate([x, y])
y = Flatten()(y)
y = Dropout(dropout)(y)
y = Dense(num_labels)(y)
outputs = Activation('softmax')(y)

model = Model([inputs1, inputs2], outputs)
plot_model(model, to_file='cnn-y-network.png', show_shapes=True)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit([x_train, x_train],
          y_train, 
          validation_data=([x_test, x_test], y_test),
          epochs=20,
          batch_size=batch_size)

score = model.evaluate([x_test, x_test], y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
