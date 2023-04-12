''' Using Functional API to build CNN

~99.3% test accuracy
'''

# Import required libraries
from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset and prepare it
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1]).astype('float32') / 255
x_test = np.reshape(x_test,[-1, image_size, image_size, 1]).astype('float32') / 255

# Set network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# Build CNN layers with functional API
inputs = Input(shape=input_shape)
y = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
y = Flatten()(y)
y = Dropout(dropout)(y)
outputs = Dense(num_labels, activation='softmax')(y)

# Build the model and show summary
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with input images and labels
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=batch_size)

# Calculate and print model accuracy on test dataset
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("Test accuracy: %.1f%%" % (100.0 * score[1]))
