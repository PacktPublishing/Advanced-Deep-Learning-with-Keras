'''
Demonstrates how to sample and plot MNIST digits
using Keras API

Project: https://github.com/roatienza/dl-keras
Usage: python3 <this file>
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# numpy package
import numpy as np

# keras mnist module
from keras.datasets import mnist

# for plotting
import matplotlib.pyplot as plt

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))

# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("Test labels: ", dict(zip(unique, counts)))

# sample 10 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=10)
images = x_train[indexes]
labels = y_train[indexes]
# plot the 10 mnist digits
for i in range(len(indexes)):
    filename = "mnist%d.png" % labels[i]
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.savefig(filename)
    plt.show()

plt.close('all')
