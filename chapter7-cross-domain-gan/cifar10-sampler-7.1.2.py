'''
Demonstrates how to sample and plot CIFAR10 images
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import other_utils
import math

# load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# sample cifar10 from train dataset
size = 1
side = int(math.sqrt(size))
indexes = np.random.randint(0, x_train.shape[0], size=size)
images = x_train[indexes]
gray_images = other_utils.rgb2gray(x_train[indexes])

# plot color cifar10
plt.figure(figsize=(side,side))
for i in range(len(indexes)):
    plt.subplot(side, side, i + 1)
    image = images[i]
    plt.imshow(image)
    plt.axis('off')

plt.savefig("cifar10-color-samples.png")
plt.show()
plt.close('all')

# plot gray cifar10
plt.figure(figsize=(side,side))
for i in range(len(indexes)):
    plt.subplot(side, side, i + 1)
    image = gray_images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.savefig("cifar10-gray-samples.png")
plt.show()
plt.close('all')
