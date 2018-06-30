'''
Demonstrates how to sample and plot CIFAR10 images
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import mnist_svhn_utils
import math

# load dataset
data, _ = mnist_svhn_utils.load_data()
_, svhn, _, _ = data

# sample svhn from train dataset
size = 1
side = int(math.sqrt(size))
indexes = np.random.randint(0, svhn.shape[0], size=size)
images = svhn[indexes]

# plot color cifar10
plt.figure(figsize=(side,side))
for i in range(len(indexes)):
    plt.subplot(side, side, i + 1)
    image = images[i]
    plt.imshow(image)
    plt.axis('off')

plt.savefig("svhn-samples.png")
plt.show()
plt.close('all')
