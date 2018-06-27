"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import cifar10
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import other_utils


def load_data():
    # load CIFAR10 data
    (x_train, _), (x_test, _) = cifar10.load_data()

    # input image dimensions
    # we assume data format "channels_last"
    rows = x_train.shape[1]
    cols = x_train.shape[2]
    channels = x_train.shape[3]

    # convert color train and test images to gray
    x_train_gray = other_utils.rgb2gray(x_train)
    x_test_gray = other_utils.rgb2gray(x_test)

    # display color version of test images
    imgs = x_test[:100]
    title = 'Test color images (Ground  Truth)'
    img_shape = (rows, cols, channels)
    filename = 'test_color.png'
    other_utils.display_images(imgs,
                               img_shape=img_shape,
                               filename=filename,
                               title=title)

    # display grayscale version of test images
    imgs = x_test_gray[:100]
    title = 'Test gray images (Input)'
    filename = 'test_gray.png'
    other_utils.display_images(imgs,
                               img_shape=(rows, cols, 1),
                               filename=filename,
                               title=title)


    # normalize output train and test color images
    x_train = x_train.astype('float32')  / 255
    x_test = x_test.astype('float32') / 255

    # normalize input train and test grayscale images
    x_train_gray = x_train_gray.astype('float32')  / 255
    x_test_gray = x_test_gray.astype('float32') / 255

    # reshape images to row x col x channels
    # for CNN output/validation
    x_train = x_train.reshape(x_train.shape[0],
                              rows,
                              cols,
                              channels)
    x_test = x_test.reshape(x_test.shape[0],
                            rows,
                            cols,
                            channels)

    # reshape images to row x col x channel for CNN input
    x_train_gray = x_train_gray.reshape(x_train_gray.shape[0],
                                        rows,
                                        cols,
                                        1)
    x_test_gray = x_test_gray.reshape(x_test_gray.shape[0],
                                      rows,
                                      cols,
                                      1)

    # source data, target data, test_source data
    data = (x_train_gray, x_train, x_test_gray)
    gray_shape = (rows, cols, 1)
    color_shape = (rows, cols, channels)
    # source shape, target shape
    shapes = (gray_shape, color_shape)
    
    return data, shapes


