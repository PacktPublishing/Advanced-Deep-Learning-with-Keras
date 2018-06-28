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
    (target_data, _), (test_target_data, _) = cifar10.load_data()

    # input image dimensions
    # we assume data format "channels_last"
    rows = target_data.shape[1]
    cols = target_data.shape[2]
    channels = target_data.shape[3]

    # convert color train and test images to gray
    source_data = other_utils.rgb2gray(target_data)
    test_source_data = other_utils.rgb2gray(test_target_data)
    # reshape images to row x col x channel for CNN input
    source_data = source_data.reshape(source_data.shape[0],
                                      rows,
                                      cols,
                                      1)
    test_source_data = test_source_data.reshape(test_source_data.shape[0],
                                                rows,
                                                cols,
                                                1)

    # source data, target data, test_source data
    data = (source_data, target_data, test_source_data, test_target_data)
    filenames = ('cifar10_test_source.png', 'cifar10_test_target.png')
    titles = ('CIFAR10 test source images', 'CIFAR10 test target images')
    
    return other_utils.load_data(data, titles, filenames)


def _load_data():
    # load CIFAR10 data
    (target_data, _), (test_target_data, _) = cifar10.load_data()

    # input image dimensions
    # we assume data format "channels_last"
    rows = target_data.shape[1]
    cols = target_data.shape[2]
    channels = target_data.shape[3]

    # convert color train and test images to gray
    source_data = other_utils.rgb2gray(target_data)
    test_source_data = other_utils.rgb2gray(test_target_data)

    # display color version of test images
    imgs = test_target_data[:100]
    title = 'CIFAR10 test color images (Ground  Truth)'
    img_shape = (rows, cols, channels)
    filename = 'test_color.png'
    other_utils.display_images(imgs,
                               img_shape=img_shape,
                               filename=filename,
                               title=title)

    # display grayscale version of test images
    imgs = test_source_data[:100]
    title = 'CIFAR10 test gray images (Input)'
    filename = 'test_gray.png'
    other_utils.display_images(imgs,
                               img_shape=(rows, cols, 1),
                               filename=filename,
                               title=title)


    # normalize output train and test color images
    target_data = target_data.astype('float32')  / 255
    test_target_data = test_target_data.astype('float32') / 255

    # normalize input train and test grayscale images
    source_data = source_data.astype('float32')  / 255
    test_source_data = test_source_data.astype('float32') / 255

    # reshape images to row x col x channels
    # for CNN output/validation
    target_data = target_data.reshape(target_data.shape[0],
                              rows,
                              cols,
                              channels)
    test_target_data = test_target_data.reshape(test_target_data.shape[0],
                            rows,
                            cols,
                            channels)

    # reshape images to row x col x channel for CNN input
    source_data = source_data.reshape(source_data.shape[0],
                                      rows,
                                      cols,
                                      1)
    test_source_data = test_source_data.reshape(test_source_data.shape[0],
                                                rows,
                                                cols,
                                                1)

    # source data, target data, test_source data
    data = (source_data, target_data, test_source_data, test_target_data)
    source_shape = (rows, cols, 1)
    target_shape = (rows, cols, channels)
    shapes = (source_shape, target_shape)
    
    return data, shapes


