"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import mnist
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import io
import other_utils


def load_data():
    # load mnist data
    (source_data, _), (test_source_data, _) = mnist.load_data()


    source_data = np.pad(source_data,
                         ((0,0), (2,2), (2,2)),
                         'constant',
                         constant_values=0)
    test_source_data = np.pad(test_source_data,
                              ((0,0), (2,2), (2,2)),
                              'constant',
                              constant_values=0)
    print(source_data.shape)
    print(test_source_data.shape)
    # input image dimensions
    # we assume data format "channels_last"
    rows = source_data.shape[1]
    cols = source_data.shape[2]
    channels = 1

    source_shape = (rows, cols, channels)

    # display test images
    imgs = test_source_data[:100]
    title = 'Test mnist source images' 
    img_shape = (rows, cols, channels)
    filename = 'mnist_source.png'
    other_utils.display_images(imgs,
                               img_shape=img_shape,
                               filename=filename,
                               title=title)

    # normalize mnist data
    source_data = source_data.astype('float32')  / 255
    test_source_data = test_source_data.astype('float32') / 255

    # reshape images to row x col x channels
    # for CNN output/validation
    source_data = source_data.reshape(source_data.shape[0],
                                      rows,
                                      cols,
                                      channels)
    test_source_data = test_source_data.reshape(test_source_data.shape[0],
                                                rows,
                                                cols,
                                                channels)

    # load SVHN dataset
    target_mat = io.loadmat("datasets/train_32x32.mat")
    target_data = target_mat['X']
    print(target_data.shape)
    rows = target_data.shape[0]
    cols = target_data.shape[1]
    channels = target_data.shape[2]
    target_data = np.transpose(target_data, (3, 0, 1, 2))
    print(target_data.shape)
    imgs = target_data[:100]
    print(imgs.shape)
    title = 'Target data images'
    filename = 'svhn_target.png'
    other_utils.display_images(imgs,
                               img_shape=(rows, cols, channels),
                               filename=filename,
                               title=title)
    target_data = target_data.astype('float32')  / 255
    target_shape = (rows, cols, channels)

    # source data, target data, test_source data
    data = (source_data, target_data, test_source_data)
    # source shape, target shape
    shapes = (source_shape, target_shape)
    
    return data, shapes
