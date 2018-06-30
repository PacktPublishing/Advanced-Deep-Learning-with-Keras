"""Load CIFAR10 data

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import cifar10

import numpy as np
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
