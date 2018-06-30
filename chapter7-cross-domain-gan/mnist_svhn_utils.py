"""Utilities for loading MNIST and SVHN

Street View House Number (SVHN) dataset:
http://ufldl.stanford.edu/housenumbers/

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, 
Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with 
Unsupervised Feature Learning NIPS Workshop on Deep Learning 
and Unsupervised Feature Learning 2011. (PDF)


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import mnist

import numpy as np
from scipy import io
import other_utils


def load_data():
    # load mnist data
    (source_data, _), (test_source_data, _) = mnist.load_data()

    # pad with zeros 28x28 MNIST image to become 32x32
    # svhn is 32x32
    source_data = np.pad(source_data,
                         ((0,0), (2,2), (2,2)),
                         'constant',
                         constant_values=0)
    test_source_data = np.pad(test_source_data,
                              ((0,0), (2,2), (2,2)),
                              'constant',
                              constant_values=0)
    # input image dimensions
    # we assume data format "channels_last"
    rows = source_data.shape[1]
    cols = source_data.shape[2]
    channels = 1

    # reshape images to row x col x channels
    # for CNN output/validation
    size = source_data.shape[0]
    source_data = source_data.reshape(size,
                                      rows,
                                      cols,
                                      channels)
    size = test_source_data.shape[0]
    test_source_data = test_source_data.reshape(size,
                                                rows,
                                                cols,
                                                channels)

    # load SVHN data
    target_data = loadmat("datasets/train_32x32.mat")
    test_target_data = loadmat("datasets/test_32x32.mat")

    # source data, target data, test_source data
    data = (source_data, target_data, test_source_data, test_target_data)
    filenames = ('mnist_test_source.png', 'svhn_test_target.png')
    titles = ('MNIST test source images', 'SVHN test target images')
    
    return other_utils.load_data(data, titles, filenames)


def loadmat(filename):
    # load SVHN dataset
    mat = io.loadmat(filename)
    # the key to image data is 'X', the image label key is 'y'
    data = mat['X']
    rows =data.shape[0]
    cols = data.shape[1]
    channels = data.shape[2]
    # in matlab data, the image index is the last index
    # in keras, the image index is the first index so
    # perform transpose for the last index
    data = np.transpose(data, (3, 0, 1, 2))
    return data
