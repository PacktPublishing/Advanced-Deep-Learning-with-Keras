"""Helper function for building FCN model.
Utilities for building network layers are also provided.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Softmax, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np

def conv_layer(inputs,
               filters=32,
               kernel_size=3,
               strides=1,
               use_maxpool=True,
               postfix=None,
               activation=None):
    """Helper function to build Conv2D-BN-ReLU layer
        with optional MaxPooling2D.
    """

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               kernel_initializer='he_normal',
               name="conv_"+postfix,
               padding='same')(inputs)
    x = BatchNormalization(name="bn_"+postfix)(x)
    x = Activation('relu', name='relu_'+postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool'+postfix)(x)
    return x


def tconv_layer(inputs,
                filters=32,
                kernel_size=3,
                strides=2,
                postfix=None):
    """Helper function to build Conv2DTranspose-BN-ReLU 
        layer
    """
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        kernel_initializer='he_normal',
                        name='tconv_'+postfix)(inputs)
    x = BatchNormalization(name="bn_"+postfix)(x)
    x = Activation('relu', name='relu_'+postfix)(x)
    return x


def build_fcn(input_shape,
              backbone,
              n_classes=4):
    """Helper function to build an FCN model.
        
    Arguments:
        backbone (Model): A backbone network
            such as ResNetv2 or v1
        n_classes (int): Number of object classes
            including background.
    """

    inputs = Input(shape=input_shape)
    features = backbone(inputs)

    main_feature = features[0]
    features = features[1:]
    out_features = [main_feature]
    feature_size = 8
    size = 2
    # other half of the features pyramid
    # including upsampling to restore the
    # feature maps to the dimensions
    # equal to 1/4 the image size
    for feature in features:
        postfix = "fcn_" + str(feature_size)
        feature = conv_layer(feature,
                             filters=256,
                             use_maxpool=False,
                             postfix=postfix)
        postfix = postfix + "_up2d"
        feature = UpSampling2D(size=size,
                               interpolation='bilinear',
                               name=postfix)(feature)
        size = size * 2
        feature_size = feature_size * 2
        out_features.append(feature)

    # concatenate all upsampled features
    x = Concatenate()(out_features)
    # perform 2 additional feature extraction 
    # and upsampling
    x = tconv_layer(x, 256, postfix="up_x2")
    x = tconv_layer(x, 256, postfix="up_x4")
    # generate the pixel-wise classifier
    x = Conv2DTranspose(filters=n_classes,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        name="pre_activation")(x)
    x = Softmax(name="segmentation")(x)

    model = Model(inputs, x, name="fcn")

    return model

