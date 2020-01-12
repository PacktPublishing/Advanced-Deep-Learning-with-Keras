"""SSD model builder
Utilities for building network layers are also provided
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import ELU, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np

def conv2d(inputs,
           filters=32,
           kernel_size=3,
           strides=1,
           name=None):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer='he_normal',
                  name=name,
                  padding='same')

    return conv(inputs)


def conv_layer(inputs,
               filters=32,
               kernel_size=3,
               strides=1,
               use_maxpool=True,
               postfix=None,
               activation=None):

    x = conv2d(inputs,
               filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               name='conv'+postfix)
    x = BatchNormalization(name="bn"+postfix)(x)
    x = ELU(name='elu'+postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool'+postfix)(x)
    return x


def build_ssd(input_shape,
              backbone,
              n_layers=4,
              n_classes=4,
              aspect_ratios=(1, 2, 0.5)):
    """Build SSD model given a backbone

    Arguments:
        input_shape (list): input image shape
        backbone (model): Keras backbone model
        n_layers (int): Number of layers of ssd head
        n_classes (int): Number of obj classes
        aspect_ratios (list): annchor box aspect ratios

    Returns:
        n_anchors (int): Number of anchor boxes per feature pt
        feature_shape (tensor): SSD head feature maps
        model (Keras model): SSD model
    """
    # number of anchor boxes per feature map pt
    n_anchors = len(aspect_ratios) + 1

    inputs = Input(shape=input_shape)
    # no. of base_outputs depends on n_layers
    base_outputs = backbone(inputs)
    
    outputs = []
    feature_shapes = []
    out_cls = []
    out_off = []

    for i in range(n_layers):
        # each conv layer from backbone is used
        # as feature maps for class and offset predictions
        # also known as multi-scale predictions
        conv = base_outputs if n_layers==1 else base_outputs[i]
        name = "cls" + str(i+1)
        classes  = conv2d(conv,
                          n_anchors*n_classes,
                          kernel_size=3,
                          name=name)

        # offsets: (batch, height, width, n_anchors * 4)
        name = "off" + str(i+1)
        offsets  = conv2d(conv,
                          n_anchors*4,
                          kernel_size=3,
                          name=name)

        shape = np.array(K.int_shape(offsets))[1:]
        feature_shapes.append(shape)

        # reshape the class predictions, yielding 3D tensors of 
        # shape (batch, height * width * n_anchors, n_classes)
        # last axis to perform softmax on them
        name = "cls_res" + str(i+1)
        classes = Reshape((-1, n_classes), 
                          name=name)(classes)

        # reshape the offset predictions, yielding 3D tensors of
        # shape (batch, height * width * n_anchors, 4)
        # last axis to compute the (smooth) L1 or L2 loss
        name = "off_res" + str(i+1)
        offsets = Reshape((-1, 4),
                          name=name)(offsets)
        # concat for alignment with ground truth size
        # made of ground truth offsets and mask of same dim
        # needed during loss computation
        offsets = [offsets, offsets]
        name = "off_cat" + str(i+1)
        offsets = Concatenate(axis=-1,
                              name=name)(offsets)

        # collect offset prediction per scale
        out_off.append(offsets)

        name = "cls_out" + str(i+1)

        #activation = 'sigmoid' if n_classes==1 else 'softmax'
        #print("Activation:", activation)

        classes = Activation('softmax',
                             name=name)(classes)

        # collect class prediction per scale
        out_cls.append(classes)

    if n_layers > 1:
        # concat all class and offset from each scale
        name = "offsets"
        offsets = Concatenate(axis=1,
                              name=name)(out_off)
        name = "classes"
        classes = Concatenate(axis=1,
                              name=name)(out_cls)
    else:
        offsets = out_off[0]
        classes = out_cls[0]

    outputs = [classes, offsets]
    model = Model(inputs=inputs,
                  outputs=outputs,
                  name='ssd_head')

    return n_anchors, feature_shapes, model
