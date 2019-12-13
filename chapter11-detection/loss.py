"""Loss functions for object detection

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Huber

import numpy as np

def focal_loss_ce(y_true, y_pred):
    """Alternative CE focal loss (not used)
    """
    # only missing in this FL is y_pred clipping
    weight = (1 - y_pred)
    weight *= weight
    # alpha = 0.25
    weight *= 0.25
    return K.categorical_crossentropy(weight*y_true, y_pred)


def focal_loss_binary(y_true, y_pred):
    """Binary cross-entropy focal loss
    """
    gamma = 2.0
    alpha = 0.25

    pt_1 = tf.where(tf.equal(y_true, 1),
                    y_pred,
                    tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0),
                    y_pred,
                    tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN and Inf
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    weight = alpha * K.pow(1. - pt_1, gamma)
    fl1 = -K.sum(weight * K.log(pt_1))
    weight = (1 - alpha) * K.pow(pt_0, gamma)
    fl0 = -K.sum(weight * K.log(1. - pt_0))

    return fl1 + fl0


def focal_loss_categorical(y_true, y_pred):
    """Categorical cross-entropy focal loss"""
    gamma = 2.0
    alpha = 0.25

    # scale to ensure sum of prob is 1.0
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # clip the prediction value to prevent NaN and Inf
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # calculate cross entropy
    cross_entropy = -y_true * K.log(y_pred)

    # calculate focal loss
    weight = alpha * K.pow(1 - y_pred, gamma)
    cross_entropy *= weight

    return K.sum(cross_entropy, axis=-1)


def mask_offset(y_true, y_pred): 
    """Pre-process ground truth and prediction data"""
    # 1st 4 are offsets
    offset = y_true[..., 0:4]
    # last 4 are mask
    mask = y_true[..., 4:8]
    # pred is actually duplicated for alignment
    # either we get the 1st or last 4 offset pred
    # and apply the mask
    pred = y_pred[..., 0:4]
    offset *= mask
    pred *= mask
    return offset, pred


def l1_loss(y_true, y_pred):
    """MAE or L1 loss
    """
    offset, pred = mask_offset(y_true, y_pred)
    # we can use L1
    return K.mean(K.abs(pred - offset), axis=-1)


def smooth_l1_loss(y_true, y_pred):
    """Smooth L1 loss using tensorflow Huber loss
    """
    offset, pred = mask_offset(y_true, y_pred)
    # Huber loss as approx of smooth L1
    return Huber()(offset, pred)
