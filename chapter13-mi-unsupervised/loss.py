"""Collection of loss functions

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras import backend as K
import numpy as np

def iic_mi_loss(batch_size, n_heads):
    def mi_loss(y_true, y_pred):
        """Compute MI loss using IIC MI max algorithm

        """
        n_labels = y_pred.shape[-1]
        # lower half is Z
        Z = y_pred[0: batch_size, :]
        Z = K.expand_dims(Z, axis=2)
        # upper half is Zbar
        Zbar = y_pred[batch_size: y_pred.shape[0], :]
        Zbar = K.expand_dims(Zbar, axis=1)
        # compute joint distribution
        P = K.batch_dot(Z, Zbar)
        P = K.sum(P, axis=0)
        # enforce symmetric joint distribution
        P = (P + K.transpose(P)) / 2.0
        P = P / K.sum(P)
        # marginal distributions
        Pi = K.expand_dims(K.sum(P, axis=1), axis=1)
        Pj = K.expand_dims(K.sum(P, axis=0), axis=0)
        Pi = K.repeat_elements(Pi, rep=n_labels, axis=1)
        Pj = K.repeat_elements(Pj, rep=n_labels, axis=0)
        P = K.clip(P, K.epsilon(), np.finfo(float).max)
        Pi = K.clip(Pi, K.epsilon(), np.finfo(float).max)
        Pj = K.clip(Pj, K.epsilon(), np.finfo(float).max)
        # negative MI loss
        neg_mi = K.sum((P * (K.log(Pi) + K.log(Pj) - K.log(P))))
        # each head contribute 1/n_heads to the total loss
        return neg_mi/n_heads

    return mi_loss
