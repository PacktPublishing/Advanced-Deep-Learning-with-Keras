"""Utility functions

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.keras.callbacks import Callback

def print_log(param, verbose=0):
    if verbose > 0:
        print(param)

class AccuracyCallback(Callback):
    """Callback to compute the accuracy every epoch by
        calling the eval() method.

    Argument:
        net (Model): Object with a network model to evaluate. 
            Must support the eval() method.
    """
    def __init__(self, net):
        super(AccuracyCallback, self).__init__()
        self.net = net 

    def on_epoch_end(self, epoch, logs=None):
        self.net.eval()


