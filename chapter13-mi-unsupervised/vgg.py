"""VGG backbone creator

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

# A to E are standard VGG backbones
# F was customized for IIC
# G is experimental
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',512, 512, 512, 512, 'M'],
    'F': [64, 'M', 128, 'M', 256, 'M', 512],
    'G': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'A'],
}

class VGGBlock(Layer):
    def __init__(self, 
                 n_filters,
                 name,
                 maxpool=True,
                 **kwargs):
        """A VGG block is made of Conv2D-BN-ReLU.
        Maxpooling2D is optional at the last layer.

        Arguments:
            n_filters (int): Number of Conv2D filters
            name (string): Block name
            maxpool (Bool): Use MaxPoolinng2D as the
                last layer.
        """
        super(VGGBlock, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(n_filters,
                           kernel_size=3,
                           padding='same',
                           kernel_initializer='he_normal')
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        if maxpool:
            self.maxp = MaxPooling2D()
        else:
            self.maxp = None


    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        if self.maxp is not None:
            x = self.maxp(x)
        return x
    

class VGGBackbone(Model):
    def __init__(self, 
                 name='VGG_backbone', 
                 **kwargs):
        """VGG backbone object creator.
        This is a simplified VGG made of 4 conv for
        feature extraction.
            
        Arguments
            name (string): Layer name
        """
        super(VGGBackbone, self).__init__(name=name, **kwargs)
        self.block1 = VGGBlock(64 , "VGG_block1")
        self.block2 = VGGBlock(128, "VGG_block2")
        self.block3 = VGGBlock(256, "VGG_block3")
        self.block4 = VGGBlock(512, "VGG_block4", maxpool=False)


    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


if __name__ == '__main__':
    backbone = VGGBackbone()
    y = backbone(tf.ones(shape=(1, 24, 24, 1)))
    backbone.summary()
    #print(len(backbone.weights))
