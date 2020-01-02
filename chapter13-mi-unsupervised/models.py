"""Build, train and evaluate an IIC Model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from vgg import VGGBackbone

class IICEncoder(Model):
    def __init__(self, 
                 backbone=None,
                 n_heads=1,
                 n_labels=10,
                 name="IIC_encoder",
                 **kwargs):
        """Build an IICEncoder using VGGBackbone

        Arguments:
            n_heads (int): Number of heads
            n_labels (int): Number of object classes
            name (string): Model name
        """
        super(IICEncoder, self).__init__(name=name, **kwargs)
        if backbone is None:
            raise ValueError("Please indicate the backbone network")
        self.backbone = backbone
        self._layers = []
        for i in range(n_heads):
            name = "z_head%d" % i
            self._layers.append(Dense(n_labels,
                                      activation='softmax',
                                      name=name))


    # build the n_heads of the IIC model
    def call(self, inputs):
        x = self.backbone(inputs)
        x = Flatten()(x)
        outputs = []
        for layer in self._layers:
            outputs.append(layer(x))
        
        return outputs


if __name__ == '__main__':
    encoder = IICEncoder(backbone=VGGBackbone())
    input_shape = (1, 24, 24, 1)
    encoder.build(input_shape)
    encoder.summary()
