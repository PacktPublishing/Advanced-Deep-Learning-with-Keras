"""Project config

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
        'epoch_offset': 0,
        'classes' : ["background", "Water", "Soda", "Juice"],
        'prices' : [0.0, 10.0, 40.0, 35.0]
        }

# aspect ratios
def anchor_aspect_ratios():
    aspect_ratios = config.params['aspect_ratios']
    return aspect_ratios

