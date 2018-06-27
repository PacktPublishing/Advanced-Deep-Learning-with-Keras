"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os

# convert from color image (RGB) to grayscale
# source: opencv.org
# grayscale = 0.299*red + 0.587*green + 0.114*blue
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def display_images(imgs,
                   img_shape,
                   filename,
                   title='',
                   imgs_dir=None,
                   show=False):
    # display the 1st 100 input images (color and gray)
    # create saved_images folder
    if imgs_dir is None:
        imgs_dir = 'saved_images'
    save_dir = os.path.join(os.getcwd(), imgs_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(imgs_dir, filename)
    rows, cols, channels = img_shape
    if channels==1:
        imgs = imgs.reshape((10, 10, rows, cols))
    else:
        imgs = imgs.reshape((10, 10, rows, cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title(title)
    if channels==1:
        plt.imshow(imgs, interpolation='none', cmap='gray')
    else:
        plt.imshow(imgs, interpolation='none')
    plt.savefig(filename)
    if show:
        plt.show()
    
    plt.close('all')



def test_generator(generator,
                   test_source_data,
                   step,
                   show=False):
    # predict the output from test data
    test_target_data = generator.predict(test_source_data)

    # display the 1st 100 colorized images
    imgs = test_target_data[:100]
    title = "{:,}".format(step)
    title = 'CycleGAN predicted target images. Step: %s' % title
    filename = '%05d.png' % step
    imgs_dir = 'test_outputs'
    img_shape = (test_source_data.shape[1],
                 test_source_data.shape[2],
                 3)
    display_images(imgs,
                   img_shape=img_shape,
                   filename=filename,
                   imgs_dir=imgs_dir,
                   title=title,
                   show=False)

