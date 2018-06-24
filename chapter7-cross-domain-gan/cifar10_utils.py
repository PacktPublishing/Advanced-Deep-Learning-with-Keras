"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import cifar10
from keras.models import load_model

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



def load_data():
    # load the CIFAR10 data
    (x_train, _), (x_test, _) = cifar10.load_data()

    # input image dimensions
    # we assume data format "channels_last"
    rows = x_train.shape[1]
    cols = x_train.shape[2]
    channels = x_train.shape[3]


    # convert color train and test images to gray
    x_train_gray = rgb2gray(x_train)
    x_test_gray = rgb2gray(x_test)

    imgs = x_test[:100]
    title = 'Test color images (Ground  Truth)'
    img_shape = (rows, cols, channels)
    filename = 'test_color.png'
    display_images(imgs,
                   img_shape=img_shape,
                   filename=filename,
                   title=title)

    # display grayscale version of test images
    imgs = x_test_gray[:100]
    title = 'Test gray images (Input)'
    filename = 'test_gray.png'
    display_images(imgs,
                   img_shape=(rows, cols, 1),
                   filename=filename,
                   title=title)


    # normalize output train and test color images
    x_train = x_train.astype('float32')  / 255
    x_test = x_test.astype('float32') / 255

    # normalize input train and test grayscale images
    x_train_gray = x_train_gray.astype('float32')  / 255
    x_test_gray = x_test_gray.astype('float32') / 255

    # reshape images to row x col x channels
    # for CNN output/validation
    x_train = x_train.reshape(x_train.shape[0],
                              rows,
                              cols,
                              channels)
    x_test = x_test.reshape(x_test.shape[0],
                            rows,
                            cols,
                            channels)

    # reshape images to row x col x channel for CNN input
    x_train_gray = x_train_gray.reshape(x_train_gray.shape[0],
                                        rows,
                                        cols,
                                        1)
    x_test_gray = x_test_gray.reshape(x_test_gray.shape[0],
                                      rows,
                                      cols,
                                      1)

    data = (x_train, x_test, x_train_gray, x_test_gray)
    img_shape = (rows, cols, channels)
    
    return data, img_shape


def test_generator(generator,
                   x_test_gray,
                   step,
                   show=False):
    # predict the output from test data
    imgs_color = generator.predict(x_test_gray)

    # display the 1st 100 colorized images
    imgs = imgs_color[:100]
    # imgs = 0.5 * (imgs + 1.0)
    title = 'Colorized test images (Predicted)'
    filename = '%05d.png' % step
    imgs_dir = 'test_outputs'
    img_shape = (x_test_gray.shape[1],
                 x_test_gray.shape[2],
                 3)
    display_images(imgs,
                   img_shape=img_shape,
                   filename=filename,
                   imgs_dir=imgs_dir,
                   title=title,
                   show=False)

