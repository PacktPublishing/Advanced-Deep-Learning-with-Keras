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
import datetime


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

    # display color version of test images
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


def train(models, data, params):
    # the models
    gen_gray, gen_color, dis_gray, dis_color, adv = models
    # network parameters
    batch_size, train_steps, dis_patch, model_name = params
    # train dataset
    x_train, x_test, x_train_gray, x_test_gray = data
    # the generator image is saved every 500 steps
    save_interval = 500
    # number of elements in train dataset
    train_size = x_train.shape[0]

    # valid = np.ones((batch_size,) + dis_patch)
    # fake = np.zeros((batch_size,) + dis_patch)
    valid = np.ones([batch_size, 1])
    fake = np.zeros([batch_size, 1])
    valid_fake = np.concatenate((valid, fake))

    start_time = datetime.datetime.now()

    for i in range(train_steps):
        # train the discriminator1 for 1 batch
        # 1 batch of real (label=1.0) and fake feature1 (label=0.0)
        # randomly pick real images from dataset
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_color = x_train[rand_indexes]

        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_gray = x_train_gray[rand_indexes]
        fake_color = gen_color.predict(real_gray)
        
        x = np.concatenate((real_color, fake_color))
        metrics = dis_color.train_on_batch(x, valid_fake)
        log = "%d: [dis_color loss: %f]" % (i, metrics[0])

        # rand_indexes = np.random.randint(0, train_size, size=batch_size)
        # real_gray = x_train_gray[rand_indexes]

        # rand_indexes = np.random.randint(0, train_size, size=batch_size)
        # real_color = x_train[rand_indexes]
        fake_gray = gen_gray.predict(real_color)

        x = np.concatenate((real_gray, fake_gray))
        metrics = dis_gray.train_on_batch(x, valid_fake)
        log = "%s [dis_gray loss: %f]" % (log, metrics[0])

        # rand_indexes = np.random.randint(0, train_size, size=batch_size)
        # real_gray = x_train_gray[rand_indexes]
        # fake_color = gen_color.predict(real_gray)

        # rand_indexes = np.random.randint(0, train_size, size=batch_size)
        # real_color = x_train[rand_indexes]
        # fake_gray = gen_gray.predict(real_color)

        x = [real_gray, real_color]
        y = [valid, valid, real_gray, real_color, real_gray, real_color]
        metrics = adv.train_on_batch(x, y)
        # print(adv.metrics_names)
        elapsed_time = datetime.datetime.now() - start_time
        fmt = "%s [adv loss: %f] [time: %s]"
        log = fmt % (log, metrics[0], elapsed_time)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False

            test_generator(gen_color,
                           x_test_gray,
                           step=i+1,
                           show=show)

    # save the model after training the generator
    gen_gray.save(model_name + "-gray.h5")
    gen_color.save(model_name + "-color.h5")



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

