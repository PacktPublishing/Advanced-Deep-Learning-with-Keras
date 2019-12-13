"""Data generator
This is a multi-threaded, scalable, and efficient way of reading huge images
from a filesystem as dataset 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.python.keras.utils.data_utils import Sequence

import numpy as np
import layer_utils
import label_utils
import os
import skimage

from layer_utils import get_gt_data
from layer_utils import anchor_boxes

from skimage.io import imread
from skimage.util import random_noise
from skimage import exposure


class DataGenerator(Sequence):
    """Multi-threaded data generator.
    Each thread reads a batch of images and their object labels

    Arguments:
        args : User-defined configuration
        dictionary : Dictionary of image filenames and object labels
        n_classes (int): Number of object classes
        feature_shapes (tensor): Shapes of ssd head feature maps
        n_anchors (int): Number of anchor boxes per feature map pt
        shuffle (Bool): If dataset should be shuffled bef sampling
    """
    def __init__(self,
                 args,
                 dictionary,
                 n_classes,
                 feature_shapes=[],
                 n_anchors=4,
                 shuffle=True):
        self.args = args
        self.dictionary = dictionary
        self.n_classes = n_classes
        self.keys = np.array(list(self.dictionary.keys()))
        self.input_shape = (args.height, 
                            args.width,
                            args.channels)
        self.feature_shapes = feature_shapes
        self.n_anchors = n_anchors
        self.shuffle = shuffle
        self.on_epoch_end()
        self.get_n_boxes()


    def __len__(self):
        """Number of batches per epoch"""
        blen = np.floor(len(self.dictionary) / self.args.batch_size)
        return int(blen)


    def __getitem__(self, index):
        """Get a batch of data"""
        start_index = index * self.args.batch_size
        end_index = (index+1) * self.args.batch_size
        keys = self.keys[start_index : end_index]
        x, y = self.__data_generation(keys)
        return x, y


    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.keys)


    def get_n_boxes(self):
        """Total number of bounding boxes"""
        self.n_boxes = 0
        for shape in self.feature_shapes:
            self.n_boxes += np.prod(shape) // self.n_anchors
        return self.n_boxes


    def apply_random_noise(self, image, percent=30):
        """Apply random noise on an image (not used)"""
        random = np.random.randint(0, 100)
        if random < percent:
            image = random_noise(image)
        return image


    def apply_random_intensity_rescale(self, image, percent=30):
        """Apply random intensity rescale on an image (not used)"""
        random = np.random.randint(0, 100)
        if random < percent:
            v_min, v_max = np.percentile(image, (0.2, 99.8))
            image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        return image


    def apply_random_exposure_adjust(self, image, percent=30):
        """Apply random exposure adjustment on an image (not used)"""
        random = np.random.randint(0, 100)
        if random < percent:
            image = exposure.adjust_gamma(image, gamma=0.4, gain=0.9)
            # another exposure algo
            # image = exposure.adjust_log(image)
        return image


    def __data_generation(self, keys):
        """Generate train data: images and 
        object detection ground truth labels 

        Arguments:
            keys (array): Randomly sampled keys
                (key is image filename)

        Returns:
            x (tensor): Batch images
            y (tensor): Batch classes, offsets, and masks
        """
        # train input data
        x = np.zeros((self.args.batch_size, *self.input_shape))
        dim = (self.args.batch_size, self.n_boxes, self.n_classes)
        # class ground truth
        gt_class = np.zeros(dim)
        dim = (self.args.batch_size, self.n_boxes, 4)
        # offsets ground truth
        gt_offset = np.zeros(dim)
        # masks of valid bounding boxes
        gt_mask = np.zeros(dim)

        for i, key in enumerate(keys):
            # images are assumed to be stored in self.args.data_path
            # key is the image filename 
            image_path = os.path.join(self.args.data_path, key)
            image = skimage.img_as_float(imread(image_path))
            # assign image to a batch index
            x[i] = image
            # a label entry is made of 4-dim bounding box coords
            # and 1-dim class label
            labels = self.dictionary[key]
            labels = np.array(labels)
            # 4 bounding box coords are 1st four items of labels
            # last item is object class label
            boxes = labels[:,0:-1]
            for index, feature_shape in enumerate(self.feature_shapes):
                # generate anchor boxes
                anchors = anchor_boxes(feature_shape,
                                       image.shape,
                                       index=index,
                                       n_layers=self.args.layers)
                # each feature layer has a row of anchor boxes
                anchors = np.reshape(anchors, [-1, 4])
                # compute IoU of each anchor box 
                # with respect to each bounding boxes
                iou = layer_utils.iou(anchors, boxes)

                # generate ground truth class, offsets & mask
                gt = get_gt_data(iou,
                                 n_classes=self.n_classes,
                                 anchors=anchors,
                                 labels=labels,
                                 normalize=self.args.normalize,
                                 threshold=self.args.threshold)
                gt_cls, gt_off, gt_msk = gt
                if index == 0:
                    cls = np.array(gt_cls)
                    off = np.array(gt_off)
                    msk = np.array(gt_msk)
                else:
                    cls = np.append(cls, gt_cls, axis=0)
                    off = np.append(off, gt_off, axis=0)
                    msk = np.append(msk, gt_msk, axis=0)

            gt_class[i] = cls
            gt_offset[i] = off
            gt_mask[i] = msk


        y = [gt_class, np.concatenate((gt_offset, gt_mask), axis=-1)]

        return x, y
