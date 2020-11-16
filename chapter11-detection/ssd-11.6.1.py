"""SSD class to build, train, eval an SSD network

1)  ResNet50 (v2) backbone.
    Train with 6 layers of feature maps.
    Pls adjust batch size depending on your GPU memory.
    For 1060 with 6GB, -b=1. For V100 with 32GB, -b=4

python3 ssd-11.6.1.py -t -b=4

2)  ResNet50 (v2) backbone.
    Train from a previously saved model:

python3 ssd-11.6.1.py --restore-weights=saved_models/ResNet56v2_4-layer_weights-200.h5 -t -b=4

2)  ResNet50 (v2) backbone.
    Evaluate:

python3 ssd-11.6.1.py -e --restore-weights=saved_models/ResNet56v2_4-layer_weights-200.h5 \
        --image-file=dataset/drinks/0010000.jpg

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber

import layer_utils
import label_utils
import config

import os
import skimage
import numpy as np
import argparse

from skimage.io import imread
from data_generator import DataGenerator
from label_utils import build_label_dictionary
from boxes import show_boxes
from model import build_ssd
from loss import focal_loss_categorical, smooth_l1_loss, l1_loss
from model_utils import lr_scheduler, ssd_parser
from common_utils import print_log


class SSD:
    """Made of an ssd network model and a dataset generator.
    SSD defines functions to train and validate 
    an ssd network model.

    Arguments:
        args: User-defined configurations

    Attributes:
        ssd (model): SSD network model
        train_generator: Multi-threaded data generator for training
    """
    def __init__(self, args):
        """Copy user-defined configs.
        Build backbone and ssd network models.
        """
        self.args = args
        self.ssd = None
        self.train_generator = None
        self.build_model()


    def build_model(self):
        """Build backbone and SSD models."""
        # store in a dictionary the list of image files and labels
        self.build_dictionary()
        
        # input shape is (480, 640, 3) by default
        self.input_shape = (self.args.height, 
                            self.args.width,
                            self.args.channels)

        # build the backbone network (eg ResNet50)
        # the number of feature layers is equal to n_layers
        # feature layers are inputs to SSD network heads
        # for class and offsets predictions
        self.backbone = self.args.backbone(self.input_shape,
                                           n_layers=self.args.layers)

        # using the backbone, build ssd network
        # outputs of ssd are class and offsets predictions
        anchors, features, ssd = build_ssd(self.input_shape,
                                           self.backbone,
                                           n_layers=self.args.layers,
                                           n_classes=self.n_classes)
        # n_anchors = num of anchors per feature point (eg 4)
        self.n_anchors = anchors
        # feature_shapes is a list of feature map shapes
        # per output layer - used for computing anchor boxes sizes
        self.feature_shapes = features
        # ssd network model
        self.ssd = ssd


    def build_dictionary(self):
        """Read input image filenames and obj detection labels
        from a csv file and store in a dictionary.
        """
        # train dataset path
        path = os.path.join(self.args.data_path,
                            self.args.train_labels)

        # build dictionary: 
        # key=image filaname, value=box coords + class label
        # self.classes is a list of class labels
        self.dictionary, self.classes = build_label_dictionary(path)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))


    def build_generator(self):
        """Build a multi-thread train data generator."""

        self.train_generator = \
                DataGenerator(args=self.args,
                              dictionary=self.dictionary,
                              n_classes=self.n_classes,
                              feature_shapes=self.feature_shapes,
                              n_anchors=self.n_anchors,
                              shuffle=True)


    def train(self):
        """Train an ssd network."""
        # build the train data generator
        if self.train_generator is None:
            self.build_generator()

        optimizer = Adam(lr=1e-3)
        # choice of loss functions via args
        if self.args.improved_loss:
            print_log("Focal loss and smooth L1", self.args.verbose)
            loss = [focal_loss_categorical, smooth_l1_loss]
        elif self.args.smooth_l1:
            print_log("Smooth L1", self.args.verbose)
            loss = ['categorical_crossentropy', smooth_l1_loss]
        else:
            print_log("Cross-entropy and L1", self.args.verbose)
            loss = ['categorical_crossentropy', l1_loss]

        self.ssd.compile(optimizer=optimizer, loss=loss)

        # model weights are saved for future validation
        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), self.args.save_dir)
        model_name = self.backbone.name
        model_name += '-' + str(self.args.layers) + "layer"
        if self.args.normalize:
            model_name += "-norm"
        if self.args.improved_loss:
            model_name += "-improved_loss"
        elif self.args.smooth_l1:
            model_name += "-smooth_l1"

        if self.args.threshold < 1.0:
            model_name += "-extra_anchors" 

        model_name += "-" 
        model_name += self.args.dataset
        model_name += '-{epoch:03d}.h5'

        log = "# of classes %d" % self.n_classes
        print_log(log, self.args.verbose)
        log = "Batch size: %d" % self.args.batch_size
        print_log(log, self.args.verbose)
        log = "Weights filename: %s" % model_name
        print_log(log, self.args.verbose)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # prepare callbacks for saving model weights
        # and learning rate scheduler
        # learning rate decreases by 50% every 20 epochs
        # after 60th epoch
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)
        scheduler = LearningRateScheduler(lr_scheduler)

        callbacks = [checkpoint, scheduler]
        # train the ssd network
        self.ssd.fit(self.train_generator,
                     use_multiprocessing=False,
                     callbacks=callbacks,
                     epochs=self.args.epochs)


    def restore_weights(self):
        """Load previously trained model weights"""
        if self.args.restore_weights:
            save_dir = os.path.join(os.getcwd(), self.args.save_dir)
            filename = os.path.join(save_dir, self.args.restore_weights)
            log = "Loading weights: %s" % filename
            print(log, self.args.verbose)
            self.ssd.load_weights(filename)


    def detect_objects(self, image):
        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        image = np.squeeze(image, axis=0)
        classes = np.squeeze(classes)
        offsets = np.squeeze(offsets)
        return image, classes, offsets


    def evaluate(self, image_file=None, image=None):
        """Evaluate image based on image (np tensor) or filename"""
        show = False
        if image is None:
            image = skimage.img_as_float(imread(image_file))
            show = True

        image, classes, offsets = self.detect_objects(image)
        class_names, rects, _, _ = show_boxes(args,
                                              image,
                                              classes,
                                              offsets,
                                              self.feature_shapes,
                                              show=show)
        return class_names, rects


    def evaluate_test(self):
        # test labels csv path
        path = os.path.join(self.args.data_path,
                            self.args.test_labels)
        # test dictionary
        dictionary, _ = build_label_dictionary(path)
        keys = np.array(list(dictionary.keys()))
        # sum of precision
        s_precision = 0
        # sum of recall
        s_recall = 0
        # sum of IoUs
        s_iou = 0
        # evaluate per image
        for key in keys:
            # grounnd truth labels
            labels = np.array(dictionary[key])
            # 4 boxes coords are 1st four items of labels
            gt_boxes = labels[:, 0:-1]
            # last one is class
            gt_class_ids = labels[:, -1]
            # load image id by key
            image_file = os.path.join(self.args.data_path, key)
            image = skimage.img_as_float(imread(image_file))
            image, classes, offsets = self.detect_objects(image)
            # perform nms
            _, _, class_ids, boxes = show_boxes(args,
                                                image,
                                                classes,
                                                offsets,
                                                self.feature_shapes,
                                                show=False)

            boxes = np.reshape(np.array(boxes), (-1,4))
            # compute IoUs
            iou = layer_utils.iou(gt_boxes, boxes)
            # skip empty IoUs
            if iou.size ==0:
                continue
            # the class of predicted box w/ max iou
            maxiou_class = np.argmax(iou, axis=1)

            # true positive
            tp = 0
            # false positiove
            fp = 0
            # sum of objects iou per image
            s_image_iou = []
            for n in range(iou.shape[0]):
                # ground truth bbox has a label
                if iou[n, maxiou_class[n]] > 0:
                    s_image_iou.append(iou[n, maxiou_class[n]])
                    # true positive has the same class and gt
                    if gt_class_ids[n] == class_ids[maxiou_class[n]]:
                        tp += 1
                    else:
                        fp += 1

            # objects that we missed (false negative)
            fn = abs(len(gt_class_ids) - tp - fp)
            s_iou += (np.sum(s_image_iou) / iou.shape[0])
            s_precision += (tp/(tp + fp))
            s_recall += (tp/(tp + fn))


        n_test = len(keys)
        print_log("mIoU: %f" % (s_iou/n_test),
                  self.args.verbose)
        print_log("Precision: %f" % (s_precision/n_test),
                  self.args.verbose)
        print_log("Recall : %f" % (s_recall/n_test),
                  self.args.verbose)


    def print_summary(self):
        """Print network summary for debugging purposes."""
        from tensorflow.keras.utils import plot_model
        if self.args.summary:
            self.backbone.summary()
            self.ssd.summary()
            plot_model(self.backbone,
                       to_file="backbone.png",
                       show_shapes=True)


if __name__ == '__main__':
    parser = ssd_parser()
    args = parser.parse_args()
    ssd = SSD(args)

    if args.summary:
        ssd.print_summary()

    if args.restore_weights:
        ssd.restore_weights()
        if args.evaluate:
            if args.image_file is None:
                ssd.evaluate_test()
            else:
                ssd.evaluate(image_file=args.image_file)
            
    if args.train:
        ssd.train()
