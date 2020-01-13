"""FCN class to build, train, eval an FCN network

1)  ResNet50 (v2) backbone.
    Train with 6 layers of feature maps.
    Pls adjust batch size depending on your GPU memory.
    For 1060 with 6GB, -b=1. For V100 with 32GB, -b=4

python3 fcn-11.6.1.py -t -b=4

2)  ResNet50 (v2) backbone.
    Train from a previously saved model:

python3 fcn-11.6.1.py --restore-weights=saved_models/ResNet56v2_4-layer_weights-200.h5 -t -b=4

2)  ResNet50 (v2) backbone.
    Evaluate:

python3 fcn-11.6.1.py -e --restore-weights=saved_models/ResNet56v2_4-layer_weights-200.h5 \
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

import os
import skimage
import numpy as np
import argparse

from data_generator import DataGenerator
from model_utils import parser
os.sys.path.append("../lib")
from common_utils import print_log
from model import build_fcn


class FCN:
    """Made of an fcn network model and a dataset generator.
    FCN defines functions to train and validate 
    an fcn network model.

    Arguments:
        args: User-defined configurations

    Attributes:
        fcn (model): FCN network model
        train_generator: Multi-threaded data generator for training
    """
    def __init__(self, args):
        """Copy user-defined configs.
        Build backbone and fcn network models.
        """
        self.args = args
        self.fcn = None
        self.train_generator = DataGenerator(args)
        self.build_model()


    def build_model(self):
        """Build backbone and FCN model."""
        
        # input shape is (480, 640, 3) by default
        self.input_shape = (self.args.height, 
                            self.args.width,
                            self.args.channels)

        # build the backbone network (eg ResNet50)
        # the number of feature layers is equal to n_layers
        # feature layers are inputs to FCN network heads
        # for class and offsets predictions
        self.backbone = self.args.backbone(self.input_shape,
                                           n_layers=self.args.layers)

        # using the backbone, build fcn network
        # outputs of fcn are class and offsets predictions
        self.fcn = build_fcn(self.input_shape,
                             self.backbone)
        self.fcn.summary()


    def train(self):
        """Train an fcn network."""
        optimizer = Adam(lr=1e-3)
        loss = Softmax()
        self.fcn.compile(optimizer=optimizer, loss=loss)

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
        # train the fcn network
        self.fcn.fit_generator(generator=self.train_generator,
                               use_multiprocessing=True,
                               callbacks=callbacks,
                               epochs=self.args.epochs,
                               workers=self.args.workers)


    def restore_weights(self):
        """Load previously trained model weights"""
        if self.args.restore_weights:
            save_dir = os.path.join(os.getcwd(), self.args.save_dir)
            filename = os.path.join(save_dir, self.args.restore_weights)
            log = "Loading weights: %s" % filename
            print(log, self.args.verbose)
            self.fcn.load_weights(filename)


    def detect_objects(self, image):
        image = np.expand_dims(image, axis=0)
        classes, offsets = self.fcn.predict(image)
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
            fn = abs(len(gt_class_ids) - tp)
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
            self.fcn.summary()
            plot_model(self.backbone,
                       to_file="backbone.png",
                       show_shapes=True)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()
    fcn = FCN(args)

    if args.summary:
        fcn.print_summary()

    if args.restore_weights:
        fcn.restore_weights()
        if args.evaluate:
            if args.image_file is None:
                fcn.evaluate_test()
            else:
                fcn.evaluate(image_file=args.image_file)
            
    if args.train:
        fcn.train()
