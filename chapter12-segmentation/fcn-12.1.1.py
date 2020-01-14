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

import os
import skimage
import numpy as np

from data_generator import DataGenerator
from model_utils import parser, lr_scheduler
os.sys.path.append("../lib")
from common_utils import print_log
from model import build_fcn
from skimage.io import imread



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
        """Build backbone and FCN models"""
        
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
        self.n_classes =  self.train_generator.n_classes
        self.fcn = build_fcn(self.input_shape,
                             self.backbone,
                             self.n_classes)


    def train(self):
        """Train an fcn network."""
        optimizer = Adam(lr=1e-3)
        loss = 'categorical_crossentropy'
        self.fcn.compile(optimizer=optimizer, loss=loss)

        # model weights are saved for future validation
        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), self.args.save_dir)
        model_name = self.backbone.name
        model_name += '-' + str(self.args.layers) + "layer-"
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
        self.fcn.fit(generator=self.train_generator,
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


    def segment_objects(self, image, normalized=True):
        from tensorflow.keras.utils import to_categorical
        image = np.expand_dims(image, axis=0)
        segmentation = self.fcn.predict(image)
        segmentation = np.squeeze(segmentation, axis=0)
        segmentation = np.argmax(segmentation, axis=-1)
        segmentation = to_categorical(segmentation,
                                      num_classes=self.n_classes)
        if not normalized:
            segmentation = segmentation * 255
        segmentation = segmentation.astype('uint8')
        return segmentation


    def evaluate(self):
        """Evaluate image based on filename"""
        import matplotlib.pyplot as plt
        if self.args.image_file is None:
            raise ValueError("--image-file must be known")
        
        image = skimage.img_as_float(imread(self.args.image_file))
        segmentation = self.segment_objects(image, normalized=False)
        mask = segmentation[..., 1:]
        #bg = segmentation[..., 0]
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Input image', fontsize=14)
        plt.imshow(image)
        plt.show()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Semantic segmentation', fontsize=14)
        plt.imshow(mask)
        plt.show()


    def evaluate_test(self):
        # test labels csv path
        path = os.path.join(self.args.data_path,
                            self.args.test_labels)

        dictionary = np.load(path, allow_pickle=True).flat[0]
        keys = np.array(list(dictionary.keys()))
        s_iou = 0
        for key in keys:
            image_path = os.path.join(self.args.data_path, key)
            image = skimage.img_as_float(imread(image_path))
            segmentation = self.segment_objects(image) 
            gt = dictionary[key]
            i_iou = 0
            n_masks = 0
            for i in range(self.n_classes):
                is_mask = np.sum(gt[..., i]) > 0
                if is_mask is False:
                    continue
                mask = segmentation[..., i]
                intersection = mask * gt[..., i]
                union = np.ceil((mask + gt[..., i]) / 2.0)
                intersection = np.sum(intersection) 
                union = np.sum(union) 
                if union > 0.0:
                    iou = intersection / union
                    i_iou += iou
                    n_masks += 1
            #break
            
            i_iou /= n_masks
            print(n_masks, i_iou)
            s_iou += i_iou
            #print(intersection, union, iou)
            #break

        n_test = len(keys)
        print_log("mIoU: %f" % (s_iou/n_test),
                  self.args.verbose)


    def print_summary(self):
        """Print network summary for debugging purposes."""
        from tensorflow.keras.utils import plot_model
        if self.args.summary:
            self.backbone.summary()
            self.fcn.summary()
            plot_model(self.fcn,
                       to_file="fcn.png",
                       show_shapes=True)
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
            fcn.evaluate()
            
    if args.train:
        fcn.train()
