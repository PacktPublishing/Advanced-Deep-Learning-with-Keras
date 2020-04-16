"""FCN class to build, train, eval an FCN model for semantic
    segmentation

1)  ResNet50 (v2) backbone.
    Train with 6 layers of feature maps.
    Pls adjust batch size depending on your GPU memory.
    For 1060 with 6GB, --batch-size=1. For V100 with 32GB, 
    --batch-size=4

python3 fcn-12.3.1.py --train --batch-size=4

2)  ResNet50 (v2) backbone.
    Train from a previously saved model:

python3 fcn-12.3.1.py --restore-weights=ResNet56v2-3layer-drinks-200.h5 \
        --train --batch-size=4

2)  ResNet50 (v2) backbone.
    Evaluate:

python3 fcn-12.3.1.py --restore-weights=ResNet56v2-3layer-drinks-200.h5 \
        --evaluate --image-file=dataset/drinks/0010018.jpg

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
from common_utils import print_log, AccuracyCallback
from model import build_fcn
from skimage.io import imread


class FCN:
    """Made of an fcn model and a dataset generator.
    Define functions to train and validate an FCN model.

    Arguments:
        args: User-defined configurations

    Attributes:
        fcn (model): FCN network model
        train_generator: Multi-threaded 
            data generator for training
    """
    def __init__(self, args):
        """Copy user-defined configs.
        Build backbone and fcn network models.
        """
        self.args = args
        self.fcn = None
        self.train_generator = DataGenerator(args)
        self.build_model()
        self.eval_init()


    def build_model(self):
        """Build a backbone network and use it to
            create a semantic segmentation 
            network based on FCN.
        """
        
        # input shape is (480, 640, 3) by default
        self.input_shape = (self.args.height, 
                            self.args.width,
                            self.args.channels)

        # build the backbone network (eg ResNet50)
        # the backbone is used for 1st set of features
        # of the features pyramid
        self.backbone = self.args.backbone(self.input_shape,
                                           n_layers=self.args.layers)

        # using the backbone, build fcn network
        # output layer is a pixel-wise classifier
        self.n_classes =  self.train_generator.n_classes
        self.fcn = build_fcn(self.input_shape,
                             self.backbone,
                             self.n_classes)


    def eval_init(self):
        """Housekeeping for trained model evaluation"""
        # model weights are saved for future validation
        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), self.args.save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_name = self.backbone.name
        model_name += '-' + str(self.args.layers) + "layer-"
        model_name += self.args.dataset
        model_name += '-best-iou.h5'
        log = "Weights filename: %s" % model_name
        print_log(log, self.args.verbose)
        self.weights_path = os.path.join(save_dir, model_name)
        self.preload_test()
        self.miou = 0
        self.miou_history = []
        self.mpla_history = []


    def preload_test(self):
        """Pre-load test dataset to save time """
        path = os.path.join(self.args.data_path,
                            self.args.test_labels)

        # ground truth data is stored in an npy file
        self.test_dictionary = np.load(path,
                                       allow_pickle=True).flat[0]
        self.test_keys = np.array(list(self.test_dictionary.keys()))
        print_log("Loaded %s" % path, self.args.verbose)


    def train(self):
        """Train an FCN"""
        optimizer = Adam(lr=1e-3)
        loss = 'categorical_crossentropy'
        self.fcn.compile(optimizer=optimizer, loss=loss)

        log = "# of classes %d" % self.n_classes
        print_log(log, self.args.verbose)
        log = "Batch size: %d" % self.args.batch_size
        print_log(log, self.args.verbose)

        # prepare callbacks for saving model weights
        # and learning rate scheduler
        # model weights are saved when test iou is highest
        # learning rate decreases by 50% every 20 epochs
        # after 40th epoch
        accuracy = AccuracyCallback(self)
        scheduler = LearningRateScheduler(lr_scheduler)

        callbacks = [accuracy, scheduler]
        # train the fcn network
        self.fcn.fit(x=self.train_generator,
                     use_multiprocessing=False,
                     callbacks=callbacks,
                     epochs=self.args.epochs)
                     #workers=self.args.workers)


    def restore_weights(self):
        """Load previously trained model weights"""
        if self.args.restore_weights:
            save_dir = os.path.join(os.getcwd(), self.args.save_dir)
            filename = os.path.join(save_dir, self.args.restore_weights)
            log = "Loading weights: %s" % filename
            print_log(log, self.args.verbose)
            self.fcn.load_weights(filename)


    def segment_objects(self, image, normalized=True):
        """Run segmentation prediction for a given image
    
        Arguments:
            image (tensor): Image loaded in a numpy tensor.
                RGB components range is [0.0, 1.0]
            normalized (Bool): Use normalized=True for 
                pixel-wise categorical prediction. False if 
                segmentation will be displayed in RGB
                image format.
        """

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


    def evaluate(self, imagefile=None, image=None):
        """Perform segmentation on a given image filename
            and display the results.
        """
        import matplotlib.pyplot as plt
        save_dir = "prediction"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if image is not None:
            imagefile = os.path.splitext(imagefile)[0]
        elif self.args.image_file is not None:
            image = skimage.img_as_float(imread(self.args.image_file))
            imagefile = os.path.split(self.args.image_file)[-1]
            print("imagefile:", imagefile)
        else:
            raise ValueError("Image file must be known")

        maskfile = imagefile + "-mask.png"
        mask_path = os.path.join(save_dir, maskfile)
        inputfile = imagefile + "-input.png"
        input_path = os.path.join(save_dir, inputfile)
        segmentation = self.segment_objects(image,
                                            normalized=False)
        mask = segmentation[..., 1:]
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Input image', fontsize=14)
        plt.imshow(image)
        plt.savefig(input_path)
        #plt.show()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Semantic segmentation', fontsize=14)
        plt.imshow(mask)
        plt.savefig(mask_path)
        #plt.show()


    def eval(self):
        """Evaluate a trained FCN model using mean IoU
            metric.
        """
        s_iou = 0
        s_pla = 0
        # evaluate iou per test image
        eps = np.finfo(float).eps
        for key in self.test_keys:
            # load a test image
            image_path = os.path.join(self.args.data_path, key)
            image = skimage.img_as_float(imread(image_path))
            segmentation = self.segment_objects(image) 
            # load test image ground truth labels
            gt = self.test_dictionary[key]
            i_pla = 100 * (gt == segmentation).all(axis=(2)).mean()
            s_pla += i_pla
            
            i_iou = 0
            n_masks = 0
            # compute mask for each object in the test image
            # including background
            for i in range(self.n_classes):
                if np.sum(gt[..., i]) < eps: 
                    continue
                mask = segmentation[..., i]
                intersection = mask * gt[..., i]
                union = np.ceil((mask + gt[..., i]) / 2.0)
                intersection = np.sum(intersection) 
                union = np.sum(union) 
                if union > eps:
                    iou = intersection / union
                    i_iou += iou
                    n_masks += 1
            
            # average iou per image
            i_iou /= n_masks
            if not self.args.train:
                log = "%s: %d objs, miou=%0.4f ,pla=%0.2f%%"\
                      % (key, n_masks, i_iou, i_pla)
                print_log(log, self.args.verbose)

            # accumulate all image ious
            s_iou += i_iou
            if self.args.plot:
                self.evaluate(key, image)

        n_test = len(self.test_keys)
        m_iou = s_iou / n_test 
        self.miou_history.append(m_iou)
        np.save("miou_history.npy", self.miou_history)
        m_pla = s_pla / n_test
        self.mpla_history.append(m_pla)
        np.save("mpla_history.npy", self.mpla_history)
        if m_iou > self.miou and self.args.train:
            log = "\nOld best mIoU=%0.4f, New best mIoU=%0.4f, Pixel level accuracy=%0.2f%%"\
                    % (self.miou, m_iou, m_pla)
            print_log(log, self.args.verbose)
            self.miou = m_iou
            print_log("Saving weights... %s"\
                      % self.weights_path,\
                      self.args.verbose)
            self.fcn.save_weights(self.weights_path)
        else:
            log = "\nCurrent mIoU=%0.4f, Pixel level accuracy=%0.2f%%"\
                    % (m_iou, m_pla)
            print_log(log, self.args.verbose)


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
            fcn.eval()
        else:
            fcn.evaluate()
            
    if args.train:
        fcn.train()
