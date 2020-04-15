"""Utility functionns for model building, training and evaluation

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
from resnet import build_resnet

def lr_scheduler(epoch):
    """Learning rate scheduler - called every epoch"""
    lr = 1e-3
    if epoch > 80:
        lr *= 5e-2
    elif epoch > 60:
        lr *= 1e-1
    elif epoch > 40:
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr

def parser():
    """Instatiate a command line parser for ssd network model
    building, training, and testing
    """
    parser = argparse.ArgumentParser(description='FCN for object segmentation')
    # arguments for model building and training
    help_ = "Number of feature extraction layers of FCN head after backbone"
    parser.add_argument("--layers",
                        default=3,
                        type=int,
                        help=help_)
    help_ = "Batch size during training"
    parser.add_argument("--batch-size",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Number of epochs to train"
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help=help_)
    help_ = "Number of data generator worker threads"
    parser.add_argument("--workers",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Backbone or base network"
    parser.add_argument("--backbone",
                        default=build_resnet,
                        help=help_)
    help_ = "Train the model"
    parser.add_argument("-t",
                        "--train",
                        action='store_true',
                        help=help_)
    help_ = "Print model summary (text and png)"
    parser.add_argument("--summary",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Directory for saving filenames"
    parser.add_argument("--save-dir",
                        default="weights",
                        help=help_)
    help_ = "Dataset name"
    parser.add_argument("--dataset",
                        default="drinks",
                        help=help_)

    # inputs configurations
    help_ = "Input image height"
    parser.add_argument("--height",
                        default=480,
                        type=int,
                        help=help_)
    help_ = "Input image width"
    parser.add_argument("--width",
                        default=640,
                        type=int,
                        help=help_)
    help_ = "Input image channels"
    parser.add_argument("--channels",
                        default=3,
                        type=int,
                        help=help_)

    # dataset configurations
    help_ = "Path to dataset directory"
    parser.add_argument("--data-path",
                        default="dataset/drinks",
                        help=help_)
    help_ = "Train data npy filename in --data-path"
    parser.add_argument("--train-labels",
                        default="segmentation_train.npy",
                        help=help_)
    help_ = "Test data npy filename in --data-path"
    parser.add_argument("--test-labels",
                        default="segmentation_test.npy",
                        help=help_)

    # configurations for evaluation of a trained model
    help_ = "Load h5 model trained weights"
    parser.add_argument("--restore-weights",
                        help=help_)
    help_ = "Evaluate model"
    parser.add_argument("-e",
                        "--evaluate",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Image file for evaluation"
    parser.add_argument("--image-file",
                        default=None,
                        help=help_)
    help_ = "Plot prediction during evaluation"
    parser.add_argument("--plot",
                        default=False,
                        action='store_true', 
                        help=help_)

    # debug configuration
    help_ = "Level of verbosity for print function"
    parser.add_argument("--verbose",
                        default=1,
                        type=int,
                        help=help_)

    return parser
