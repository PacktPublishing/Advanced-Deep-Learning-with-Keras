'''


'''

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_history(args):
    y = np.load(args.history)
    print("Max: ", np.amax(y))
    x = np.arange(1, 101)
    plt.xlabel('epoch')
    if "iou" in args.history:
        plt.ylabel('mIoU')
        plt.title('mIoU on test dataset', fontsize=14)
    else:
        plt.ylabel('average pixel accuracy')
        plt.title('Average pixel accuracy on test dataset', fontsize=14)
    plt.plot(x, y)
    filename = args.history + ".png"
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--history",
                        default='miou_history.npy',
                        help='Metric history vs epoch')
    args = parser.parse_args()
    plot_history(args)
