"""

python3 videocapture.py --camera=1

"""

import numpy as np
import cv2
import argparse
import datetime
import os
import time
from skimage.io import imsave

class  VideoCapture():
    def __init__(self,
                 camera=0,
                 width=640,
                 height=480,
                 path="dataset/capture",
                 index=0):
        self.camera = camera
        self.width = width
        self.height = height
        self.path = path
        self.index = index
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        # cap.set(cv2.CAP_PROP_FPS, 5)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def loop(self):
        i = self.index
        start_time = datetime.datetime.now()
        while True:
            ret, image = self.capture.read()
            # img = cv2.resize(img, dsize=(320, 240), 
            # interpolation=cv2.INTER_CUBIC)
            # image = image / 255.0
            cv2.imshow('image', image)

            elapsed_time = datetime.datetime.now() - start_time
            secs = int(elapsed_time.total_seconds())
            if secs > 5:
                target_file = "%07d" % i
                target_file += ".jpg"
                filename = os.path.join(self.path, target_file)
                # imsave(filename, image)
                cv2.imwrite(filename, image)

                i += 1
                start_time = datetime.datetime.now()
                print(filename)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Camera index"
    parser.add_argument("--camera",
                        default=0,
                        type=int,
                        help=help_)
    help_ = "Image index"
    parser.add_argument("--index",
                        default=0,
                        type=int,
                        help=help_)


    args = parser.parse_args()

    videocap = VideoCapture(camera=args.camera, index=args.index)
    videocap.loop()
