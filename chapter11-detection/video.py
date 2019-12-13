"""Video demo utils for showing live object detection from a camera

python3 video_demo.py --restore-weights=weights/<weights.h5>

"""

import numpy as np
import cv2
import argparse
import datetime
import skimage
from skimage.io import imread


class  VideoDemo():
    def __init__(self,
                 camera=0,
                 width=640,
                 height=480,
                 record=False,
                 filename="demo.mp4"):
        self.camera = camera
        self.width = width
        self.height = height
        self.record = record
        self.filename = filename
        self.videowriter = None
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        # cap.set(cv2.CAP_PROP_FPS, 5)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.record:
            self.videowriter = cv2.VideoWriter(self.filename,
                                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                10,
                                                (self.width, self.height), 
                                                isColor=True)

    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        pos = (10,30)
        font_scale = 0.9
        font_color = (0, 0, 0)
        line_type = 1

        while True:
            start_time = datetime.datetime.now()
            ret, image = self.capture.read()

            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #/ 255.0

            cv2.imshow('image', image)
            if self.videowriter is not None:
                if self.videowriter.isOpened():
                    self.videowriter.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video capture')
    help_ = "Camera index"
    parser.add_argument("--camera",
                        default=0,
                        type=int,
                        help=help_)
    help_ = "Record video"
    parser.add_argument("--record",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Video filename"
    parser.add_argument("--filename",
                        default="demo.mp4",
                        help=help_)

    args = parser.parse_args()

    videodemo = VideoDemo(camera=args.camera,
                          record=args.record,
                          filename=args.filename)
    videodemo.loop()
