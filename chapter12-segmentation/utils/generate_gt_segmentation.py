'''

Pretty print: python3 -m json.tool < some.json

'''

import json
import argparse
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_json(data_path, jsfile):
    with open(os.path.join(data_path, jsfile), 'r') as f:
        js = json.load(f)

    return js


def generate_dataset(args):
    data_dict = {}
    js = load_json(args.data_path, args.json)
    js = js["_via_img_metadata"]
    keys = js.keys()
    rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    images_no_objs = []
    for key in keys:
        entry = js[key]
        filename = entry["filename"]
        path = os.path.join(args.data_path, filename)
        regions = entry["regions"]
        masks = []
        for region in regions:
            shape = region["shape_attributes"]
            x = shape["all_points_x"]
            y = shape["all_points_y"]
            name = region["region_attributes"]
            class_id = name["Name"]
            fmt = "%s,%s,%s,%s"
            line = fmt % (filename, x, y, class_id)
            # print(line)
            xy = np.array([x, y], dtype=np.int32)
            xy = np.transpose(xy)
            xy = np.reshape(xy, [1, -1, 2])
            mask = { class_id : xy }
            masks.append(mask)

        image = plt.imread(path)
        if args.show:
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Input image', fontsize=14)
            fname = os.path.splitext(filename)[0]
            fname = fname + "-input.png"
            path = os.path.join("images", fname)
            plt.imshow(image)
            plt.savefig(path)
            #plt.show()
        else:
            image = np.zeros_like(image)

        shape = image.shape
        shape = (shape[0], shape[1])
        bg = np.ones(shape, dtype="uint8")
        bg.fill(255)
        #i = 0
        image = np.zeros_like(image)
        #image[:] = [128, 0, 128]
        for mask in masks:
            name = list(mask)[0]
            mask = mask[name]
            cv2.fillPoly(image, mask, rgb[int(name)-1])
            # cv2.fillPoly(image, mask, rgb[i])
            #i += 1
            cv2.fillPoly(bg, mask, 0)

        if args.show:
            name = os.path.splitext(filename)[0]

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Ground truth semantic segmentation', fontsize=14)
            fname = name + "-semantic.png"
            path = os.path.join("images", fname)
            plt.imshow(image)
            plt.savefig(path)
            #plt.show()

            #plt.xlabel('x')
            #plt.ylabel('y')
            #plt.title('Background segmentation', fontsize=14)
            #fname = name + "-bg.png"
            #path = os.path.join("images", fname)
            #plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
            #plt.savefig(path)
            #plt.show()


        shape = (*shape, 1)
        bg = np.reshape(bg, shape)
        #print(bg.shape)
        data = np.concatenate((bg, image), axis=-1)
        data = data.astype('float32') / 255
        data = data.astype('uint8')
        data_dict[filename] = data
        print(filename, len(masks))
        if len(masks) == 0:
            images_no_objs.append(filename)

    if not args.show:
        np.save(args.save_filename, data_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j",
                        "--json",
                        default='segmentation_train.json',
                        help='Json filename')
    parser.add_argument("-p",
                        "--data-path",
                        default='../dataset/drinks',
                        help='Path to dataset')
    parser.add_argument("--save-filename",
                        default="segmentation_train.npy",
                        help='Path to dataset')
    help_ = "Show and save images"
    parser.add_argument("--show",
                        default=False,
                        action='store_true', 
                        help=help_)
    args = parser.parse_args()

    generate_dataset(args)
