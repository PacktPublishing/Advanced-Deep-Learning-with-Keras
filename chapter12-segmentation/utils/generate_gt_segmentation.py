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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j",
                        "--json",
                        default='seg_train.json',
                        help='Json filename')
    parser.add_argument("-p",
                        "--data_path",
                        default='',
                        help='Csv filename')
    args = parser.parse_args()

    #print("frame,xmin,xmax,ymin,ymax,class_id")
    js = load_json(args.data_path, args.json)
    js = js["_via_img_metadata"]
    keys = js.keys()
    masks = []
    for key in keys:
        entry = js[key]
        filename = entry["filename"]
        regions = entry["regions"]
        for region in regions:
            shape = region["shape_attributes"]
            x = shape["all_points_x"]
            y = shape["all_points_y"]
            name = region["region_attributes"]
            class_id = name["Name"]
            fmt = "%s,%s,%s,%s"
            line = fmt % (filename, x, y, class_id)
            print(line)
            xy = np.array([x, y], dtype=np.int32)
            xy = np.transpose(xy)
            xy = np.reshape(xy, [1, -1, 2])
            mask = { class_id : xy }
            masks.append(mask)
        break
            # print(filename,",",xmin, ",", ymin, ",", str(xmax), ",",  str(ymax), ",", class_id)

    image = plt.imread(filename)
    image = np.zeros_like(image)
    shape = image.shape
    shape = (shape[0], shape[1])
    bg = np.ones(shape, dtype="uint8")
    bg.fill(255)
    print(image.dtype)
    print(image.shape)
    print(bg.shape)
    xy = []
    rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for mask in masks:
        name = list(mask)[0]
        mask = mask[name]
        xy.append(mask)
        cv2.fillPoly(image, mask, rgb[int(name)-1])
        cv2.fillPoly(bg, mask, 0)

    plt.imshow(image)
    plt.show()

    plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
    plt.show()

    shape = (*shape, 1)
    bg = np.reshape(bg, shape)
    print(bg.shape)
    data = np.concatenate((bg, image), axis=-1)
    data = data.astype('float32') / 255
    print(data.shape)
    print(data.dtype)
    print(np.amax(data))
    print(np.sum(data[:,:,0]))
    print(np.sum(data[:,:,1]))
    print(np.sum(data[:,:,2]))
    print(np.sum(data[:,:,3]))
    data_dict = {}
    data_dict[filename] = data
    np.save("segmentation_train.npy", data_dict)
    

