"""

python3 -m json.tool < via.json > grab.json

"""


import json
import argparse
import os
import copy


def load_json(data_path, jsfile):
    with open(os.path.join(data_path, jsfile), 'r') as f:
        js = json.load(f)

    return js



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j",
                        "--json",
                        default='labels_train.json',
                        help='Json filename')
    parser.add_argument("-c",
                        "--csv",
                        default='labels_train.csv',
                        help='Csv filename')
    parser.add_argument("-p",
                        "--data_path",
                        default='dataset/drinks',
                        help='Csv filename')
    parser.add_argument("-f",
                        "--factor",
                        default=1.575,
                        type=float,
                        help='Csv filename')
    args = parser.parse_args()

    js = load_json(args.data_path, args.json)
    meta = js["_via_img_metadata"]
    keys = meta.keys()
    for key in keys:
        entry = meta[key]
        filename = entry["filename"]
        path = os.path.join(args.data_path, filename)
        statinfo = os.stat(path)
        filesize = statinfo.st_size
        filesize = int(filesize)
        js["_via_img_metadata"][key]["size"] = filesize
        regions = entry["regions"]
        for i, region in enumerate(regions):
            shape = region["shape_attributes"]
            x = float(shape["x"]) / args.factor
            y = float(shape["y"]) / args.factor
            width = float(shape["width"]) / args.factor
            height = float(shape["height"]) / args.factor
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)
            js["_via_img_metadata"][key]["regions"][i]["shape_attributes"]["x"] = x
            js["_via_img_metadata"][key]["regions"][i]["shape_attributes"]["y"] = y
            js["_via_img_metadata"][key]["regions"][i]["shape_attributes"]["width"] = width
            js["_via_img_metadata"][key]["regions"][i]["shape_attributes"]["height"] = height
            # dictionary[new_key] = dictionary.pop(old_key)

    newjs = copy.deepcopy(js)
    for key in keys:
        entry = meta[key]
        filename = entry["filename"]
        path = os.path.join(args.data_path, filename)
        statinfo = os.stat(path)
        filesize = statinfo.st_size
        newkey = filename + str(filesize)
        newjs["_via_img_metadata"][newkey] = js["_via_img_metadata"][key]
        del newjs["_via_img_metadata"][key]

    with open('resized.json', 'w') as outfile:
        json.dump(newjs, outfile)
        # print(filename,",",xmin, ",", ymin, ",", str(xmax), ",",  str(ymax), ",", class_id)

    # print(keys)


    

