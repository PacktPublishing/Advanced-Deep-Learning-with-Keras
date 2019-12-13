'''

Pretty print: python3 -m json.tool < some.json

'''

import json
import argparse
import os


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
                        default='',
                        help='Csv filename')
    args = parser.parse_args()

    print("frame,xmin,xmax,ymin,ymax,class_id")
    js = load_json(args.data_path, args.json)
    js = js["_via_img_metadata"]
    keys = js.keys()
    for key in keys:
        entry = js[key]
        filename = entry["filename"]
        regions = entry["regions"]
        for region in regions:
            shape = region["shape_attributes"]
            xmin = shape["x"]
            ymin = shape["y"]
            xmax = int(xmin) + int(shape["width"])
            ymax = int(ymin) + int(shape["height"])
            name = region["region_attributes"]
            class_id = name["name"]
            fmt = "%s,%s,%s,%d,%d,%s"
            line = fmt % (filename, xmin, xmax, ymin, ymax, class_id)
            print(line)
            # print(filename,",",xmin, ",", ymin, ",", str(xmax), ",",  str(ymax), ",", class_id)

    # print(keys)


    

