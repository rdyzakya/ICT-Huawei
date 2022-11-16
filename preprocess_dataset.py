import argparse
from PIL import Image
import datasets
import os


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='data/images', help='path to images')
    parser.add_argument('--label', type=str, default='data/labels', help='path to labels')
    parser.add_argument("--format", type=str, default="yolo", help="format of label (coco,label,xml)")
    args = parser.parse_args()

    return args

def count_area(bbox, format = "yolo"):
    if format == "yolo":
        # x_center, y_center, width, height
        return bbox[2] * bbox[3]
    elif format == "coco":
        # xmin, ymin, width, height
        return bbox[2] * bbox[3]
    elif format == "xml":
        # xmin, ymin, xmax, ymax
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def read_dataset(image_path,label_path,format="yolo"):
    image_files = os.listdir(image_path)

    # filter image from jpg and png
    image_files = [file for file in image_files if file.endswith('.jpg') or file.endswith('.png')]

    ds = {
        "image_id": [],
        "image" : [],
        "width" : [],
        "height" : [],
        "objects" : []
    }

    for image_file in image_files:
        label_file = image_file.replace(".jpg",".txt") if image_file.endswith(".jpg") else image_file.replace(".png",".txt")
        # image
        image = Image.open(os.path.join(image_path, image_file))
        with open(os.path.join(label_path, label_file)) as f:
            label = f.read().splitlines()
        label = [el.split() for el in label]

        # objects
        category = [] # list of category
        bbox = [] # list of bbox

        for l in label:
            category.append(int(l[0]))
            bbox.append([float(el) for el in l[1:]]) # xmin, ymin, xmax, ymax
        area = []
        for b in bbox:
            area.append(count_area(b, format = format))
        bbox_id = [i for i in range(len(bbox))] # id of bbox
        width, height = image.size # width and height of image

        objects = {
            "id" : bbox_id,
            "area" : area,
            "bbox" : bbox,
            "category" : category
        }

        ds["image_id"].append(image_file)
        ds["image"].append(image)
        ds["width"].append(width)
        ds["height"].append(height)
        ds["objects"].append(objects)

    return datasets.Dataset.from_dict(ds)