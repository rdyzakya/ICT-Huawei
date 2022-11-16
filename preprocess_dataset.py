from PIL import Image
import datasets
import os
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

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
    class_path = os.path.join(label_path, "_classes.txt")
    with open(class_path, "r") as f:
        classes = eval(f.read().splitlines()[0])

    # filter image from jpg and png
    image_files = [file for file in image_files if file.endswith('.jpg') or file.endswith('.png')]

    ds = {
        "file_name" : [],
        "image_id": [],
        "image" : [],
        "image_tensor" : [],
        "width" : [],
        "height" : [],
        "objects" : []
    }

    for image_index,image_file in enumerate(image_files):
        label_file = image_file.replace(".jpg",".txt") if image_file.endswith(".jpg") else image_file.replace(".png",".txt")
        # image
        image = Image.open(os.path.join(image_path,image_file))
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

        # convert to tensor
        image_tensor = pil_to_tensor(image)

        ds["file_name"].append(image_file)
        ds["image_id"].append(image_index)
        ds["image"].append(image)
        ds["image_tensor"].append(image_tensor)
        ds["width"].append(width)
        ds["height"].append(height)
        ds["objects"].append(objects)

    ds = datasets.Dataset.from_dict(ds)
    # change ds.features['objects'] to Sequence
    ds.features['objects'] = datasets.Sequence(ds.features['objects'])
    ds.features['objects'].feature['category'] = datasets.ClassLabel(names=classes)
    return ds

def coco_format_annotation(ds):
    result = []
    for i in range(len(ds)):
        objects = ds[i]["objects"]
        annotations = {
            "image_id" : ds[i]["image_id"],
            "annotations" : []
        }
        for j in range(len(objects["id"])):
            annotations["annotations"].append({
                "image_id": ds[i]["image_id"],
                "category_id": objects["category"][j],
                "bbox": objects["bbox"][j],
                "area": objects["area"][j],
                "id": objects["id"][j]
            })
        result.append(annotations)
    
    return result