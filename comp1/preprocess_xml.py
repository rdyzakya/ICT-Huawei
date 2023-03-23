import xml.etree.ElementTree as ET
import glob
import os
import json

import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/annotations', help='path to input annotations')
    parser.add_argument('--output', type=str, default='data/annotations', help='path to output annotations')
    parser.add_argument('--image', type=str, default='data/images', help='path to images')
    parser.add_argument('--format', type=str, default='yolo', help='format to convert to (coco,yolo,xml)')
    parser.add_argument('--normalize', action='store_true', help='normalize bounding boxes')
    args = parser.parse_args()

    return args

# https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5
# https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco


def xml_to_yolo_bbox(bbox, w, h, normalize=True):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w if normalize else ((bbox[2] + bbox[0]) / 2)
    y_center = ((bbox[3] + bbox[1]) / 2) / h if normalize else ((bbox[3] + bbox[1]) / 2)
    width = (bbox[2] - bbox[0]) / w if normalize else (bbox[2] - bbox[0])
    height = (bbox[3] - bbox[1]) / h if normalize else (bbox[3] - bbox[1])
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h, normalize=True):
    # x_center, y_center, width, height
    w_half_len = (bbox[2] * w) / 2 if normalize else (bbox[2] / 2)
    h_half_len = (bbox[3] * h) / 2 if normalize else (bbox[3] / 2)
    xmin = int((bbox[0] * w) - w_half_len) if normalize else int(bbox[0] - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len) if normalize else int(bbox[1] - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len) if normalize else int(bbox[0] + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len) if normalize else int(bbox[1] + h_half_len)
    return [xmin, ymin, xmax, ymax]

def xml_to_coco_bbox(bbox, w, h, normalize=True):
    # xmin, ymin, xmax, ymax
    x_min = bbox[0] / w if normalize else bbox[0]
    y_min = bbox[1] / h if normalize else bbox[1]
    width = (bbox[2] - bbox[0]) / w if normalize else (bbox[2] - bbox[0])
    height = (bbox[3] - bbox[1]) / h if normalize else (bbox[3] - bbox[1])
    return [x_min, y_min, width, height]

def main():
    args = init_args()

    classes = []
    input_dir = args.input
    output_dir = args.output
    image_dir = args.image

    # create the labels folder (output directory)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(input_dir, '*.xml'))
    # loop through each 
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")) and not os.path.exists(os.path.join(image_dir, f"{filename}.png")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            # bbox = xml_to_yolo_bbox(pil_bbox, width, height) if args.format == 'yolo' else xml_to_coco_bbox(pil_bbox, width, height)
            bbox = None
            if args.format == 'yolo':
                bbox = xml_to_yolo_bbox(pil_bbox, width, height, args.normalize)
            elif args.format == 'coco':
                bbox = xml_to_coco_bbox(pil_bbox, width, height, args.normalize)
            elif args.format == 'xml':
                bbox = pil_bbox
            else:
                print("Invalid format")
                exit(1)
            # convert data to string
            bbox_string = " ".join([str(x) for x in bbox])
            # bbox_string = " ".join([str(x) for x in pil_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    # generate the classes file as reference
    with open(os.path.join(args.output,'_classes.txt'), 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))

if __name__ == '__main__':
    main()