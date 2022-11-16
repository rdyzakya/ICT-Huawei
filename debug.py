from preprocess_dataset import read_dataset, coco_format_annotation

image_path = "./dataset_masks/train/Images"
label_path = "./dataset_masks/train/Labels"

ds = read_dataset(image_path, label_path, format="coco")