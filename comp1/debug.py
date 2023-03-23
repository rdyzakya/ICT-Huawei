from preprocess_dataset import read_dataset, coco_format_annotation, map_coco_annotation

image_path = "./dataset_masks/train/Images"
label_path = "./dataset_masks/train/Labels"

ds = read_dataset(image_path, label_path, format="coco")

tes = ds.map(map_coco_annotation, batched=False, remove_columns=ds.column_names)
print(tes[0])