input_dir = "./dataset_masks/train/Annotations"
image_dir = "./dataset_masks/train/Images"
output_dir = "./dataset_masks/train/Labels"

python preprocessing.py --input ${input_dir} --image ${image_dir} --output ${output_dir}