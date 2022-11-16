#!/usr/bin/env bash

input_dir="./dataset_masks/test/Annotations"
image_dir="./dataset_masks/test/Images"
output_dir="./dataset_masks/test/Labels"

python preprocess_xml.py --input ${input_dir} --image ${image_dir} --output ${output_dir}