import argparse
import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import transformers
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

import datasets
from PIL import Image

import os
import json

from preprocess_dataset import read_dataset, coco_format_annotation

# https://huggingface.co/docs/datasets/object_detection

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/detr-resnet-50", help="Model name or path")
    
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--train", type=str, default="train", help="Path to train folder")
    parser.add_argument("--val", type=str, default="val", help="Path to val folder")
    parser.add_argument("--test", type=str, default="test", help="Path to test folder")

    parser.add_argument("--do_train", action="store_true", help="Train the model")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--do_predict", action="store_true", help="Predict with the model")

    parser.add_argument("--train_args", type=str, default="train_args.json", help="Path to train args file")

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return args

def train(args,model,feature_extractor,dataset,annotations,train_args):
    # For native pt : https://huggingface.co/docs/transformers/training#train-in-native-pytorch

    # Feature extract the dataset
    inputs_train = feature_extractor(images=dataset["train"]["image"], annotations=annotations["train"], return_tensors="pt")
    # Prepare the training arguments

    # For reference : https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    training_arguments = transformers.TrainingArguments(**train_args)
    trainer_args = {
        "model": model,
        "args": training_arguments,
        "train_dataset": inputs_train,
        "tokenizer": feature_extractor,
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "do_predict": args.do_predict,
    }
    # If evaluation then...
    if args.do_eval:
        inputs_val = feature_extractor(images=dataset["val"]["image"], annotations=annotations["val"], return_tensors="pt")
        trainer_args["eval_dataset"] = inputs_val
    # Prepare the trainer
    trainer = transformers.Trainer(**trainer_args)
    # Train the model
    print("Start the training...")
    trainer.train()

def main():
    args = init_args()
    # Get training args
    train_args = json.load(open(args.train_args, "r"))

    # Prepare dataset
    print("Preparing dataset...")
    dataset = {}
    annotations = {}

    if args.do_predict:
        train_base_path = os.path.join(args.data_dir, args.train)
        dataset["train"] = read_dataset(os.path.join(train_base_path,"Images"), os.path.join(train_base_path,"Labels"), format="coco")
        annotations["train"] = dataset["train"].map(coco_format_annotation, batched=True, remove_columns=dataset["train"].column_names)
    if args.do_eval:
        eval_base_path = os.path.join(args.data_dir, args.val)
        dataset["val"] = read_dataset(os.path.join(eval_base_path,"Images"), os.path.join(eval_base_path,"Labels"), format="coco")
        annotations["val"] = dataset["val"].map(coco_format_annotation, batched=True, remove_columns=dataset["val"].column_names)
    if args.do_predict:
        test_base_path = os.path.join(args.data_dir, args.test)
        dataset["test"] = read_dataset(os.path.join(test_base_path,"Images"), os.path.join(test_base_path,"Labels"), format="coco")
        annotations["test"] = dataset["test"].map(coco_format_annotation, batched=True, remove_columns=dataset["test"].column_names)
    
    dataset = datasets.DatasetDict(dataset)
    annotations = datasets.DatasetDict(annotations)

    print("Loading model...")

    model = AutoModelForObjectDetection.from_pretrained(args.model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)

    # https://huggingface.co/docs/transformers/model_doc/yolos#transformers.YolosFeatureExtractor.__call__.annotations
    # annotations (Dict, List[Dict], optional) — The corresponding annotations in COCO format.
    # In case DetrFeatureExtractor was initialized with format = "coco_detection", 
    # the annotations for each image should have the following format: {‘image_id’: int, ‘annotations’: [annotation]},
    #  with the annotations being a list of COCO object annotations.

    # In case DetrFeatureExtractor was initialized with format = "coco_panoptic", the annotations 
    # for each image should have the following format: {‘image_id’: int, ‘file_name’: str, ‘segments_info’: 
    # [segment_info]} with segments_info being a list of COCO panoptic annotations.
    
    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

    # https://towardsdatascience.com/on-object-detection-metrics-with-worked-example-216f173ed31e

    if args.do_train:
        train(args,model,feature_extractor,dataset,annotations,train_args)
    if args.do_predict:
        pass

if __name__ == "__main__":
    main()