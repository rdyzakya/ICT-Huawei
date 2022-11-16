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

from preprocess_dataset import read_dataset

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

def train(args,model,feature_extractor,dataset,train_args):
    training_arguments = transformers.TrainingArguments(**train_args)
    # trainer = transformers.Trainer(
    #     model=model,
    #     args=training_arguments,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["val"],
    #     tokenizer=feature_extractor,
    #     compute_metrics=datasets.load_metric("coco")
    # )

def main():
    args = init_args()
    # Get training args
    train_args = json.load(open(args.train_args, "r"))

    # Prepare dataset
    dataset = {}

    if args.do_predict:
        train_base_path = os.path.join(args.data_dir, args.train)
        dataset["train"] = read_dataset(os.path.join(train_base_path,"Images"), os.path.join(train_base_path,"Labels"), format="yolo")
    if args.do_eval:
        eval_base_path = os.path.join(args.data_dir, args.val)
        dataset["val"] = read_dataset(os.path.join(eval_base_path,"Images"), os.path.join(eval_base_path,"Labels"), format="yolo")
    if args.do_predict:
        test_base_path = os.path.join(args.data_dir, args.testr)
        dataset["test"] = read_dataset(os.path.join(test_base_path,"Images"), os.path.join(test_base_path,"Labels"), format="yolo")
    
    dataset = datasets.DatasetDict(dataset)

    model = AutoModelForObjectDetection.from_pretrained(args.model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)

    if args.do_train:
        pass
    if args.do_predict:
        pass

if __name__ == "__main__":
    main()