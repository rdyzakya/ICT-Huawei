import argparse
# from torchvision.ops import box_convert
# from torchvision.utils import draw_bounding_boxes
# from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import math

import torch
# from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, AutoConfig
from transformers import get_scheduler
import eval_utils

import datasets
from preprocess_dataset import read_dataset, coco_format_annotation

import os
import json

from tqdm import tqdm

# https://huggingface.co/docs/datasets/object_detection

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/detr-resnet-50", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")

    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--train", type=str, default="train", help="Path to train folder")
    parser.add_argument("--val", type=str, default="val", help="Path to val folder")
    parser.add_argument("--test", type=str, default="test", help="Path to test folder")

    parser.add_argument("--n_gpu", type=str, default="0", help="Index of GPU to use")

    parser.add_argument("--do_train", action="store_true", help="Train the model")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--do_predict", action="store_true", help="Predict with the model")

    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for evaluation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--train_args", type=str, default="train_args.json", help="Path to train args file")

    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for localizations")

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return args

def train(args,model,feature_extractor,dataset,annotations,train_args):
    torch.cuda.empty_cache()
    # For native pt : https://huggingface.co/docs/transformers/training#train-in-native-pytorch
    inputs = {}
    # dataloaders = {}

    # Feature extract the dataset
    # https://stackoverflow.com/questions/67691530/key-error-while-fine-tunning-t5-for-summarization-with-huggingface
    inputs["train"] = feature_extractor(images=dataset["train"]["image"], annotations=annotations["train"], return_tensors="pt")
    # dataloaders["train"] = DataLoader(inputs["train"], shuffle=False, batch_size=train_args["per_device_train_batch_size"])

    if args.do_eval:
        inputs["val"] = feature_extractor(images=dataset["val"]["image"], annotations=annotations["val"], return_tensors="pt")
        # dataloaders["val"] = DataLoader(inputs["val"], shuffle=False, batch_size=train_args["per_device_eval_batch_size"])

    lr = train_args["learning_rate"] or 5e-5
    adam_epsilon = train_args["adam_epsilon"] or 1e-8
    weight_decay = train_args["weight_decay"] or 0.0
    betas = (train_args["adam_beta1"] or 0.9, train_args["adam_beta2"] or 0.999)
    optimizer = AdamW(model.parameters() , lr=lr, eps=adam_epsilon, weight_decay=weight_decay, betas=betas)

    num_epochs = train_args["num_train_epochs"] or 3
    num_training_steps = num_epochs * math.ceil(len(inputs["train"]) / train_args["per_device_train_batch_size"])
    num_warmup_steps = train_args["warmup_steps"] or 0
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Start training...")
    print("Training arguments:")
    print(train_args)
    # progress_bar = tqdm(range(num_training_steps))

    for epoch in tqdm(range(num_epochs)):
        print("Epoch : ", epoch)
        model.train()
        train_batch_size = train_args["per_device_train_batch_size"]
        for i in range(0, len(inputs["train"]["pixel_values"]), train_batch_size):
            pixel_values = inputs["train"]["pixel_values"][i:i+train_batch_size].to(device)
            labels = inputs["train"]["labels"][i:i+train_batch_size]
            for j in range(len(labels)):
                label = labels[j]
                for k in label.keys():
                    # to device
                    labels[j][k] = label[k].to(device)
            batch = {"pixel_values" : pixel_values, "labels" : labels}
            outputs = model(**batch)
            loss = outputs.loss
            print("Training loss : ", loss.item())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar.update(1)
        
        if args.do_eval:
            print("Do evaluation...")
            model.eval()
            eval_batch_size = train_args["per_device_eval_batch_size"]
            results = []
            for i in range(0, len(inputs["val"]["pixel_values"]), eval_batch_size):
                pixel_values = inputs["val"]["pixel_values"][i:i+eval_batch_size].to(device)
                labels = inputs["val"]["labels"][i:i+eval_batch_size]
                for j in range(len(labels)):
                    label = labels[j]
                    for k in label.keys():
                        # to device
                        labels[j][k] = label[k].to(device)
                batch = {"pixel_values" : pixel_values, "labels" : labels}

                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                print("Evaluation loss : ", loss.item())
                target_sizes = torch.tensor([
                    image.size[::-1] for image in dataset["val"]["image"][i:i+eval_batch_size]
                ])

                for k in outputs.keys():
                    if isinstance(outputs[k], torch.Tensor):
                        outputs[k] = outputs[k].to('cpu')

                r = feature_extractor.post_process_object_detection(
                    outputs, threshold=args.threshold, target_sizes=target_sizes
                )

                results.extend(r)

            ground_truths = dataset["val"]["objects"]

            evaluation_score = eval_utils.map_score(ground_truths, results, args.iou_threshold)

            print("Evaluation score : ", evaluation_score)
    model.save_pretrained(args.output_dir)
            #     batch = {k: v.to(device) for k, v in batch.items()}
            #     with torch.no_grad():
            #         outputs = model(**batch)

            #     logits = outputs.logits
            #     predictions = torch.argmax(logits, dim=-1)
            #     metric.add_batch(predictions=predictions, references=batch["labels"])
    # Prepare the training arguments
    # train_args.update({
    #     "do_train": args.do_train,
    #     "do_eval": args.do_eval,
    #     "do_predict": args.do_predict,
    # })
    # For reference : https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    # training_arguments = transformers.TrainingArguments(**train_args)
    # trainer_args = {
    #     "model": model,
    #     "args": training_arguments,
    #     "train_dataset": inputs_train,
    #     "tokenizer": feature_extractor,
    # }
    # # If evaluation then...
    # if args.do_eval:
    #     inputs_val = feature_extractor(images=dataset["val"]["image"], annotations=annotations["val"], return_tensors="pt")
    #     trainer_args["eval_dataset"] = inputs_val
    # # Prepare the trainer
    # trainer = transformers.Trainer(**trainer_args)
    # # Train the model
    # print("Start the training...")
    # trainer.train()

def main():
    args = init_args()
    # Get training args
    train_args = json.load(open(args.train_args, "r"))
    config = json.load(open(args.config, "r"))

    # Prepare dataset
    print("Preparing dataset...")
    dataset = {}
    annotations = {}

    if args.do_predict:
        train_base_path = os.path.join(args.data_dir, args.train)
        dataset["train"] = read_dataset(os.path.join(train_base_path,"Images"), os.path.join(train_base_path,"Labels"), format="coco")
        # annotations["train"] = dataset["train"].map(coco_format_annotation, batched=True, remove_columns=dataset["train"].column_names)
        annotations["train"] = coco_format_annotation(dataset["train"])
    if args.do_eval:
        eval_base_path = os.path.join(args.data_dir, args.val)
        dataset["val"] = read_dataset(os.path.join(eval_base_path,"Images"), os.path.join(eval_base_path,"Labels"), format="coco")
        # annotations["val"] = dataset["val"].map(coco_format_annotation, batched=True, remove_columns=dataset["val"].column_names)
        annotations["val"] = coco_format_annotation(dataset["val"])
    if args.do_predict:
        test_base_path = os.path.join(args.data_dir, args.test)
        dataset["test"] = read_dataset(os.path.join(test_base_path,"Images"), os.path.join(test_base_path,"Labels"), format="coco")
        # annotations["test"] = dataset["test"].map(coco_format_annotation, batched=True, remove_columns=dataset["test"].column_names)
        annotations["test"] = coco_format_annotation(dataset["test"])

    dataset = datasets.DatasetDict(dataset)
    annotations = datasets.DatasetDict(annotations)

    print("Loading model...")

    model_config = AutoConfig.from_pretrained(args.model_name_or_path, **config)
    model = AutoModelForObjectDetection.from_pretrained(args.model_name_or_path,config=config)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path,size=256)

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