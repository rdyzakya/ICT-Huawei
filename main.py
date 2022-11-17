import argparse

import math

import torch
from torch.optim import AdamW
import transformers
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, AutoConfig
from transformers import DetrConfig, DetrForObjectDetection, DetrFeatureExtractor
from transformers import ConditionalDetrConfig, ConditionalDetrForObjectDetection, ConditionalDetrFeatureExtractor
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection, DeformableDetrFeatureExtractor
from transformers import TableTransformerConfig, TableTransformerForObjectDetection
from transformers import YolosConfig, YolosForObjectDetection, YolosFeatureExtractor
from transformers import get_scheduler
import eval_utils

import datasets
from preprocess_dataset import read_dataset, coco_format_annotation

import os
import json

# https://danielvanstrien.xyz/huggingface/huggingface-datasets/transformers/2022/08/16/detr-object-detection.html

from tqdm import tqdm

# https://huggingface.co/docs/datasets/object_detection

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="detr", type=str, help="Model type selected in the list: " + ", ".join(transformers.MODEL_FOR_OBJECT_DETECTION_MAPPING.keys()))
    parser.add_argument("--model_name_or_path", type=str, default="facebook/detr-resnet-50", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")

    parser.add_argument("--size", type=int, default=512, help="Image size")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--train", type=str, default="train", help="Path to train folder")
    parser.add_argument("--val", type=str, default="val", help="Path to val folder")
    parser.add_argument("--test", type=str, default="test", help="Path to test folder")

    parser.add_argument("--n_gpu", type=str, default="0", help="Index of GPU to use")

    parser.add_argument("--do_train", action="store_true", help="Train the model")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--do_predict", action="store_true", help="Predict with the model")

    parser.add_argument("--per_device_predict_batch_size", type=int, default=8, help="Batch size for prediction")

    parser.add_argument("--native", action="store_true", help="Train the model using native pytorch")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for evaluation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--model_args", type=str, default="model_args.json", help="Path to model args file")
    parser.add_argument("--feature_extractor_args", type=str, default="feature_extractor_args.json", help="Path to feature extractor args file")
    parser.add_argument("--train_args", type=str, default="train_args.json", help="Path to train args file")

    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for localizations")

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return args

def collate_fn(batch,feature_extractor):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

def transform(example_batch,feature_extractor):
    images = example_batch["image"]
    ids_ = example_batch["image_id"]
    objects = example_batch["objects"]
    targets = [
        {"image_id": id_, "annotations": object_} for id_, object_ in zip(ids_, objects)
    ]
    return feature_extractor(images=images, annotations=targets, return_tensors="pt")

def load_model(model_type,model_name_or_path,config,model_args={},feature_extractor_args={}):
    if model_type == "detr":
        model_config = DetrConfig.from_pretrained(model_name_or_path,**config)
        model = DetrForObjectDetection.from_pretrained(model_name_or_path,config=model_config,**model_args)
        feature_extractor = DetrFeatureExtractor.from_pretrained(model_name_or_path,**feature_extractor_args)

        return model,feature_extractor
    if model_type == "conditional_detr":
        model_config = ConditionalDetrConfig.from_pretrained(model_name_or_path,**config)
        model = ConditionalDetrForObjectDetection.from_pretrained(model_name_or_path,config=model_config,**model_args)
        feature_extractor = ConditionalDetrFeatureExtractor.from_pretrained(model_name_or_path,**feature_extractor_args)

        return model,feature_extractor
    if model_type == "deformable_detr":
        model_config = DeformableDetrConfig.from_pretrained(model_name_or_path,**config)
        model = DeformableDetrForObjectDetection.from_pretrained(model_name_or_path,config=model_config,**model_args)
        feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(model_name_or_path,**feature_extractor_args)

        return model,feature_extractor
    if model_type == "table_transformer":
        model_config = TableTransformerConfig.from_pretrained(model_name_or_path,**config)
        model = TableTransformerForObjectDetection.from_pretrained(model_name_or_path,config=model_config,**model_args)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path,**feature_extractor_args)

        return model,feature_extractor
    if model_type == "yolos":
        model_config = YolosConfig.from_pretrained(model_name_or_path,**config)
        model = YolosForObjectDetection.from_pretrained(model_name_or_path,config=model_config,**model_args)
        feature_extractor = YolosFeatureExtractor.from_pretrained(model_name_or_path,**feature_extractor_args)

        return model,feature_extractor
    
    model_config = AutoConfig.from_pretrained(model_name_or_path,**config)
    model = AutoModelForObjectDetection.from_pretrained(model_name_or_path,config=model_config,**model_args)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path,**feature_extractor_args)

    return model,feature_extractor

def train_hf(args,model,feature_extractor,dataset,annotations,train_args):
    torch.cuda.empty_cache()

    inputs = {}

    # inputs["train"] = dataset["train"].with_transform(transform)
    # dataloaders = {}

    # Feature extract the dataset
    # https://stackoverflow.com/questions/67691530/key-error-while-fine-tunning-t5-for-summarization-with-huggingface
    inputs["train"] = feature_extractor(images=dataset["train"]["image"], annotations=annotations["train"], return_tensors="pt")

    if args.do_eval:
        inputs["val"] = feature_extractor(images=dataset["val"]["image"], annotations=annotations["val"], return_tensors="pt")
    
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        seed=args.random_seed,
        **train_args
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=lambda x : collate_fn(x,feature_extractor),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=feature_extractor,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)

def train_native(args,model,feature_extractor,dataset,annotations,train_args):
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
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
    }

    for epoch in tqdm(range(num_epochs)):
        print("Epoch : ", epoch)
        model.train()
        train_batch_size = train_args["per_device_train_batch_size"]
        train_loss_value = 0
        eval_loss_value = 0
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
            # print("Training loss : ", loss.item())
            train_loss_value += loss.item()
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
                # print("Evaluation loss : ", loss.item())
                eval_loss_value += loss.item()
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
            history["val_map"].append(evaluation_score)

        current_epoch_train_loss = train_loss_value/len(inputs["train"]["pixel_values"])
        print("Average training loss : ", current_epoch_train_loss)
        history["train_loss"].append(current_epoch_train_loss)
        if args.do_eval:
            current_epoch_eval_loss = eval_loss_value/len(inputs["val"]["pixel_values"])
            print("Average evaluation loss : ", current_epoch_eval_loss)
            history["val_loss"].append(current_epoch_eval_loss)
    
    model.save_pretrained(args.output_dir)

    #write history to json
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f)

    return history

def predict(args, model, feature_extractor, dataset, annotations):
    test_batch_size = args.per_device_predict_batch_size
    inputs_test = feature_extractor(images=dataset["test"]["image"], annotations=annotations["test"], return_tensors="pt")
    device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    results = []
    test_loss_value = 0
    for i in range(0,len(inputs_test["pixel_values"]),test_batch_size):
        pixel_values = inputs_test["pixel_values"][i:i+test_batch_size].to(device)
        labels = inputs_test["labels"][i:i+test_batch_size]
        for j in range(len(labels)):
            label = labels[j]
            for k in label.keys():
                # to device
                labels[j][k] = label[k].to(device)
        batch = {"pixel_values" : pixel_values, "labels" : labels}

        with torch.no_grad():
            outputs = model(**batch)
        
        loss = outputs.loss
        test_loss_value += loss.item()

        target_sizes = torch.tensor([
            image.size[::-1] for image in dataset["test"]["image"][i:i+test_batch_size]
        ])

        for k in outputs.keys():
            if isinstance(outputs[k], torch.Tensor):
                outputs[k] = outputs[k].to('cpu')
        
        r = feature_extractor.post_process_object_detection(
            outputs, threshold=args.threshold, target_sizes=target_sizes
        )

        results.extend(r)
    
    current_test_loss = test_loss_value/len(inputs_test["pixel_values"])
    print("Average test loss : ", current_test_loss)

    ground_truths = dataset["test"]["objects"]

    evaluation_score = eval_utils.map_score(ground_truths, results, args.iou_threshold)
    print("Evaluation score : ", evaluation_score)

    # write to metrics_and_loss.json and results.json
    metrics_and_loss = {"test_loss" : current_test_loss, "test_map" : evaluation_score}
    with open(os.path.join(args.output_dir, "metrics_and_loss.json"), "w") as f:
        json.dump(metrics_and_loss, f)

    for i in range(len(results)):
        for k in results[i].keys():
            if isinstance(results[i][k], torch.Tensor):
                results[i][k] = results[i][k].tolist()

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
    
    return results, metrics_and_loss

def main():
    args = init_args()
    # Get training args
    train_args = json.load(open(args.train_args, "r"))
    config = json.load(open(args.config, "r"))
    model_args = json.load(open(args.model_args, "r"))
    feature_extractor_args = json.load(open(args.feature_extractor_args, "r"))

    # Prepare dataset
    print("Preparing dataset...")
    dataset = {}
    annotations = {}

    if args.do_predict:
        train_base_path = os.path.join(args.data_dir, args.train)
        dataset["train"] = read_dataset(os.path.join(train_base_path,"Images"), os.path.join(train_base_path,"Labels"), format="coco")
        annotations["train"] = coco_format_annotation(dataset["train"])
    if args.do_eval:
        eval_base_path = os.path.join(args.data_dir, args.val)
        dataset["val"] = read_dataset(os.path.join(eval_base_path,"Images"), os.path.join(eval_base_path,"Labels"), format="coco")
        annotations["val"] = coco_format_annotation(dataset["val"])
    if args.do_predict:
        test_base_path = os.path.join(args.data_dir, args.test)
        dataset["test"] = read_dataset(os.path.join(test_base_path,"Images"), os.path.join(test_base_path,"Labels"), format="coco")
        annotations["test"] = coco_format_annotation(dataset["test"])

    dataset = datasets.DatasetDict(dataset)
    annotations = datasets.DatasetDict(annotations)

    print("Loading model...")

    # model_config = AutoConfig.from_pretrained(args.model_name_or_path, **config)
    # model = AutoModelForObjectDetection.from_pretrained(args.model_name_or_path,config=model_config,ignore_mismatched_sizes=True,num_labels=2)
    # feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path,size=args.size)
    model, feature_extractor = load_model(args.model_type,args.model_name_or_path,config,model_args,feature_extractor_args)

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
        training_function = train_native if args.native else train_hf
        history = training_function(args,model,feature_extractor,dataset,annotations,train_args)
    if args.do_predict:
        results, metrics_and_loss = predict(args,model,feature_extractor,dataset,annotations)

if __name__ == "__main__":
    main()