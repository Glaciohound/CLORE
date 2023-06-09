import scipy.io as sio
import math
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
from contextlib import nullcontext
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.datasets import ImageFolder

from models.guided_bert_cub import GuidedBertCUB
from models.guided_clip_cub import GuidedCLIPCUB
from models.simple_clip_model import SimpleCLIPModel
from experiments.utils import RunningMean, Mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="bert-base-uncased")
    parser.add_argument("--vision_model", type=str, default="IMAGENET1K_V2")
    parser.add_argument("--vision_num_patches", type=int, default=49)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--definition_file", type=str, required=True)
    parser.add_argument("--class_names_file", type=str, required=True)
    # parser.add_argument("--test_classes_file", type=str)
    # parser.add_argument("--train_test_split_file", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--phase", type=str, default="main",
                        choices=["main", "fix_bert", "fix_visual", "fix_both"])
    parser.add_argument("--similarity", action="store_true")
    # parser.add_argument("--subtask", type=str, default="clues_real")
    parser.add_argument("--program_length", type=int, default=10)
    # parser.add_argument("--single_task", type=str, default=None)
    # parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--train_on_all", action="store_true")
    parser.add_argument("--individual_scores", action="store_true")
    # parser.add_argument("--normalize_datasets", action="store_true")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", type=str, default="")
    parser.add_argument("--encoder_feature_layer", type=str,
                        default="transformer-encoder")
    parser.add_argument("--guideline_feature_layer", type=str,
                        default="transformer-encoder")
    parser.add_argument("--encoder_feature_layer_num_head",
                        type=int, default=8)
    parser.add_argument("--guideline_feature_layer_num_head",
                        type=int, default=8)
    parser.add_argument("--model_version", type=str, default="1")
    parser.add_argument("--batch_per_epoch", type=int, default=2500)
    parser.add_argument("--num_feature_layer", type=int, default=1)
    parser.add_argument("--mask_out_templates", type=int,
                        nargs="*", default=[])
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--program_pooler", default="sum")
    parser.add_argument("--remove_name", action="store_true")
    parser.add_argument("--remove_first_sentence", action="store_true")
    parser.add_argument("--do_not_load_optimizer", action="store_true")

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)
    return args


def get_data(data_dir, transform, split_mat, train_on_all):
    def create_split(dataset, locations):
        split = deepcopy(dataset)
        split.samples = [split.samples[_idx-1] for _idx in locations[:, 0]]
        split.imgs = split.samples
        split.targets = [split.targets[_idx-1] for _idx in locations[:, 0]]
        return split

    whole_cub = ImageFolder(data_dir, transform=transform)
    if not train_on_all:
        train_dataset = create_split(whole_cub, split_mat["train_loc"])
    else:
        train_dataset = deepcopy(whole_cub)
    val_dataset = create_split(whole_cub, split_mat["val_loc"])
    test_seen_dataset = create_split(whole_cub, split_mat["test_seen_loc"])
    test_unseen_dataset = create_split(whole_cub, split_mat["test_unseen_loc"])
    return train_dataset, val_dataset, test_seen_dataset, test_unseen_dataset


def get_definitions(definition_file, class_names_file,
                    tokenizer, device, is_clip,
                    remove_name, remove_first_sentence):
    with open(class_names_file, "r") as f:
        class_names = [_line.strip().split(".")[1].replace("_", " ")
                       for _line in f.readlines()]
    with open(definition_file, "r") as f:
        definition_lines = f.readlines()
    definitions = []
    definition_indices = []
    for i in range(200):
        assert definition_lines.pop(0).strip() == str(i+1)
        group = []
        while True and len(definition_lines) > 0:
            sentence = definition_lines.pop(0).strip()
            if sentence != "":
                if remove_name:
                    _name = class_names[i]
                    for _part in _name.split(" ")[1:]:
                        sentence = sentence.replace(_part, "bird")
                        sentence = sentence.replace(
                            _part[0].upper()+_part[1:], "bird")
                        sentence = sentence.replace(
                            _part[0].lower()+_part[1:], "bird")
                group.append(sentence)
            else:
                break
        if remove_first_sentence and len(group) > 1:
            group = group[1:]
        definitions.extend(group)
        definition_indices.extend([int(i)] * len(group))

    if not is_clip:
        tokenized = tokenizer(definitions, padding=True, max_length=512,
                              return_tensors="pt")
        tokenized = {_key: _value.to(device)
                     for _key, _value in tokenized.items()}
    else:
        import clip
        tokenized = clip.tokenize(definitions).to(device)
    return definitions, tokenized, \
        torch.LongTensor(definition_indices).to(device)


def one_epoch(dataloader, model, optimizer, scheduler, tokenizer,
              device, split, i_epoch, individual_scores,
              definitions_tokenized, definition_indices):
    def mean_of_mean(means):
        values = [_mean.value for _mean in means]
        return sum(values) / len(values)

    print(f"{split} epoch {i_epoch}")
    is_train = split == "train"
    pbar = tqdm(dataloader, leave=False)
    if split == "train":
        loss_mean = RunningMean(p=0.99)
        acc_mean = RunningMean(p=0.99)
    else:
        loss_mean = Mean()
        acc_mean = Mean()
        all_targets = []
        all_scores = []
        all_losses = []

    for batch_i, batch in enumerate(pbar):
        optimizer.zero_grad()
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.shape[0]
        with torch.no_grad() if split != "train" else nullcontext():
            scores = model(
                images, targets,
                definitions_tokenized, definition_indices, is_train)
        loss = F.cross_entropy(
            scores, targets, reduction="none")
        pred = scores.argmax(1)
        targets_onehot = F.one_hot(targets, 200)
        pred_onehot = F.one_hot(pred, 200)
        if is_train:
            (loss.sum() / batch_size).backward()
            optimizer.step()
            scheduler.step()
            task_acc = (
                ((targets_onehot == 1) * (pred_onehot == 1)).sum(0) /
                targets_onehot.sum(0).clamp_min(1)
            )
            classes_involved = targets_onehot.any(0)
            acc = task_acc[classes_involved].mean()

            loss_mean.update(loss.detach().tolist())
            acc_mean.update_one(acc.detach().tolist())
            pbar.set_description(f"loss: {loss_mean:.4f}, acc: {acc_mean:.4f}")
        else:
            all_targets.append(targets_onehot.cpu())
            all_scores.append(scores.cpu())
            all_losses.append(loss.detach().cpu())

    if is_train:
        loss_mean = loss_mean.value
        acc_mean = acc_mean.value
    else:
        all_targets = torch.cat(all_targets)
        classes_involved = all_targets.any(0)
        all_scores = torch.cat(all_scores)
        all_scores[:, classes_involved.logical_not()] = -100
        all_preds = F.one_hot(all_scores.argmax(1), 200)
        all_losses = torch.cat(all_losses)
        loss_mean = all_losses.mean()
        task_acc = (
            ((all_targets == 1) * (all_preds == 1)).sum(0) /
            all_targets.sum(0).clamp_min(1)
        )
        acc_mean = task_acc[classes_involved].mean()

    print()
    print(f"{split} epoch-{i_epoch}:"
          f" average loss: {loss_mean}, average accuracy: {acc_mean}")
    return loss_mean, acc_mean


def save_model(model, optimizer, scheduler, start_epoch, filename):
    # print(f"saving model to {filename}")
    torch.save(
        [model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
         start_epoch],
        filename
    )


def main(args):
    device = torch.device("cpu") if torch.cuda.device_count() == 0 else \
        torch.device("cuda:0")
    if args.vision_model != "clip":
        weights = {
            'IMAGENET1K_V1': ResNet101_Weights.IMAGENET1K_V1,
            'IMAGENET1K_V2': ResNet101_Weights.IMAGENET1K_V2,
        }[args.vision_model]
        vision_model = resnet101(weights=weights)
        transforms = weights.transforms()
    else:
        import clip
        vision_model, preprocess = clip.load('ViT-B/32', device)
        vision_model.float()
        transforms = preprocess

    if args.model_name_or_path != "clip":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        bert = AutoModel.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = None
        bert = vision_model

    split_mat = sio.loadmat(args.split_file)
    train_dataset, val_dataset, test_seen_dataset, test_unseen_dataset = \
        get_data(args.data_dir, transforms, split_mat,
                 args.train_on_all)
    definitions, definitions_tokenized, definition_indices = get_definitions(
        args.definition_file, args.class_names_file, tokenizer, device,
        args.model_name_or_path == "clip", args.remove_name,
        args.remove_first_sentence)

    train_dataloader = DataLoader(
        train_dataset, args.per_device_train_batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, args.per_device_eval_batch_size, shuffle=False)
    test_seen_dataloader = DataLoader(
        test_seen_dataset, args.per_device_eval_batch_size, shuffle=False)
    test_unseen_dataloader = DataLoader(
        test_unseen_dataset, args.per_device_eval_batch_size, shuffle=False)

    if args.vision_model != "clip" and not args.clip:
        model = GuidedBertCUB(
            vision_model, bert,
            args.vision_num_patches, args.phase, args.similarity,
            args.encoder_feature_layer, args.guideline_feature_layer,
            args.encoder_feature_layer_num_head,
            args.guideline_feature_layer_num_head,
            args.num_feature_layer,
            args.model_version, args.mask_out_templates
        ).to(device)
    elif args.clip:
        import clip
        model = SimpleCLIPModel(vision_model, args.phase).to(device)
    else:
        model = GuidedCLIPCUB(
            vision_model, args.phase, args.similarity,
            args.encoder_feature_layer, args.guideline_feature_layer,
            args.encoder_feature_layer_num_head,
            args.guideline_feature_layer_num_head,
            args.num_feature_layer,
            args.model_version, args.mask_out_templates, args.program_pooler
        ).to(device)
    print(f"model type: {type(model)}")
    optimizer = {
        "AdamW": AdamW,
        "Adam": Adam,
        "SGD": SGD
    }[args.optimizer](model.parameters(), args.learning_rate,
                      weight_decay=args.weight_decay)
    warmup_steps = 0.1 * args.num_train_epochs * len(train_dataloader)
    scheduler = LambdaLR(
        optimizer, lr_lambda={
            "": lambda step: 1,
            "warmup": lambda step: min(
                (step+1) ** (-0.5), (step+1) * warmup_steps ** (-1.5)
            ) * warmup_steps ** 0.5
        }[args.scheduler]
    )
    last_model = args.output_dir+"/last_model.pt"
    start_epoch = 0
    if os.path.exists(last_model):
        print("loading from checkpoint:", last_model)
        ckpt = torch.load(last_model)
        model.load_state_dict(ckpt[0])
        if not args.do_not_load_optimizer:
            optimizer.load_state_dict(ckpt[1])
            scheduler.load_state_dict(ckpt[2])
        start_epoch = ckpt[3] + 1

    best_dev_acc = -math.inf
    best_epoch = -1
    best_dev_loss = math.inf
    best_test_seen_acc = -math.inf
    best_test_unseen_acc = -math.inf
    best_test_acc_H = -math.inf
    for i_epoch in tqdm(range(start_epoch, args.num_train_epochs)):
        print("="*64)
        one_epoch(train_dataloader, model, optimizer, scheduler, tokenizer,
                  device, "train", i_epoch, False,
                  definitions_tokenized, definition_indices)
        if args.do_eval:
            dev_loss, dev_acc = one_epoch(
                val_dataloader, model, optimizer, scheduler, tokenizer, device,
                "dev", i_epoch, args.individual_scores,
                definitions_tokenized, definition_indices)
        if args.do_test:
            _, test_seen_acc = one_epoch(
                test_seen_dataloader, model, optimizer, scheduler, tokenizer,
                device, "test_seen", i_epoch, args.individual_scores,
                definitions_tokenized, definition_indices)
            _, test_unseen_acc = one_epoch(
                test_unseen_dataloader, model, optimizer, scheduler, tokenizer,
                device, "test_unseen", i_epoch, args.individual_scores,
                definitions_tokenized, definition_indices)
            test_acc_H = 2 * test_unseen_acc * test_seen_acc / (
                test_unseen_acc + test_seen_acc)
            print("test_acc_H: ", test_acc_H)

        ckpt_file = f"{args.output_dir}/epoch{i_epoch}_model.pt"
        save_model(model, optimizer, scheduler, i_epoch, ckpt_file)
        save_model(model, optimizer, scheduler, i_epoch, last_model)
        if args.do_eval and dev_acc > best_dev_acc:
            best_model_file = f"{args.output_dir}/best_model.pt"
            print("best dev accuracy")
            save_model(
                model, optimizer, scheduler, i_epoch, best_model_file)
            best_dev_acc = dev_acc
            best_epoch = i_epoch
            best_dev_loss = dev_loss
            if args.do_test:
                best_test_seen_acc = test_seen_acc
                best_test_unseen_acc = test_unseen_acc
                best_test_acc_H = test_acc_H

    if args.do_eval and args.do_test and start_epoch != args.num_train_epochs:
        print("final results:")
        print(dict(zip(
            ["best_epoch", "best_dev_acc", "best_dev_loss",
             "best_test_seen_acc", "best_test_unseen_acc", "best_test_acc_H"],
            [best_epoch, best_dev_acc, best_dev_loss,
             best_test_seen_acc, best_test_unseen_acc, best_test_acc_H])))
    else:
        one_epoch(
            test_seen_dataloader, model, optimizer, scheduler, tokenizer,
            device, "test_seen", start_epoch, args.individual_scores,
            definitions_tokenized, definition_indices)
        one_epoch(
            test_unseen_dataloader, model, optimizer, scheduler, tokenizer,
            device, "test_unseen", start_epoch, args.individual_scores,
            definitions_tokenized, definition_indices)


if __name__ == "__main__":
    args = parse_args()
    main(args)
