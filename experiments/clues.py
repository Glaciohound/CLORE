import csv
import math
import json
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, Sampler, default_collate
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from typing import List
from copy import deepcopy
from contextlib import nullcontext
from collections import defaultdict

from guidelines.guidelines import Guidelines
from models.guided_bert_clues import GuidedBertClues
from experiments.utils import RunningMean, Mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int,
                        default=1)

    parser.add_argument("--guided_model", type=str, default="main",
                        choices=["main"])
    parser.add_argument("--phase", type=str, default="program",
                        choices=["program", "program-only", "similarity",
                                 "attributes-plain"])
    parser.add_argument("--subtask", type=str, default="clues_real")
    # parser.add_argument("--program_length", type=int, default=10)
    parser.add_argument("--single_task", type=str, default=None)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--individual_scores", action="store_true")
    parser.add_argument("--normalize_datasets", action="store_true")
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
    parser.add_argument("--model_version", type=str, default="2")
    parser.add_argument("--batch_per_epoch", type=int, default=2500)
    parser.add_argument("--remove_label_text", action="store_true")
    parser.add_argument("--num_feature_layer", type=int, default=1)
    parser.add_argument("--data_variance", type=str, default=None)
    parser.add_argument("--mask_out_templates", type=int,
                        nargs="*", default=[])

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)
    return args


def get_data(data_dir, subtask, single_task, zero_shot, remove_label_text):
    def load_jsonl(filename):
        with open(filename, "r") as f:
            return [json.loads(line) for line in f.readlines()]

    def load_json(filename):
        with open(filename, "r") as f:
            return json.load(f)

    def load_csv(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            return list(reader)

    def revert_explanations(explanations, entails, target_key):
        return {
            _key: [
                (_expl, entails is None or entails[_expl] == "entail")
                for _expl, _key2 in explanations.items()
                if _key2 == _key
            ]
            for _key in target_key
        }

    def remove_label_text_fn(explanations):
        new_explanations = {}
        for _key, sentences in explanations.items():
            new_explanations[_key] = []
            for _s, _l in sentences:
                _s = _s.replace(_key, "this")
                _s = _s.replace(_key[0].lower()+_key[1:], "this")
                _s = _s.replace(_key[0].upper()+_key[1:], "this")
                _s = _s.replace(_key.lower(), "this")
                _s = _s.replace(_key.upper(), "this")
                new_explanations[_key].append((_s, _l))
        return new_explanations

    def get_data_clues_real():
        this_data = {
            sub2task: {
                split: load_jsonl(
                    f"{data_dir}/clues_real/{sub1task}/"
                    f"{sub2task}/{split}.jsonl")
                for split in ("train", "dev", "test")
            }
            for sub1task in os.listdir(f"{data_dir}/clues_real")
            for sub2task in os.listdir(f"{data_dir}/clues_real/{sub1task}")
        }
        zero_shot_tasks = (
            "banknote-authentication", "tic-tac-toe-endgame", "car-evaluation",
            "contraceptive-method-choice", "indian-liver-patient",
            "travel-insurance", "entrepreneur-competency",
            "award-nomination-result", "coin-face-value", "coin-metal",
            "driving-championship-points", "election-outcome", "hotel-rating",
            "manifold-orientability", "soccer-club-region",
            "soccer-league-type",
        )
        wikipedia_tasks = (
            "award-nomination-result", "coin-metal", "soccer-league-type",
            "driving-championship-points", "hotel-rating", "coin-face-value",
            "election-outcome", "manifold-orientability", "soccer-club-region",
        )
        if zero_shot:
            for _task, _task_data in this_data.items():
                if _task in zero_shot_tasks:
                    if _task in wikipedia_tasks:
                        _task_data["test"] = (
                            _task_data["train"] + _task_data["dev"] +
                            _task_data["test"])
                    _task_data["train"] = []
                    _task_data["dev"] = []
                else:
                    _task_data["train"] = \
                        _task_data["train"] + _task_data["test"]
                    _task_data["test"] = []
        explanations = load_json(
            f"{data_dir}/clues_real_utils/real_lbl_indicator_map.json")
        entail = load_json(
            f"{data_dir}/clues_real_utils/real_exp_entail_indicator.json")
        task_details = load_json(
            f"{data_dir}/clues_real_utils/real_task_details.json")
        assert set(explanations.keys()) == set(this_data.keys())
        for sub2task, _expl in explanations.items():
            _entail = entail[sub2task]
            target_key = task_details[sub2task]["task_lbls"]
            this_data[sub2task]["target_key"] = target_key
            _expl = revert_explanations(
                _expl, _entail, target_key)
            if remove_label_text:
                _expl = remove_label_text_fn(_expl)
            this_data[sub2task]["explanations"] = _expl
        if single_task is not None:
            this_data = {single_task: this_data[single_task]}
        return this_data

    def get_data_clues_syn():
        this_data = {
            sub1task: {
                split: load_jsonl(
                    f"{data_dir}/clues_syn/{sub1task}/{split}.jsonl")
                for split in ("train", "valid", "test")
            }
            for sub1task in os.listdir(f"{data_dir}/clues_syn")
        }
        if zero_shot:
            this_data = {
                sub1task: {
                    "train": this_task_data["train"] + this_task_data["valid"]
                    + this_task_data["test"],
                    "valid": [],
                    "test": []
                } for sub1task, this_task_data in this_data.items()
            }
        targets = {
            "bird species": ["wug", "blicket", "dax", "toma", "pimwit", "zav",
                             "speff", "tulver", "gazzer", "fem", "fendle",
                             "tupa",
                             "not blicket", "not dax", "not fem", "not fendle",
                             "not pimwit", "not speff", "not toma", "not tupa",
                             "not wug", "not zav",
                             ],
            "animal species": ["wug", "blicket", "dax", "toma", "pimwit",
                               "zav", "speff", "tulver", "gazzer", "fem",
                               "fendle", "tupa",
                               "not blicket", "not dax", "not fendle",
                               "not speff", "not toma", "not tulver",
                               "not tupa", "not wug", "not zav"],
            "rain tomorrow": ["yes", "no", "not yes", "not no"],
            "final position": ["1", "2", "3", "4", "Not Qualified",
                               "not 1", "not 2", "not 3", "not 4",
                               "not Not Qualified"],
            "relevance score": ["1", "2", "3", "4", "5",
                                "not 1", "not 2", "not 4"]
        }
        for sub1task in os.listdir(f"{data_dir}/clues_syn"):
            explanations = [
                _expl[0] for _expl in load_csv(
                    f"{data_dir}/clues_syn/{sub1task}/explanations.csv"
                )
            ]
            target_type = list(set(targets.keys()).intersection(
                set(this_data[sub1task]["train"][0].keys())))
            assert len(target_type) == 1
            target_type = target_type[0]
            target_key = targets[target_type]
            this_data[sub1task]["target_key"] = target_key
            for split in ("train", "valid", "test"):
                for item in this_data[sub1task][split]:
                    item["lbl"] = item.pop(target_type)
                    item["target_key"] = target_key
            explanation_keys = []
            for _expl in explanations:
                this_target_key = [_key for _key in target_key if
                                   _expl.endswith(_key) or
                                   _expl.endswith(_key+".")]
                assert len(this_target_key) in [1, 2], f"{_expl}, {target_key}"
                if len(this_target_key) == 2:
                    assert this_target_key[1] == "not " + this_target_key[0]
                    this_target_key.pop(0)
                explanation_keys.append(this_target_key[0])
            explanations = {_expl: _key for _expl, _key in
                            zip(explanations, explanation_keys)}
            this_data[sub1task]["explanations"] = revert_explanations(
                explanations, None, target_key)
        return this_data

    def mix_data(list_of_data, valid_to_dev=True):
        final_data = {}
        id_iter = iter(range(1000000))
        for each_data in list_of_data:
            for sub1task, this_task_data in each_data.items():
                if valid_to_dev and "valid" in this_task_data:
                    this_task_data["dev"] = this_task_data.pop("valid")
                final_data[str(next(id_iter))+"_"+sub1task] = this_task_data
        return final_data

    if subtask == "clues_real":
        return get_data_clues_real()
    elif subtask == "clues_syn":
        return get_data_clues_syn()
    else:
        _clues_real = get_data_clues_real()
        _clues_syn = get_data_clues_syn()
        if subtask == "real:syn=1:1":
            return mix_data([_clues_real, _clues_syn])
        elif subtask == "real:syn=1:3":
            return mix_data([_clues_real, _clues_syn, _clues_syn, _clues_syn])
        elif subtask == "real:syn=1:5":
            return mix_data([_clues_real, _clues_syn, _clues_syn, _clues_syn,
                             _clues_syn, _clues_syn])
    raise NotImplementedError()


def datum_to_sentence(datum, data_variance):
    def piece_fn(_key, _value):
        if data_variance is None:
            return f"{_key} is {_value}"
        elif data_variance.startswith("replace:"):
            return f"{_key} {data_variance[8:]} {_value}"
        elif data_variance.startswith("append:") or \
                data_variance.startswith("prepend"):
            return f"{_key} is {_value}"

    fat = (
        "For this item, " +
        ", ".join([
            piece_fn(_key, _value)
            for _key, _value in datum.items()
            if _key not in ['lbl', 'target_key']
        ])
    )
    if data_variance is None or data_variance.startswith("replace"):
        fat = fat + "."
    elif data_variance.startswith("append"):
        fat = fat + data_variance.split(":")[-1]
    elif data_variance.startswith("prepend"):
        fat = data_variance.split(":")[-1] + fat
    return fat


class CLUES_Dataset(Dataset):
    def __init__(self, data, tokenizer, split, data_variance):
        self.data = data
        self.tokenizer = tokenizer
        self.split = split
        self.task_names = list(data.keys())
        self.task_names.sort()
        self.task_data_num = {
            _task: len(data[_task][split])
            for _task in self.task_names
        }
        self.datum_indices = [
            _task
            for _task in self.task_names
            for _j in data[_task][split]
        ]
        self.task_start = {_task: self.datum_indices.index(_task)
                           for _task in self.task_names if _task in
                           self.datum_indices}
        self.num_data = len(self.datum_indices)
        self.data_variance = data_variance

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        task_name = self.datum_indices[index]
        datum = self.data[task_name][self.split][
            index-self.task_start[task_name]]
        datum_sentence = datum_to_sentence(datum, self.data_variance)
        label = datum["lbl"]
        target_key = datum["target_key"]
        guidelines = self.data[task_name]["guidelines"]
        return datum_sentence, target_key.index(str(label)), task_name, \
            guidelines.inputs, guidelines.target_text_matrix, \
            guidelines.entail_indicators


class CLUES_BatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size, shuffle, normalize_datasets,
                 batch_per_epoch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize_datasets = normalize_datasets
        assert shuffle or not normalize_datasets
        self.all_batches = []
        task_num_batches = defaultdict(lambda: 0)
        task_names = []
        batch = []
        prev_task_name = None
        for idx, (_, _, task_name, _, _, _) in enumerate(dataset):
            if (prev_task_name is None or task_name == prev_task_name) \
                    and len(batch) < self.batch_size:
                batch.append(idx)
            else:
                self.all_batches.append(batch)
                task_num_batches[prev_task_name] += 1
                task_names.append(prev_task_name)
                batch = [idx]
            prev_task_name = task_name
        self.all_batches.append(batch)
        task_num_batches[task_name] += 1
        task_names.append(task_name)
        self.sampling_weights = [
            1/len(task_num_batches)/task_num_batches[_task_name]
            for _task_name in task_names
        ]
        self.batch_per_epoch = batch_per_epoch

    def __iter__(self):
        all_batches = deepcopy(self.all_batches)
        if self.shuffle:
            if not self.normalize_datasets:
                random.shuffle(all_batches)
            else:
                all_batches = np.random.choice(
                    np.array(all_batches, dtype=object),
                    p=self.sampling_weights,
                    size=self.batch_per_epoch
                )
        for batch in all_batches:
            yield batch

    def __len__(self):
        if self.shuffle and self.normalize_datasets:
            return self.batch_per_epoch
        else:
            return len(self.all_batches)


def CLUES_collate(batch):
    collated = default_collate(batch)
    collated = [
        collated[0],
        collated[1],
        collated[2][0],
        {_key: _value[0] for _key, _value in collated[3].items()},
        collated[4][0],
        torch.stack([x[0] for x in collated[5]]),
    ]
    # inputs, labels, task_name, guideline_inputs,
    # guideline_matrix, entail_indicators
    return collated


def postprocess_batch(batch, tokenizer, device):
    inputs = tokenizer(list(batch[0]), padding=True, max_length=256,
                       return_tensors="pt")
    batch[0] = {
        _key: _value.to(device)
        for _key, _value in inputs.items()
    }
    if "token_type_ids" not in batch[0]:
        batch[0]["token_type_ids"] = None
    batch[1] = batch[1].to(device)
    batch[3] = {
        "guideline_"+_key: _value.to(device)
        for _key, _value in batch[3].items()
    }
    if "guideline_token_type_ids" not in batch[3]:
        batch[3]["guideline_token_type_ids"] = None
    return batch


def one_epoch(dataloader, model, optimizer, scheduler, tokenizer,
              device, split, i_epoch, individual_scores):
    def mean_of_mean(means):
        values = [_mean.value for _mean in means]
        return sum(values) / len(values)

    print(f"{split} epoch {i_epoch}")
    pbar = tqdm(dataloader, leave=False)
    if split == "train":
        loss_mean = RunningMean()
        acc_mean = RunningMean()
    elif not individual_scores:
        loss_mean = Mean()
        acc_mean = Mean()
    else:
        each_loss_mean = defaultdict(lambda: Mean())
        each_acc_mean = defaultdict(lambda: Mean())

    for batch in pbar:
        optimizer.zero_grad()
        batch = postprocess_batch(batch, tokenizer, device)
        inputs, labels, task_name, guideline_inputs, target_text_matrix, \
            entail_indicators = batch
        batch_size = inputs["input_ids"].shape[0]
        with torch.no_grad() if split != "train" else nullcontext():
            scores = model(**inputs, **guideline_inputs,
                           target_text_matrix=target_text_matrix,
                           entail_indicators=entail_indicators)
        loss = F.cross_entropy(
            scores, labels, reduction="none")
        acc = (scores.argmax(1) == labels).float()
        if split == "train":
            (loss.sum() / batch_size).backward()
            optimizer.step()
            scheduler.step()
        if not individual_scores:
            loss_mean.update(loss.detach().tolist())
            acc_mean.update(acc.detach().tolist())
        else:
            each_loss_mean[task_name].update(loss.detach().tolist())
            each_acc_mean[task_name].update(acc.detach().tolist())
            loss_mean = mean_of_mean(each_loss_mean.values())
            acc_mean = mean_of_mean(each_acc_mean.values())
        pbar.set_description(f"loss: {loss_mean:.4f}, acc: {acc_mean:.4f}")
    print(f"{split} epoch-{i_epoch}:"
          f" average loss: {loss_mean}, average accuracy: {acc_mean}")
    if not individual_scores:
        loss_mean = loss_mean.value
        acc_mean = acc_mean.value
    else:
        print(f"individual loss: {dict(each_loss_mean)}")
        print(f"individual acc: {dict(each_acc_mean)}", flush=True)
    return loss_mean, acc_mean


def save_model(model, optimizer, scheduler, start_epoch, filename):
    print(f"saving model to {filename}")
    torch.save(
        [model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
         start_epoch],
        filename
    )


def main(args):
    data = get_data(args.data_dir, args.subtask, args.single_task,
                    args.zero_shot, args.remove_label_text)
    device = torch.device("cpu") if torch.cuda.device_count() == 0 else \
        torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    bert = AutoModel.from_pretrained(args.model_name_or_path)

    for task in data.values():
        task["guidelines"] = Guidelines(
            task["explanations"], tokenizer, task["target_key"], True)

    train_dataset = CLUES_Dataset(data, tokenizer, "train", args.data_variance)
    dev_dataset = CLUES_Dataset(
        data, tokenizer, "valid" if args.subtask == "clues_syn" else "dev",
        args.data_variance)
    test_dataset = CLUES_Dataset(data, tokenizer, "test", args.data_variance)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=CLUES_collate,
        batch_sampler=CLUES_BatchSampler(
            train_dataset, args.per_device_train_batch_size,
            True, args.normalize_datasets, args.batch_per_epoch))
    dev_dataloader = DataLoader(
        dev_dataset, collate_fn=CLUES_collate,
        batch_sampler=CLUES_BatchSampler(
            dev_dataset, args.per_device_eval_batch_size, False, False,
            args.batch_per_epoch
        ))
    test_dataloader = DataLoader(
        test_dataset, collate_fn=CLUES_collate,
        batch_sampler=CLUES_BatchSampler(
            test_dataset, args.per_device_eval_batch_size, False, False,
            args.batch_per_epoch
        ))

    model = GuidedBertClues(
        bert, args.phase,
        args.encoder_feature_layer, args.guideline_feature_layer,
        args.encoder_feature_layer_num_head,
        args.guideline_feature_layer_num_head,
        args.num_feature_layer,
        args.model_version, args.mask_out_templates
    ).to(device)
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
        optimizer.load_state_dict(ckpt[1])
        scheduler.load_state_dict(ckpt[2])
        start_epoch = ckpt[3] + 1

    best_dev_acc = -math.inf
    best_epoch = -1
    best_epoch_dev_loss = math.inf
    best_epoch_test_acc = -math.inf
    best_epoch_test_loss = math.inf
    # test_loss, test_acc = one_epoch(
    #     test_dataloader, model, optimizer, scheduler, tokenizer, device,
    #     "test", start_epoch-1, args.individual_scores)
    for i_epoch in tqdm(range(start_epoch, args.num_train_epochs)):
        print("="*64)
        one_epoch(train_dataloader, model, optimizer, scheduler, tokenizer,
                  device, "train", i_epoch, False)
        dev_loss, dev_acc = one_epoch(
            dev_dataloader, model, optimizer, scheduler, tokenizer, device,
            "dev", i_epoch, args.individual_scores)
        test_loss, test_acc = one_epoch(
            test_dataloader, model, optimizer, scheduler, tokenizer, device,
            "test", i_epoch, args.individual_scores)

        ckpt_file = f"{args.output_dir}/epoch{i_epoch}_model.pt"
        save_model(model, optimizer, scheduler, i_epoch, ckpt_file)
        save_model(model, optimizer, scheduler, i_epoch, last_model)
        if dev_acc > best_dev_acc:
            best_model_file = f"{args.output_dir}/best_model.pt"
            print("best dev accuracy")
            save_model(
                model, optimizer, scheduler, i_epoch, best_model_file)
            best_dev_acc = dev_acc
            best_epoch = i_epoch
            best_epoch_dev_loss = dev_loss
            best_epoch_test_acc = test_acc
            best_epoch_test_loss = test_loss

    if start_epoch != args.num_train_epochs:
        print("final results:")
        print(dict(zip(["best_epoch", "best_dev_acc", "best_epoch_dev_loss",
                        "best_epoch_test_acc", "best_epoch_test_loss"],
                       [best_epoch, best_dev_acc, best_epoch_dev_loss,
                        best_epoch_test_acc, best_epoch_test_loss])))
    else:
        test_loss, test_acc = one_epoch(
            test_dataloader, model, optimizer, scheduler, tokenizer, device,
            "test", start_epoch, args.individual_scores)


if __name__ == "__main__":
    args = parse_args()
    main(args)
