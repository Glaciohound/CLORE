import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        if self.args.zero_shot_id is None:
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.float().view(-1, self.model.config.num_labels))
        else:
            label_mask = torch.ones_like(labels[0], dtype=bool)
            label_mask[self.args.zero_shot_id] = False
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels)[:, label_mask],
                labels.float().view(-1, self.model.config.num_labels)[:, label_mask])
        return (loss, outputs) if return_outputs else loss


class MulticlassBinaryTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        labels_one_hot = F.one_hot(
            labels, self.model.config.num_labels).float()
        if hasattr(self.args, "zero_shot_labels"):
            train_labels = [
                class_i for class_i in range(self.model.config.num_labels)
                if class_i not in self.args.zero_shot_labels]
            logits = logits[:, train_labels]
            labels_one_hot = labels_one_hot[:, train_labels]
        loss = loss_fct(logits, labels_one_hot)
        return (loss, outputs) if return_outputs else loss


class PairedContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        inputs = {key: value.transpose(1, 0).reshape(-1, value.shape[-1])
                  for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        # labels = torch.zeros(batch_size, dtype=int, device=device)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class PairedBinaryTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        inputs = {key: value.transpose(1, 0).reshape(-1, value.shape[-1])
                  for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        # labels = torch.zeros(batch_size, dtype=int, device=device)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class PairedBinaryContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        inputs = {key: value.transpose(1, 0).reshape(-1, value.shape[-1])
                  for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = torch.diag(labels)
        loss_fct = nn.BCEWithLogitsLoss()
        # labels = torch.zeros(batch_size, dtype=int, device=device)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
