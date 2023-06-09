# -*- coding: utf-8 -*-
# file: bert_base.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BOW_SVM(nn.Module):
    def __init__(self, args, num_class, base="bert-base-uncased",
                 data="jigsaw2018", problem_type="multi_label_classification"):
        super(BOW_SVM, self).__init__()
        self.bert = AutoModel.from_pretrained(base)
        if args.fix_bert:
            self.bert.requires_grad_(False)
        self.problem_type = problem_type
        self.few_shot_ratio = args.few_shot_ratio

        self.tokenizer = AutoTokenizer.from_pretrained(base)

        embed_dim = self.bert.pooler.dense.in_features
        self.final = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_class)
        )

    def forward(self, **inputs):
        # labels = inputs.get("labels", None)
        if "labels" in inputs:
            del inputs["labels"]

        word_features = self.bert.embeddings.word_embeddings(
            inputs["input_ids"]
        )
        bow_features = []
        for _bow, _mask in zip(word_features, inputs["attention_mask"]):
            bow_features.append(
                _bow[_mask.bool()].mean(0))
        bow_features = torch.stack(bow_features, 0)
        target_scores = self.final(bow_features)

        return target_scores, target_scores
