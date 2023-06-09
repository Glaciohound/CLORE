from dataclasses import dataclass
from typing import Optional, Tuple

from contextlib import nullcontext
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.file_utils import ModelOutput
from models import guided_program


class GuidedBertClues(nn.Module):
    def __init__(self, encoder, phase,
                 encoder_feature_layer, guideline_feature_layer,
                 encoder_feature_layer_num_head,
                 guideline_feature_layer_num_head,
                 num_feature_layer,
                 model_version, mask_out_templates
                 ):
        super(GuidedBertClues, self).__init__()
        supported_models = ['bert', 'roberta', 'deberta']
        assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.embed_size = encoder.embeddings.word_embeddings.embedding_dim
        self.encoder_feature_layer_type = encoder_feature_layer
        self.guideline_feature_layer_type = guideline_feature_layer

        # program-related parameters
        self.model_version = model_version
        model_library = guided_program
        self.program_generator = model_library.AttnDecoderRNN(
            self.embed_size, self.hidden_size
        )

        def build_feature_layer(feature_layer, feature_layer_num_head,
                                num_layers):
            if feature_layer == "transformer-encoder":
                output = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                    self.hidden_size, feature_layer_num_head,
                    batch_first=True), num_layers)
            elif feature_layer == "linear":
                output = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size*2),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_size*2, self.hidden_size),
                )
            elif feature_layer == "None":
                return None
            return output

        self.encoder_feature_layer = build_feature_layer(
            encoder_feature_layer, encoder_feature_layer_num_head,
            num_feature_layer,
        )
        self.guideline_feature_layer = build_feature_layer(
            guideline_feature_layer, guideline_feature_layer_num_head,
            num_feature_layer,
        )
        self.program_executor = model_library.ProgramExecutor(
            self.hidden_size, mask_out_templates)
        self.guideline_encoder = deepcopy(encoder)
        self.uniform_offset = nn.Parameter(torch.zeros(1)[0]-2)
        self.phase = phase

    def forward(self, input_ids, attention_mask, token_type_ids,
                guideline_input_ids, guideline_attention_mask,
                guideline_token_type_ids, target_text_matrix,
                entail_indicators):

        # BERT-BASE encoder: 768 hidden units
        device = input_ids.device

        with torch.no_grad():
            guideline_outputs = self.guideline_encoder(
                input_ids=guideline_input_ids,
                attention_mask=guideline_attention_mask,
                token_type_ids=guideline_token_type_ids,
            )
            guideline_cls = guideline_outputs.pooler_output
            guideline_features = guideline_outputs.last_hidden_state
        if self.guideline_feature_layer_type == "transformer-encoder":
            guideline_features = self.guideline_feature_layer(
                guideline_features,
                src_key_padding_mask=guideline_attention_mask.bool().logical_not()
            )
            guideline_cls = guideline_features[:, 0]
        elif self.guideline_feature_layer_type == "linear":
            guideline_features = self.guideline_feature_layer(
                guideline_features)
            guideline_cls = guideline_features[:, 0]
        elif self.guideline_feature_layer_type == "None":
            pass
        else:
            raise NotImplementedError()

        # Encode segments with BERT --> (256, 128, 768)
        with torch.no_grad() if self.phase == "program-only" else nullcontext():
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)[0]
        if self.encoder_feature_layer_type == "transformer-encoder":
            encoder_outputs = self.encoder_feature_layer(
                encoder_outputs,
                src_key_padding_mask=attention_mask.bool().logical_not()
            )
        elif self.encoder_feature_layer_type == "linear":
            encoder_outputs = self.encoder_feature_layer(
                encoder_outputs)
        elif self.encoder_feature_layer_type == "None":
            pass
        else:
            raise NotImplementedError()

        if self.phase in ["program", "program-only"]:
            # getting program representations
            program, attentions = self.program_generator.decode(
                guideline_cls[None].to(encoder_outputs),
                guideline_features.transpose(1, 0).to(encoder_outputs),
                self.program_executor.program_length,
            )
            scores, reason_attentions, program_all = self.program_executor(
                encoder_outputs, program)
        elif self.phase == "similarity":
            scores = F.cosine_similarity(encoder_outputs[:, 0, None],
                                         guideline_cls[None], dim=-1) * 5 - 2
        elif self.phase == "attributes-plain":
            token_scores = F.cosine_similarity(
                encoder_outputs[:, :, None],
                guideline_cls[None, None], dim=-1
            ) * 5 - 2
            scores = token_scores.mean(1)
        else:
            raise NotImplementedError()

        scores = scores * (entail_indicators * 2 - 1).to(device)
        labels_scores = []
        for target_select in target_text_matrix:
            labels_scores.append(scores[:, target_select].sum(-1) +
                                 self.uniform_offset)
        labels_scores = torch.stack(labels_scores, -1)

        # from transformers import AutoTokenizer, AutoModel
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # sent = tokenizer.decode(guideline_input_ids[0])
        # if "acceptable" in sent:
        #     from IPython import embed; embed();
        return labels_scores
