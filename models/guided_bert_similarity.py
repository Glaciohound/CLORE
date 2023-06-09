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
from models.functional import matrix_mul, element_wise_mul


@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class GuidedBertSimilarity(nn.Module):
    def __init__(self, encoder, guidelines=None, phase=None):
        super(GuidedBertSimilarity, self).__init__()
        supported_models = ['bert', 'roberta', 'deberta']
        self.phase = phase
        assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.embed_size = encoder.embeddings.word_embeddings.embedding_dim

        # program-related parameters
        self.guidelines = guidelines
        self.guideline_encoder = deepcopy(encoder)
        if self.guidelines is not None:
            with torch.no_grad():
                self.guideline_cls, _ = self.get_guideline_embeddings()
        # torch.randn(self.guidelines.n_guideline, self.embed_size) * 0.02
        # self.guideline_cls_bias = nn.Parameter(
        #     torch.randn(self.guidelines.n_guideline) * 0.02
        # )
        self.guideline_cls_layer = nn.Sequential(
            nn.Linear(self.embed_size, 2*self.embed_size),
            nn.LeakyReLU(),
            nn.Linear(2*self.embed_size, self.embed_size + 1),
        )

    def get_guideline_embeddings(self, device="cpu"):
        guideline_inputs = {
            k: torch.clone(v).to(device)
            for k, v in self.guidelines.inputs.items()
        }
        query_results = self.guideline_encoder(**guideline_inputs)
        return query_results.pooler_output, query_results.last_hidden_state

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units

        paired_inputs = self.guidelines is None
        device = input_ids.device

        if paired_inputs:
            batch_size = input_ids.shape[0] // 2
            input_ids, guideline_input_ids = input_ids.split(batch_size, 0)
            attention_mask, guideline_attention_mask = \
                attention_mask.split(batch_size, 0)
            token_type_ids, guideline_token_type_ids = \
                token_type_ids.split(batch_size, 0)

            with torch.no_grad():
                guideline_outputs = self.guideline_encoder(
                    input_ids=guideline_input_ids,
                    attention_mask=guideline_attention_mask,
                    token_type_ids=guideline_token_type_ids,
                )
                guideline_cls = guideline_outputs.pooler_output
        else:
            guideline_cls = self.guideline_cls

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None

        # device = input_ids.device
        # guideline_cls, _ = self.get_guideline_embeddings(device)
        guideline_cls = self.guideline_cls_layer(
            guideline_cls.to(device))
        bias = guideline_cls[:, -1]
        guideline_cls = guideline_cls[:, :-1]

        with torch.no_grad() if self.phase == "fix_encoder" else nullcontext():
            # Encode segments with BERT --> (256, 128, 768)
            encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                           attention_mask=attention_mask_reshape,
                                           token_type_ids=token_type_ids_reshape)[0]
            encoder_outputs = encoder_outputs[:, 0]

        scores = torch.matmul(encoder_outputs, guideline_cls.transpose(1, 0)) \
            + bias

        if paired_inputs:
            eye = torch.eye(batch_size, device=device, dtype=bool)
            permuted = torch.cat(
                [scores[eye][:, None],
                 scores[torch.logical_not(eye)].reshape(batch_size, -1)], 1
            )
            scores = permuted
            return SimpleOutput(last_hidden_state=scores, hidden_states=scores)

        labels_scores = []
        for target_select in self.guidelines.target_text_matrix:
            # if self.max_guidelines:
            # labels_scores.append(scores[:, target_select].max(-1)[0])
            # else:
            labels_scores.append(scores[:, target_select].mean(-1))
        labels_scores = torch.stack(labels_scores, -1)
        outputs = labels_scores

        return SimpleOutput(last_hidden_state=outputs, hidden_states=outputs)
