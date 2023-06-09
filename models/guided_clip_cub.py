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


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class GuidedCLIPCUB(nn.Module):
    def __init__(self, model, phase, similarity,
                 encoder_feature_layer, guideline_feature_layer,
                 encoder_feature_layer_num_head,
                 guideline_feature_layer_num_head,
                 num_feature_layer,
                 model_version, mask_out_templates, program_pooler
                 ):
        super(GuidedCLIPCUB, self).__init__()
        self.model = model
        # self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = model.text_projection.shape[1]
        self.embed_size = model.token_embedding.embedding_dim
        self.encoder_feature_layer_type = encoder_feature_layer
        self.guideline_feature_layer_type = guideline_feature_layer
        self.vision_feature_dim = model.visual.proj.shape[1]

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
        self.visual_adapter = nn.Linear(
            self.vision_feature_dim, self.hidden_size)
        self.uniform_offset = nn.Parameter(torch.zeros(1)[0]-2)
        self.phase = phase
        self.similarity = similarity
        self.program_pooler = program_pooler

    def vision_features(self, images):
        visual = self.model.visual
        x = images.type(self.model.dtype)
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([visual.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1],
                                   dtype=x.dtype, device=x.device),
                       x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x)
        x = x @ visual.proj
        return x

    def text_features(self, text):
        x = self.model.token_embedding(text).type(self.model.dtype)

        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)
        cls_feature = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ \
            self.model.text_projection
        return x, cls_feature

    def forward(self, images, targets,
                definitions_tokenized, definition_indices, is_train):
        device = images.device
        batch_size = images.shape[0]

        if is_train:
            target_set = sorted(set(targets.cpu().numpy()))
            related_definitions = (
                definition_indices[:, None] == targets[None]
            ).sum(1) > 0
        else:
            target_set = range(200)
            related_definitions = torch.ones_like(
                definition_indices, dtype=bool)
        with torch.no_grad() if self.phase in ["fix_bert", "fix_both"] \
                else nullcontext():
            guideline_features, guideline_cls = self.text_features(
                definitions_tokenized[related_definitions]
            )
        if self.guideline_feature_layer_type == "transformer-encoder":
            guideline_features = self.guideline_feature_layer(
                guideline_features,
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
        with torch.no_grad() if self.phase in ["fix_vision", "fix_both"] \
                else nullcontext():
            image_features = self.vision_features(images)
        encoder_outputs = self.visual_adapter(image_features)
        if self.encoder_feature_layer_type == "transformer-encoder":
            encoder_outputs = self.encoder_feature_layer(encoder_outputs)
        elif self.encoder_feature_layer_type == "linear":
            encoder_outputs = self.encoder_feature_layer(
                encoder_outputs)
        elif self.encoder_feature_layer_type == "None":
            pass
        else:
            raise NotImplementedError()

        if not self.similarity:
            # getting program representations
            program, attentions = self.program_generator.decode(
                guideline_cls[None].to(encoder_outputs),
                guideline_features.transpose(1, 0).to(encoder_outputs),
                self.program_executor.program_length,
            )
            scores, reason_attentions, program_all = self.program_executor(
                encoder_outputs, program)
        else:
            # scores = F.cosine_similarity(encoder_outputs[:, 0, None],
            #                              guideline_cls[None], dim=-1) * 5 - 2
            # encoder_outputs = encoder_outputs / encoder_outputs.norm(
            #     dim=-1, keepdim=True)
            # guideline_cls = guideline_cls / guideline_cls.norm(
            #     dim=-1, keepdim=True)
            scores = torch.matmul(
                encoder_outputs[:, 0],
                guideline_cls.transpose(0, 1)
            )  # * 10

        class_scores = torch.zeros(batch_size, 200).to(device) - 100
        for target_i in target_set:
            if self.program_pooler == "sum":
                class_scores[:, target_i] = scores[
                    :, definition_indices[related_definitions] == target_i
                ].sum(1) + self.uniform_offset
            elif self.program_pooler == "max":
                class_scores[:, target_i] = scores[
                    :, definition_indices[related_definitions] == target_i
                ].max(1)[0]
            else:
                raise NotImplementedError()

        # from IPython import embed; embed(); exit()
        return class_scores
