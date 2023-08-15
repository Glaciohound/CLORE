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
from models import guided_program_cub as guided_program


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class GuidedBertCUB(nn.Module):
    def __init__(self, resnet, encoder, vision_num_patches, phase, similarity,
                 encoder_feature_layer, guideline_feature_layer,
                 encoder_feature_layer_num_head,
                 guideline_feature_layer_num_head,
                 num_feature_layer,
                 model_version, mask_out_templates
                 ):
        super(GuidedBertCUB, self).__init__()
        self.resnet = resnet
        # self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.embed_size = encoder.embeddings.word_embeddings.embedding_dim
        self.encoder_feature_layer_type = encoder_feature_layer
        self.guideline_feature_layer_type = guideline_feature_layer
        self.vision_num_patches = vision_num_patches
        self.vision_feature_dim = self.resnet.layer4[2].conv3.out_channels
        self.vision_pos_embeddings = nn.Embedding(
            vision_num_patches, self.hidden_size,
            _weight=sinusoidal_init(
                vision_num_patches, self.hidden_size
            ))

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
        self.visual_adapter = nn.Linear(
            self.vision_feature_dim, self.hidden_size)
        self.uniform_offset = nn.Parameter(torch.zeros(1)[0]-2)
        self.phase = phase
        self.similarity = similarity

    def resnet_features(self, images):
        x = images
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

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
        with torch.no_grad() if self.phase in ["fix_text", "fix_both"] \
                else nullcontext():
            input_ids = definitions_tokenized["input_ids"][
                related_definitions]
            attention_mask = definitions_tokenized["attention_mask"][
                related_definitions]
            token_type_ids = definitions_tokenized["token_type_ids"][
                related_definitions]
            guideline_outputs = self.guideline_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            guideline_cls = guideline_outputs.pooler_output
            guideline_features = guideline_outputs.last_hidden_state
        if self.guideline_feature_layer_type == "transformer-encoder":
            guideline_features = self.guideline_feature_layer(
                guideline_features,
                src_key_padding_mask=attention_mask.bool().logical_not()
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
            resnet_features = self.resnet_features(images)
        encoder_outputs = self.visual_adapter(resnet_features.reshape(
            *resnet_features.shape[:2], self.vision_num_patches
        ).transpose(1, 2)) + self.vision_pos_embeddings.weight
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
            scores = F.cosine_similarity(encoder_outputs[:, 0, None],
                                         guideline_cls[None], dim=-1) * 5 - 2

        class_scores = torch.zeros(batch_size, 200).to(device) - 100
        for target_i in target_set:
            class_scores[:, target_i] = scores[
                :, definition_indices[related_definitions] == target_i
            ].sum(1) + self.uniform_offset

        return class_scores
