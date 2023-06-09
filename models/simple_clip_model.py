import torch
import torch.nn as nn
from contextlib import nullcontext


class SimpleCLIPModel(nn.Module):
    def __init__(self, model, phase):
        super().__init__()
        self.model = model
        self.phase = phase
        self.vision_feature_dim = model.visual.proj.shape[1]
        self.visual_adapter = nn.Linear(
            self.vision_feature_dim, self.vision_feature_dim)

    def forward(self, images, targets,
                definitions_tokenized, definition_indices, is_train):
        with torch.no_grad() if self.phase in ["fix_bert", "fix_both"] \
                else nullcontext():
            image_features = self.model.encode_image(images)
        image_features = self.visual_adapter(image_features)
        batch_size = images.shape[0]
        device = images.device

        if is_train:
            target_set = sorted(set(targets.cpu().numpy()))
            related_definitions = (
                definition_indices[:, None] == targets[None]
            ).sum(1) > 0
        else:
            target_set = range(200)
            related_definitions = torch.ones_like(
                definition_indices, dtype=bool)

        with torch.no_grad() if self.phase in ["fix_vision", "fix_both"] \
                else nullcontext():
            text_features = self.model.encode_text(
                definitions_tokenized[related_definitions])
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)
        scores = 10.0 * image_features @ text_features.T

        class_scores = torch.zeros(batch_size, 200).to(device) - 100
        for target_i in target_set:
            class_scores[:, target_i] = scores[
                :, definition_indices[related_definitions] == target_i
            ].mean(1)

        return class_scores
