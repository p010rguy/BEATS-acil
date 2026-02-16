from __future__ import annotations

from pathlib import Path

import torch
from beats.BEATs import BEATs, BEATsConfig


class BEATsWithHead(torch.nn.Module):
    def __init__(self, beats: BEATs, num_classes: int) -> None:
        super().__init__()
        self.beats = beats
        self.classifier = torch.nn.Linear(beats.cfg.encoder_embed_dim, num_classes, bias=False)

    def forward(self, audio: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        pooled = extract_pooled_features(self.beats, audio, padding_mask)  # [B, 768]
        return self.classifier(pooled)  # [B, num_classes]


def load_beats_model(checkpoint_path: Path, device: torch.device) -> BEATs:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_beats_backbone(checkpoint_path: Path, device: torch.device) -> BEATs:
    return load_beats_model(checkpoint_path, device)


def extract_pooled_features(
    beats: BEATs,
    audio: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    features, features_padding = beats.extract_features(
        audio,
        padding_mask=padding_mask,
    )
    if features_padding is not None:
        valid = ~features_padding
        valid = valid.unsqueeze(-1)
        features = features * valid
        lengths = valid.sum(dim=1).clamp_min(1)
        pooled = features.sum(dim=1) / lengths
    else:
        pooled = features.mean(dim=1)
    return pooled


def expand_classifier(model: BEATsWithHead, new_num_classes: int) -> None:
    old_weight = model.classifier.weight.data
    old_num_classes, feat_dim = old_weight.shape
    if new_num_classes <= old_num_classes:
        return
    new_layer = torch.nn.Linear(feat_dim, new_num_classes, bias=False).to(
        old_weight.device
    )
    with torch.no_grad():
        new_layer.weight.zero_()
        new_layer.weight[:old_num_classes].copy_(old_weight)
    model.classifier = new_layer


def maybe_resume_checkpoint(
    model: BEATsWithHead,
    resume_path: str,
    device: torch.device,
) -> None:
    if not resume_path:
        return
    ckpt_path = Path(resume_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    num_classes = checkpoint.get("num_classes", model.classifier.out_features)
    expand_classifier(model, num_classes)
    model.load_state_dict(checkpoint["model"], strict=False)
    print(f"Resumed from checkpoint: {ckpt_path}")
