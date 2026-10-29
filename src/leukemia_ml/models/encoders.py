"""Pretrained encoders behind one interface: crops (N,3,H,W) in [0,1] -> embeddings (N,D).

Specs
  timm:<name>      any timm model, e.g. 'timm:resnet50.a1_in1k', 'timm:efficientnet_b0.ra_in1k'
  dinov2_s/dinov2_b  generic DINOv2 (timm 'vit_{small,base}_patch14_dinov2.lvd142m')
  dinobloom_s/b      DinoBloom (hematology DINOv2; Zenodo 10908163, CC-BY-4.0) - same architecture,
                     weights loaded from a local `.pth`

ImageNet mean/std normalisation happens INSIDE the encoder so the data pipeline stays in [0,1].
ViT input size must be a multiple of 14; position embeddings are bicubically resampled when
the size differs from the pretraining resolution (DinoBloom: 224 -> 16x16 tokens + CLS).
"""
from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed

log = logging.getLogger(__name__)

_VIT = {
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "dinobloom_s": "vit_small_patch14_dinov2.lvd142m",
    "dinobloom_b": "vit_base_patch14_dinov2.lvd142m",
}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class Encoder(nn.Module):
    def __init__(self, backbone: nn.Module, kind: str, name: str):
        super().__init__()
        self.backbone, self.kind, self.name = backbone, kind, name
        self.out_dim = backbone.num_features
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)

    # transformer blocks (ViT) or None (CNN): used for last-k unfreezing and LoRA placement
    @property
    def blocks(self):
        return getattr(self.backbone, "blocks", None)


def load_dinobloom_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Backbone weights of a DinoBloom checkpoint ({'teacher': {'backbone.*', heads...}})."""
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = raw.get("teacher", raw.get("model", raw)) if isinstance(raw, dict) else raw
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue                                   # dino/ibot projection heads
        k = k[len("backbone."):]
        if k == "mask_token":
            continue                                   # training-only (iBOT masking)
        out[k] = v
    if not out:
        raise ValueError(f"no 'backbone.*' weights found in {path}")
    return out


def _resample_pos_embed(sd: dict[str, torch.Tensor], model: nn.Module) -> None:
    if "pos_embed" not in sd:
        return
    pe, target = sd["pos_embed"], model.pos_embed
    if pe.shape == target.shape:
        return
    grid = model.patch_embed.grid_size
    sd["pos_embed"] = resample_abs_pos_embed(pe, new_size=grid, num_prefix_tokens=1)


def build_encoder(spec: str, input_size: int = 112, pretrained: bool = True,
                  weights_path: str | None = None) -> Encoder:
    if spec.startswith("timm:"):
        name = spec[5:]
        backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        return Encoder(backbone, "cnn", name)
    if spec not in _VIT:
        raise ValueError(f"unknown encoder {spec!r}")
    if input_size % 14:
        raise ValueError(f"ViT/14 input size must be a multiple of 14, got {input_size}")
    arch = _VIT[spec]
    if spec.startswith("dinobloom"):
        backbone = timm.create_model(arch, pretrained=False, num_classes=0, img_size=input_size)
        if pretrained:
            if not weights_path:
                raise ValueError("DinoBloom needs weights_path (Zenodo 10908163 .pth)")
            sd = load_dinobloom_state(weights_path)
            _resample_pos_embed(sd, backbone)
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            bad = [k for k in missing if not k.startswith(("head", "fc_norm"))]
            if bad or unexpected:
                raise RuntimeError(f"DinoBloom checkpoint mismatch: missing={bad[:5]} "
                                   f"unexpected={list(unexpected)[:5]}")
    else:
        backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, img_size=input_size)
    return Encoder(backbone, "vit", spec)


def set_trainable(enc: Encoder, mode: str, last_k: int = 2) -> None:
    """Freeze/unfreeze the encoder. LoRA adapters are added separately (see lora.py)."""
    for p in enc.parameters():
        p.requires_grad = mode == "full"
    if mode == "last_k":
        blocks = enc.blocks
        if blocks is None:
            raise ValueError("last_k needs a transformer encoder")
        for blk in list(blocks)[-last_k:]:
            for p in blk.parameters():
                p.requires_grad = True
        norm = getattr(enc.backbone, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
