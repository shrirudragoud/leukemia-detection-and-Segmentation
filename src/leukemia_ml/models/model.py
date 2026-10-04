"""Bag classifier: cell encoder (+ optional feature branch) -> pooling head."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .encoders import Encoder, build_encoder, set_trainable
from .heads import build_head
from .hybrid import FeatureBranch
from .lora import apply_lora, lora_parameters


def pad_bags(h: torch.Tensor, bag: torch.Tensor, pos: torch.Tensor, n_bags: int,
             k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter flat cell rows (M,D) into (B,K,D) + boolean mask (B,K)."""
    out = h.new_zeros(n_bags, k_max, h.shape[-1])
    mask = torch.zeros(n_bags, k_max, dtype=torch.bool, device=h.device)
    out[bag, pos] = h
    mask[bag, pos] = True
    return out, mask


class LeukemiaModel(nn.Module):
    def __init__(self, encoder: Encoder | None, emb_dim: int, n_classes: int, cfg: ModelConfig,
                 n_features: int = 0, group_slices: list[slice] | None = None):
        super().__init__()
        self.encoder, self.cfg, self.n_classes = encoder, cfg, n_classes
        self.branch = (FeatureBranch(n_features, cfg.feature_hidden, group_slices,
                                     cfg.feature_group_dropout)
                       if cfg.use_features and n_features > 0 else None)
        in_dim = emb_dim + (self.branch.out_dim if self.branch is not None else 0)
        self.head = build_head(cfg.head, in_dim, n_classes, cfg.hidden, cfg.dropout)

    # --------------------------------------------------------------- encoder side
    def embed_cells(self, crops: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """crops (M,3,H,W) in [0,1] -> (M,D). Runs without grad when the encoder is frozen."""
        if self.encoder is None:
            return crops                                          # already embeddings
        frozen = not any(p.requires_grad for p in self.encoder.parameters())
        outs = []
        for i in range(0, len(crops), chunk):
            with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
                outs.append(self.encoder(crops[i:i + chunk]))
        return torch.cat(outs)

    # --------------------------------------------------------------- pooling side
    def forward(self, crops: torch.Tensor, feats: torch.Tensor, bag: torch.Tensor,
                pos: torch.Tensor, n: torch.Tensor, zero_groups: tuple[int, ...] = ()):
        h = self.embed_cells(crops)
        if self.branch is not None:
            h = torch.cat([h, self.branch(feats.to(h.dtype), zero_groups)], dim=-1)
        padded, mask = pad_bags(h, bag, pos, len(n), int(n.max()))
        logits, attn = self.head(padded, mask)
        return logits, attn

    def trainable_groups(self, tc) -> list[dict]:
        """Optimizer parameter groups with separate learning rates."""
        groups = [{"params": [p for p in self.head.parameters() if p.requires_grad],
                   "lr": tc.lr_head, "name": "head"}]
        if self.branch is not None:
            groups.append({"params": list(self.branch.parameters()), "lr": tc.lr_head,
                           "name": "branch"})
        if self.encoder is not None:
            lora = set(map(id, lora_parameters(self.encoder)))
            enc = [p for p in self.encoder.parameters() if p.requires_grad and id(p) not in lora]
            if enc:
                groups.append({"params": enc, "lr": tc.lr_encoder, "name": "encoder"})
            if lora:
                groups.append({"params": [p for p in self.encoder.parameters() if id(p) in lora],
                               "lr": tc.lr_lora, "name": "lora"})
        return [g for g in groups if g["params"]]


def build_model(cfg: ModelConfig, n_classes: int, n_features: int = 0,
                group_slices: list[slice] | None = None) -> LeukemiaModel:
    enc = build_encoder(cfg.encoder, cfg.input_size, cfg.pretrained, cfg.weights_path)
    set_trainable(enc, "full" if cfg.freeze == "full" else
                  "last_k" if cfg.freeze == "last_k" else "frozen", cfg.last_k)
    if cfg.freeze == "lora":
        apply_lora(enc.backbone, cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
        for n_, p in enc.named_parameters():
            p.requires_grad = n_.endswith((".A", ".B"))
    return LeukemiaModel(enc, enc.out_dim, n_classes, cfg, n_features, group_slices)
