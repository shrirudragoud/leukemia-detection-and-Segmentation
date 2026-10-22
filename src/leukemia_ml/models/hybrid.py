"""Interpretable feature branch (curvature / morphology / texture) fused with the embedding.

The per-cell vector f (already median-imputed and standardised on TRAIN cells) goes through a
small MLP; during training whole *feature groups* are randomly dropped (`group_dropout`) so the
network cannot become dependent on any single descriptor family and so the contribution of each
family can be measured at test time by zeroing it (a built-in ablation / permutation importance).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureBranch(nn.Module):
    def __init__(self, n_features: int, hidden: int, group_slices: list[slice] | None = None,
                 group_dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.group_slices = group_slices or [slice(0, n_features)]
        self.group_dropout = group_dropout
        self.out_dim = hidden
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())

    def forward(self, f: torch.Tensor, zero_groups: tuple[int, ...] = ()) -> torch.Tensor:
        if self.training and self.group_dropout > 0:
            f = f.clone()
            for sl in self.group_slices:
                drop = torch.rand(f.shape[0], 1, device=f.device) < self.group_dropout
                f[:, sl] = f[:, sl].masked_fill(drop, 0.0)
        if zero_groups:
            f = f.clone()
            for g in zero_groups:
                f[:, self.group_slices[g]] = 0.0
        return self.net(f)
