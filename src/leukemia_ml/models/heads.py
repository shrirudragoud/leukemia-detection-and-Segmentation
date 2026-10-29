"""Pooling / classification heads over a bag of cell embeddings.

Bags are padded to (B, K, D) with a boolean mask (B, K), True = real cell.

* `GatedAttentionMIL` (Ilse, Tomczak, Welling 2018): a_k = softmax_k( w^T [tanh(V h_k) * sigm(U h_k)] ),
  bag = sum_k a_k h_k. The softmax normalises over the *valid* cells only, so the bag embedding
  does not depend on how many cells there are (cell count is a class-correlated density
  shortcut in this data).
* `MeanPoolHead`: mean of valid cells, the permutation-invariant baseline.
* `CellHead`: classify each cell independently (cell-level training / evaluation).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~mask, float("-inf"))
    out = torch.softmax(logits, dim=1)
    return torch.nan_to_num(out, nan=0.0)          # an all-masked bag gives zeros, not NaN


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, attn_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.V = nn.Linear(hidden, attn_dim)
        self.U = nn.Linear(hidden, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.cls = nn.Linear(hidden, n_classes)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        z = self.proj(h)                                           # (B,K,H)
        a = self.w(torch.tanh(self.V(z)) * torch.sigmoid(self.U(z))).squeeze(-1)
        attn = masked_softmax(a, mask)                             # (B,K)
        bag = (attn.unsqueeze(-1) * z).sum(dim=1)
        return self.cls(bag), attn


class MeanPoolHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, n_classes)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        z = self.proj(h)
        w = mask.float()
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        attn = w / denom
        return self.cls((attn.unsqueeze(-1) * z).sum(dim=1)), attn


class CellHead(nn.Module):
    """Per-cell logits; the bag logit is the mean of the cell logits (so one code path)."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        logits = self.net(h)                                        # (B,K,C)
        w = mask.float()
        attn = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (attn.unsqueeze(-1) * logits).sum(dim=1), attn


def build_head(kind: str, in_dim: int, n_classes: int, hidden: int, dropout: float) -> nn.Module:
    if kind == "mil":
        return GatedAttentionMIL(in_dim, n_classes, hidden, dropout=dropout)
    if kind == "mean":
        return MeanPoolHead(in_dim, n_classes, hidden, dropout)
    if kind == "cell":
        return CellHead(in_dim, n_classes, hidden, dropout)
    raise ValueError(f"unknown head {kind!r}")
