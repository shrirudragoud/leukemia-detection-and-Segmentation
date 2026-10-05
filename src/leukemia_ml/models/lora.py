"""Minimal LoRA (Hu et al.) for nn.Linear layers: W x + (alpha/r) * B(A(x)).

B is zero-initialised, so the adapted model equals the pretrained one at step 0. Only A and B
train; the base weights stay frozen. `merge`/`unmerge` fold the update into W for inference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank, self.scale = rank, alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        return out + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += (self.B @ self.A) * self.scale
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= (self.B @ self.A) * self.scale
            self.merged = False


def apply_lora(model: nn.Module, targets: tuple[str, ...], rank: int, alpha: float,
               dropout: float = 0.0) -> list[str]:
    """Wrap every nn.Linear whose qualified name ends with one of `targets`. Returns the names."""
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                wrapped.append(full)
    if not wrapped:
        raise ValueError(f"no nn.Linear matched LoRA targets {targets}")
    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.endswith((".A", ".B")) and p.requires_grad]
