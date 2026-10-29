"""Temperature scaling (Guo et al. 2017): fit one scalar T on VALIDATION logits."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray, iters: int = 200) -> float:
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
