"""Frozen-encoder embedding extraction with an on-disk cache.

The cache key covers everything that changes the numbers (encoder spec, input size, weights
file, crop cache), so ablations over preprocessing variants can reuse or invalidate it safely.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data.datasets import to_tensor
from .models.encoders import build_encoder

log = logging.getLogger(__name__)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def embedding_path(crops_dir: Path, mcfg: ModelConfig) -> Path:
    w = Path(mcfg.weights_path).name if mcfg.weights_path else "-"
    key = hashlib.sha256(json.dumps([mcfg.encoder, mcfg.input_size, mcfg.pretrained, w],
                                    sort_keys=True).encode()).hexdigest()[:10]
    return Path(crops_dir) / f"emb_{mcfg.encoder}_{mcfg.input_size}_{key}.npy"


@torch.no_grad()
def extract_embeddings(crops: np.ndarray, mcfg: ModelConfig, crops_dir: Path, device="auto",
                       batch_size: int = 128, limit: int | None = None, force: bool = False
                       ) -> tuple[np.ndarray, dict]:
    """(N,D) float16 embeddings for every crop; returns (embeddings, timing info)."""
    path = embedding_path(crops_dir, mcfg)
    n = len(crops) if limit is None else min(limit, len(crops))
    if path.exists() and not force and limit is None:
        emb = np.load(path, mmap_mode="r")
        if emb.shape[0] == n:
            return emb, {"cached": True}
    dev = resolve_device(device)
    enc = build_encoder(mcfg.encoder, mcfg.input_size, mcfg.pretrained, mcfg.weights_path)
    enc.eval().to(dev)
    out = np.zeros((n, enc.out_dim), np.float16)
    t0 = time.time()
    for i in range(0, n, batch_size):
        x = torch.stack([to_tensor(np.asarray(crops[j]), mcfg.input_size)
                         for j in range(i, min(i + batch_size, n))]).to(dev)
        out[i:i + len(x)] = enc(x).float().cpu().numpy().astype(np.float16)
        if (i // batch_size) % 20 == 0:
            log.info("embedded %d/%d (%.0f cells/s)", i + len(x), n, (i + len(x)) / (time.time() - t0))
    secs = time.time() - t0
    if limit is None:
        np.save(path, out)
    return out, {"cached": False, "seconds": secs, "cells_per_s": n / max(secs, 1e-9),
                 "device": str(dev), "dim": enc.out_dim}
