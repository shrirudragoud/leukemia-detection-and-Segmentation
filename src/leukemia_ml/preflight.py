"""Pre-flight check for a (GPU) machine: environment, files, and a measured training step.

Run before a long job:  leukemia-ml preflight --config configs/ml/gpu/lora_dinobloom_s.json
It builds the configured model, times a few forward/backward steps on random tensors of the
right shape (no dataset needed), records peak GPU memory, and - if the processed run directory is
present - projects the epoch time from the real number of training images.
"""
from __future__ import annotations

import math
import platform
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .embed import resolve_device
from .models.model import build_model
from .train.loop import amp_dtype


def preflight(cfg: ExperimentConfig, steps: int = 3) -> dict:
    dev = resolve_device(cfg.train.device)
    info = {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
            "device": str(dev), "cuda_available": torch.cuda.is_available()}
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        info.update(gpu=props.name, gpu_mem_gb=round(props.total_memory / 1e9, 1),
                    bf16=torch.cuda.is_bf16_supported())
    problems = []
    run_dir = Path(cfg.data.run_dir)
    for f in ("manifest.csv", "cells.csv", "config.json"):
        if not (run_dir / f).exists():
            problems.append(f"missing {run_dir / f}")
    if cfg.model.encoder.startswith("dinobloom") and cfg.model.pretrained:
        if not cfg.model.weights_path or not Path(cfg.model.weights_path).exists():
            problems.append(f"DinoBloom weights not found at {cfg.model.weights_path}")

    if any("weights" in p for p in problems):         # cannot build the model without them
        info["problems"], info["ok"] = problems, False
        return info

    n_feat = 8 if cfg.model.use_features else 0
    if cfg.model.freeze == "frozen":
        info["note"] = "frozen encoder: training uses cached embeddings; step timing is for the head"
    model = build_model(cfg.model, 4 if cfg.data.task == "4way" else 2, n_feat).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(model.trainable_groups(cfg.train), lr=1e-4)
    k, b, s = cfg.data.max_cells, cfg.train.batch_size, cfg.model.input_size
    m = k * b
    n = torch.full((b,), k)
    bag = torch.repeat_interleave(torch.arange(b), n)
    pos = torch.cat([torch.arange(k)] * b)
    dtype = amp_dtype(cfg.train.amp_dtype, dev) if cfg.train.amp else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps + 1):
        x = torch.rand(m, 3, s, s, device=dev)
        feats = torch.randn(m, n_feat, device=dev)
        t0 = time.time()
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
            logits, _ = model(x, feats, bag.to(dev), pos.to(dev), n.to(dev))
            loss = F.cross_entropy(logits.float(), torch.zeros(b, dtype=torch.long, device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i > 0:                                         # first step includes warm-up
            times.append(time.time() - t0)
        if not torch.isfinite(loss):
            problems.append("non-finite loss")
    step_s = float(np.mean(times))
    info.update(trainable_params_m=round(sum(p.numel() for p in params) / 1e6, 3),
                cells_per_step=m, step_seconds=round(step_s, 3),
                cells_per_second=round(m / step_s, 1))
    if dev.type == "cuda":
        info["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if (run_dir / "manifest.csv").exists() and (run_dir / "cells.csv").exists():
        from .data.index import load_index
        from .data.splits import contiguous_split
        idx = load_index(run_dir, cfg.data.task, (), cfg.data.exclude_border_cells)
        if cfg.data.split_scheme == "contiguous":
            idx.image_split = contiguous_split(idx, cfg.data.n_folds, cfg.data.test_fold,
                                               cfg.data.val_offset, cfg.data.embargo)
        n_tr = int((idx.image_split == "train").sum())
        steps_per_epoch = math.ceil(n_tr / cfg.train.batch_size)
        info.update(train_images=n_tr, steps_per_epoch=steps_per_epoch,
                    est_epoch_minutes=round(steps_per_epoch * step_s / 60, 1),
                    est_total_hours_one_fold=round(steps_per_epoch * step_s * cfg.train.epochs / 3600, 2),
                    est_total_hours_all_folds=round(steps_per_epoch * step_s * cfg.train.epochs
                                                    * cfg.data.n_folds / 3600, 2),
                    note_estimate="step time measured at full bags of max_cells; real bags are "
                                  "smaller on average, so this is an upper bound")
    info["problems"] = problems
    info["ok"] = not problems
    return info
