"""Training / evaluation loop for bag classifiers.

Two data paths share one model:
  * frozen encoder  -> embeddings are extracted ONCE and bags are built from the cache
                       (epochs take seconds; no pixel augmentation is possible);
  * lora / last_k / full -> crops flow through the encoder every step, with augmentation.
Model selection uses validation macro-F1 only; the test split is evaluated once, at the end.

Reliability features for long / pre-emptible GPU jobs:
  * `last.pt` is written atomically after every epoch (trainable weights, optimizer, scheduler,
    AMP scaler, RNG states, history); a re-run in the same directory resumes EXACTLY where it
    stopped (verified by a test: interrupted + resumed == uninterrupted);
  * a finished run (metrics.json with the same config hash) is never recomputed.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.crops import build_crop_cache
from ..data.datasets import (BagDataset, EmbeddingBagDataset, FeatureScaler, collate_bags,
                             load_crops)
from ..data.index import CellIndex, load_index
from ..data.splits import contiguous_split
from ..embed import extract_embeddings, resolve_device
from ..eval import calibration, stats
from ..eval.metrics import balanced_accuracy, macro_f1, summarize
from ..models.model import LeukemiaModel, build_model

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_subset(index: CellIndex, images: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    if n is None or n >= len(images):
        return images
    rng = np.random.default_rng(seed)
    labels = index.image_label[images]
    take = []
    per = max(1, n // max(1, len(set(labels))))
    for c in sorted(set(labels)):
        pool = images[labels == c]
        take.append(rng.choice(pool, min(per, len(pool)), replace=False))
    return np.sort(np.concatenate(take))


def class_weights(labels: np.ndarray, n_classes: int, mode: str) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float).clip(min=1)
    if mode == "none":
        w = np.ones(n_classes)
    elif mode == "inv":
        w = counts.sum() / (n_classes * counts)
    else:
        w = np.sqrt(counts.sum() / (n_classes * counts))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_scheduler(opt, total_steps: int, warmup_frac: float):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k in names or not k.startswith("encoder.")}


def amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if name == "fp16":
        return torch.float16
    if name == "bf16" or (name == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def predict(model: LeukemiaModel, loader: DataLoader, device: torch.device):
    """Image-level logits (N,C), labels, image ids and per-cell attention for every bag."""
    model.eval()
    logits, labels, images, attns = [], [], [], []
    for b in loader:
        lg, at = model(b["crops"].to(device, non_blocking=True), b["feats"].to(device),
                       b["bag"].to(device), b["pos"].to(device), b["n"].to(device))
        logits.append(lg.float().cpu())
        labels.append(b["label"])
        images.append(b["image"])
        for k, n in enumerate(b["n"].tolist()):
            attns.append(at[k, :n].float().cpu().numpy())
    return (torch.cat(logits).numpy(), torch.cat(labels).numpy(), torch.cat(images).numpy(), attns)


class Trainer:
    def __init__(self, cfg: ExperimentConfig, out_dir: str | Path, index: CellIndex | None = None,
                 crops: np.ndarray | None = None):
        self.cfg, self.out = cfg, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.train.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        seed_everything(cfg.train.seed)
        d = cfg.data
        self.index = index or load_index(d.run_dir, d.task, d.feature_groups, d.exclude_border_cells)
        if d.split_scheme == "contiguous":
            sp = contiguous_split(self.index, d.n_folds, d.test_fold, d.val_offset, d.embargo)
            self.index.image_split = sp
            self.index.split = sp[self.index.image_of_cell]
        self.crops_dir = self.out          # embedding cache lives next to the run if no crop cache
        self.crops = crops
        self.emb_dim = None
        self.gen = torch.Generator().manual_seed(cfg.train.seed)
        if crops is None:
            self.crops_dir = build_crop_cache(d, self.index, workers=4)
            self.crops = load_crops(self.crops_dir)

    # ------------------------------------------------------------------ setup
    def _loaders(self):
        c, idx = self.cfg, self.index
        tr_imgs = stratified_subset(idx, idx.image_indices("train"), c.train.max_train_images, c.train.seed)
        va_imgs = stratified_subset(idx, idx.image_indices("val"), c.train.max_eval_images, c.train.seed)
        te_imgs = stratified_subset(idx, idx.image_indices("test"), c.train.max_eval_images, c.train.seed)
        if min(len(tr_imgs), len(va_imgs), len(te_imgs)) == 0:
            raise ValueError("empty train/val/test split; check split settings (embargo too large?)")
        train_cells = np.isin(idx.image_of_cell, tr_imgs)
        scaler = FeatureScaler(idx.features[train_cells]) if idx.features.shape[1] else None
        use_cache = c.model.freeze == "frozen"
        if use_cache:
            emb, info = extract_embeddings(self.crops, c.model, self.crops_dir, self.device)
            self.embed_info, self.emb_dim = info, int(emb.shape[1])
            mk = lambda imgs, train: EmbeddingBagDataset(          # noqa: E731
                idx, np.asarray(emb), imgs, train=train, max_cells=c.data.max_cells if train
                else c.data.max_cells_eval, scaler=scaler, seed=c.train.seed)
        else:
            mk = lambda imgs, train: BagDataset(                   # noqa: E731
                idx, self.crops, imgs, train=train, max_cells=c.data.max_cells if train
                else c.data.max_cells_eval, aug=c.aug if train else None, scaler=scaler,
                input_size=c.model.input_size, seed=c.train.seed)
        pin = self.device.type == "cuda"
        dl = lambda ds, shuffle: DataLoader(                       # noqa: E731
            ds, batch_size=c.train.batch_size, shuffle=shuffle, collate_fn=collate_bags,
            num_workers=c.train.num_workers, generator=self.gen if shuffle else None,
            persistent_workers=c.train.num_workers > 0 and shuffle, pin_memory=pin)
        tr, va, te = mk(tr_imgs, True), mk(va_imgs, False), mk(te_imgs, False)
        return tr, dl(tr, True), dl(va, False), dl(te, False), tr_imgs

    def _build(self, n_features: int, group_slices):
        if self.cfg.model.freeze == "frozen":       # embeddings are precomputed: no backbone at all
            m = LeukemiaModel(None, self.emb_dim, self.index.n_classes, self.cfg.model,
                              n_features, group_slices)
        else:
            m = build_model(self.cfg.model, self.index.n_classes, n_features, group_slices)
        return m.to(self.device)

    # ------------------------------------------------------------------ fit
    def fit(self, max_epochs_this_call: int | None = None) -> dict:
        """Train (or resume). `max_epochs_this_call` stops early after N epochs, simulating a
        pre-emption: a later call resumes from `last.pt`."""
        c, t0 = self.cfg, time.time()
        done = self.out / "metrics.json"
        if done.exists():
            prev = json.loads(done.read_text(encoding="utf-8"))
            if prev.get("config_hash") == c.hash():
                log.info("finished run found in %s; skipping", self.out)
                return prev
        train_ds, train_dl, val_dl, test_dl, tr_imgs = self._loaders()
        n_feat = self.index.features.shape[1] if c.model.use_features else 0
        # Re-seed right before parameters are created: loading/extracting embeddings consumes RNG
        # only on a cold cache, so without this a warm cache would change the head's init.
        seed_everything(c.train.seed)
        self.gen.manual_seed(c.train.seed)
        model = self._build(n_feat, None)
        w = class_weights(self.index.image_label[tr_imgs], self.index.n_classes,
                          c.train.class_weighting).to(self.device)
        opt = torch.optim.AdamW(model.trainable_groups(c.train), weight_decay=c.train.weight_decay)
        total = max(1, len(train_dl) * c.train.epochs)
        sched = make_scheduler(opt, total, c.train.warmup_frac)
        dtype = amp_dtype(c.train.amp_dtype, self.device) if c.train.amp else None
        scaler_amp = torch.amp.GradScaler(enabled=dtype == torch.float16)
        n_train_params = sum(p.numel() for g in opt.param_groups for p in g["params"])
        log.info("trainable params: %.2fM | steps/epoch %d | device %s | amp %s",
                 n_train_params / 1e6, len(train_dl), self.device, dtype)

        best, best_state, bad, history, start = -1.0, None, 0, [], 0
        last = self.out / "last.pt"
        if last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=False)
            if ck["config_hash"] == c.hash():
                model.load_state_dict(ck["model"], strict=False)
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                scaler_amp.load_state_dict(ck["scaler"])
                torch.set_rng_state(ck["rng"])
                self.gen.set_state(ck["gen"])
                best, best_state, bad = ck["best"], ck["best_state"], ck["bad"]
                history, start = ck["history"], ck["epoch"] + 1
                log.info("resumed from epoch %d", start)
        log_f = open(self.out / "train_log.jsonl", "a" if start else "w", encoding="utf-8")

        ran = 0
        for epoch in range(start, c.train.epochs):
            if max_epochs_this_call is not None and ran >= max_epochs_this_call:
                log_f.close()
                return {"interrupted": True, "epochs_done": epoch}
            train_ds.set_epoch(epoch)
            model.train()
            if model.encoder is not None and not any(p.requires_grad for p in model.encoder.parameters()):
                model.encoder.eval()
            t_ep, seen_cells, loss_sum, n_seen = time.time(), 0, 0.0, 0
            for b in train_dl:
                crops = b["crops"].to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, dtype=dtype, enabled=dtype is not None):
                    logits, _ = model(crops, b["feats"].to(self.device), b["bag"].to(self.device),
                                      b["pos"].to(self.device), b["n"].to(self.device))
                    loss = F.cross_entropy(logits.float(), b["label"].to(self.device), weight=w,
                                           label_smoothing=c.train.label_smoothing)
                opt.zero_grad(set_to_none=True)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                               c.train.grad_clip)
                scaler_amp.step(opt)
                scaler_amp.update()
                sched.step()
                loss_sum += float(loss.detach()) * len(b["label"])
                n_seen += len(b["label"])
                seen_cells += len(crops)
            train_secs = time.time() - t_ep
            lg, y, _, _ = predict(model, val_dl, self.device)
            pred = lg.argmax(1)
            val_f1 = macro_f1(y, pred, self.index.n_classes)
            rec = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), "val_macro_f1": val_f1,
                   "val_bal_acc": balanced_accuracy(y, pred, self.index.n_classes),
                   "lr": [g["lr"] for g in opt.param_groups], "train_seconds": round(train_secs, 2),
                   "cells_per_s": round(seen_cells / max(train_secs, 1e-9), 1)}
            if self.device.type == "cuda":
                rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            history.append(rec)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            log.info("epoch %d | loss %.4f | val macroF1 %.3f | %.1fs (%.0f cells/s)", epoch,
                     rec["train_loss"], val_f1, train_secs, rec["cells_per_s"])
            stop = False
            if val_f1 > best:
                best, bad, best_state = val_f1, 0, trainable_state(model)
            else:
                bad += 1
                stop = bad >= c.train.patience
            ran += 1
            _atomic_save({"config_hash": c.hash(), "epoch": epoch, "model": trainable_state(model),
                          "opt": opt.state_dict(), "sched": sched.state_dict(),
                          "scaler": scaler_amp.state_dict(), "rng": torch.get_rng_state(),
                          "gen": self.gen.get_state(), "best": best, "best_state": best_state,
                          "bad": bad, "history": history}, last)
            if stop:
                log.info("early stopping at epoch %d", epoch)
                break
        log_f.close()

        model.load_state_dict(best_state, strict=False)
        result = self._final_eval(model, val_dl, test_dl)
        result.update({"best_val_macro_f1": best, "epochs_run": len(history), "history": history,
                       "trainable_params": n_train_params, "seconds": round(time.time() - t0, 1),
                       "config_hash": c.hash(),
                       "split": {"scheme": c.data.split_scheme, "test_fold": c.data.test_fold,
                                 "embargo": c.data.embargo,
                                 "n_train_images": int(len(tr_imgs)),
                                 "n_val_images": int(len(val_dl.dataset)),
                                 "n_test_images": int(len(test_dl.dataset))}})
        (self.out / "config.json").write_text(json.dumps(c.to_dict(), indent=2), encoding="utf-8")
        (self.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        torch.save(best_state, self.out / "model.pt")
        return result

    def _final_eval(self, model, val_dl, test_dl) -> dict:
        idx, n_classes = self.index, self.index.n_classes
        v_lg, v_y, _, _ = predict(model, val_dl, self.device)
        temperature = calibration.fit_temperature(v_lg, v_y)
        t_lg, t_y, t_img, attn = predict(model, test_dl, self.device)
        probs = calibration.apply_temperature(t_lg, 1.0)
        probs_cal = calibration.apply_temperature(t_lg, temperature)
        groups = idx.image_group[t_img]
        pred = probs.argmax(1)
        ci = {
            "balanced_accuracy": stats.cluster_bootstrap(
                lambda i: balanced_accuracy(t_y[i], pred[i], n_classes), groups, n_boot=1000),
            "macro_f1": stats.cluster_bootstrap(
                lambda i: macro_f1(t_y[i], pred[i], n_classes), groups, n_boot=1000),
        }
        with open(self.out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image_id", "group", "label", "pred"] + [f"p_{k}" for k in range(n_classes)]
                        + ["n_cells", "top_attention"])
            for k in range(len(t_y)):
                wr.writerow([idx.images[t_img[k]], groups[k], int(t_y[k]), int(pred[k])]
                            + [f"{p:.5f}" for p in probs_cal[k]] + [len(attn[k]),
                                                                   f"{float(attn[k].max()):.4f}"])
        return {"test": summarize(t_y, probs), "test_calibrated": summarize(t_y, probs_cal),
                "temperature": temperature, "test_ci": ci,
                "val": summarize(v_y, calibration.apply_temperature(v_lg, 1.0))}
