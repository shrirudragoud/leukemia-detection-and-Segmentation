"""Torch datasets: cells, bags (all cells of an image) and precomputed embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AugConfig
from .augment import augment
from .index import CellIndex

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(crop: np.ndarray, size: int | None = None) -> torch.Tensor:
    """uint8 HxWx3 -> float CxHxW in [0, 1] (normalisation happens inside the model)."""
    if size is not None and crop.shape[0] != size:
        import cv2
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float().div_(255.0)


class FeatureScaler:
    """Median-impute + standardise per-cell features, fitted on TRAIN cells only."""

    def __init__(self, features: np.ndarray):
        self.median = np.nanmedian(features, axis=0) if features.size else np.zeros(0)
        self.median = np.nan_to_num(self.median)
        filled = np.where(np.isfinite(features), features, self.median)
        self.mean = filled.mean(axis=0) if features.size else np.zeros(0)
        self.std = np.maximum(filled.std(axis=0), 1e-6) if features.size else np.ones(0)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(features), features, self.median)
        return ((filled - self.mean) / self.std).astype(np.float32)


class BagDataset(Dataset):
    """One item = one image = a bag of its cells (random subset when training)."""

    def __init__(self, index: CellIndex, crops: np.ndarray, images: np.ndarray, *,
                 train: bool, max_cells: int, aug: AugConfig | None = None,
                 scaler: FeatureScaler | None = None, input_size: int | None = None,
                 seed: int = 0):
        self.index, self.crops, self.train = index, crops, train
        self.images = np.array([i for i in images if len(index.cells_of_image(i)) > 0])
        self.max_cells, self.aug, self.scaler, self.input_size = max_cells, aug, scaler, input_size
        self.seed, self.epoch = seed, 0
        self.scaled = scaler(index.features) if scaler is not None and index.features.shape[1] else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, k: int) -> dict:
        img = int(self.images[k])
        rows = self.index.cells_of_image(img)
        rng = np.random.default_rng([self.seed, self.epoch, img])
        if len(rows) > self.max_cells:
            rows = (rng.choice(rows, self.max_cells, replace=False) if self.train
                    else rows[: self.max_cells])
            rows = np.sort(rows)
        crops = []
        for r in rows:
            c = np.asarray(self.crops[r])
            if self.train and self.aug is not None:
                c = augment(c, self.aug, rng)
            crops.append(to_tensor(c, self.input_size))
        feats = (torch.from_numpy(self.scaled[rows]) if self.scaled is not None
                 else torch.zeros(len(rows), 0))
        return {"crops": torch.stack(crops), "feats": feats, "label": int(self.index.image_label[img]),
                "image": img, "rows": torch.as_tensor(rows)}


def collate_bags(batch: list[dict]) -> dict:
    """Concatenate all cells of the batch; `bag`/`pos` say where each cell belongs."""
    n = torch.tensor([len(b["crops"]) for b in batch])
    return {
        "crops": torch.cat([b["crops"] for b in batch]),
        "feats": torch.cat([b["feats"] for b in batch]),
        "bag": torch.repeat_interleave(torch.arange(len(batch)), n),
        "pos": torch.cat([torch.arange(k) for k in n.tolist()]),
        "n": n,
        "label": torch.tensor([b["label"] for b in batch]),
        "image": torch.tensor([b["image"] for b in batch]),
        "rows": torch.cat([b["rows"] for b in batch]),
    }


class EmbeddingBagDataset(Dataset):
    """Bags of PRECOMPUTED cell embeddings (frozen encoder): epochs take seconds."""

    def __init__(self, index: CellIndex, emb: np.ndarray, images: np.ndarray, *, train: bool,
                 max_cells: int, scaler: FeatureScaler | None = None, seed: int = 0):
        self.index, self.emb, self.train = index, emb, train
        self.images = np.array([i for i in images if len(index.cells_of_image(i)) > 0])
        self.max_cells, self.seed, self.epoch = max_cells, seed, 0
        self.scaled = scaler(index.features) if scaler is not None and index.features.shape[1] else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, k: int) -> dict:
        img = int(self.images[k])
        rows = self.index.cells_of_image(img)
        rng = np.random.default_rng([self.seed, self.epoch, img])
        if len(rows) > self.max_cells:
            rows = (rng.choice(rows, self.max_cells, replace=False) if self.train
                    else rows[: self.max_cells])
            rows = np.sort(rows)
        feats = (torch.from_numpy(self.scaled[rows]) if self.scaled is not None
                 else torch.zeros(len(rows), 0))
        return {"crops": torch.from_numpy(self.emb[rows].astype(np.float32)), "feats": feats,
                "label": int(self.index.image_label[img]), "image": img,
                "rows": torch.as_tensor(rows)}


def load_crops(cache_dir: str | Path) -> np.ndarray:
    return np.load(Path(cache_dir) / "crops.npy", mmap_mode="r")
