"""Frozen, validated, hashable experiment configuration (same conventions as leukemia_pp)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SOURCES = ("raw", "norm", "clean_rgb", "clean_gray")
TASKS = ("4way", "binary")
FREEZE_MODES = ("frozen", "lora", "last_k", "full")
HEADS = ("cell", "mean", "mil")


@dataclass(frozen=True)
class DataConfig:
    run_dir: str = "data/full_run_v2"          # output directory of `leukemia-pp run`
    input_dir: str | None = "data/dataset/Original"   # raw images (only needed for source='raw')
    source: str = "clean_rgb"                  # which pixels the network sees
    isolate: bool = True                       # blank everything except THIS cell inside its crop
    cache_size: int = 128                      # stored crop side (px)
    margin: float = 0.25                       # context around the cell, fraction of its box
    task: str = "4way"
    max_cells: int = 24                        # train-time cap per bag (random subset)
    max_cells_eval: int = 64                   # eval-time cap per bag (first by cell id)
    feature_groups: tuple[str, ...] = ("cell_shape", "cell_texture", "apc", "log")
    exclude_border_cells: bool = False

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.max_cells < 1 or self.max_cells_eval < 1:
            raise ValueError("max_cells must be >= 1")


@dataclass(frozen=True)
class ModelConfig:
    # 'timm:<name>' | 'dinov2_s' | 'dinov2_b' | 'dinobloom_s' | 'dinobloom_b'
    encoder: str = "dinobloom_s"
    pretrained: bool = True
    weights_path: str | None = None            # DinoBloom .pth (Zenodo 10908163)
    input_size: int = 112                      # network input side (ViT: multiple of 14)
    freeze: str = "frozen"
    last_k: int = 2                            # blocks to unfreeze for freeze='last_k'
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("attn.qkv", "attn.proj")
    head: str = "mil"
    hidden: int = 256
    dropout: float = 0.2
    use_features: bool = False                 # hybrid: add the interpretable feature branch
    feature_hidden: int = 32
    feature_group_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.freeze not in FREEZE_MODES:
            raise ValueError(f"freeze must be one of {FREEZE_MODES}")
        if self.head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")


@dataclass(frozen=True)
class AugConfig:
    flips: bool = True
    rot90: bool = True
    scale_jitter: float = 0.10                 # random zoom +-fraction
    colour: str = "hed"                        # 'none' | 'hsv' | 'hed'
    colour_strength: float = 0.05
    grayscale_p: float = 0.2                   # random grayscale: discourages relying on stain colour
    blur_p: float = 0.1

    def __post_init__(self) -> None:
        if self.colour not in ("none", "hsv", "hed"):
            raise ValueError("colour must be none|hsv|hed")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 8                        # bags (images) per step
    lr_head: float = 1e-3
    lr_encoder: float = 1e-5
    lr_lora: float = 1e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.1
    label_smoothing: float = 0.1
    class_weighting: str = "inv_sqrt"          # 'none' | 'inv_sqrt' | 'inv'
    grad_clip: float = 1.0
    patience: int = 5
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"                       # 'auto' | 'cpu' | 'cuda'
    amp: bool = True                           # only used on CUDA
    max_train_images: int | None = None        # smoke tests
    max_eval_images: int | None = None

    def __post_init__(self) -> None:
        if self.class_weighting not in ("none", "inv_sqrt", "inv"):
            raise ValueError("class_weighting must be none|inv_sqrt|inv")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "experiment"

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        sections = {"data": DataConfig, "model": ModelConfig, "aug": AugConfig, "train": TrainConfig}
        unknown = set(data) - set(sections) - {"name"}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k == "name"}
        for key, sec in sections.items():
            if key in data:
                names = {f.name for f in dataclasses.fields(sec)}
                bad = set(data[key]) - names
                if bad:
                    raise ValueError(f"unknown keys for {sec.__name__}: {sorted(bad)}")
                default = sec()
                vals = {}
                for k, v in data[key].items():
                    if isinstance(getattr(default, k), tuple) and isinstance(v, list):
                        v = tuple(v)
                    vals[k] = v
                kwargs[key] = sec(**vals)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
