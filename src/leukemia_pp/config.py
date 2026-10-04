"""Frozen, validated, serialisable configuration.

Every algorithm parameter lives here. The resolved config is written next to the outputs
(`config.json`) and hashed, so a processed dataset can always be traced to the exact
parameters that produced it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# class folder name -> (binary label: 0 benign / 1 malignant, four-way label)
DEFAULT_CLASS_MAP: dict[str, tuple[int, int]] = {
    "Benign": (0, 0),
    "Early": (1, 1),
    "Pre": (1, 2),
    "Pro": (1, 3),
}


@dataclass(frozen=True)
class DataConfig:
    # Downscale (never upscale) so the longest side is at most this many px. None = keep.
    max_side: int | None = None


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    # Optional regex whose first group identifies the patient/slide an image came from.
    # All images sharing a group are kept in the same split (prevents patient leakage).
    # Without it, exact duplicate files (same sha256) are still kept together.
    group_regex: str | None = None

    def __post_init__(self) -> None:
        ratios = (self.train, self.val, self.test)
        if min(ratios) < 0 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"split ratios must be >= 0 and sum to 1, got {ratios}")
        if self.train <= 0:
            raise ValueError("train ratio must be > 0")


@dataclass(frozen=True)
class StainConfig:
    enabled: bool = True
    ref_sample_size: int = 20          # train images (all classes) used to fit the reference
    min_foreground_frac: float = 0.02  # below this, statistics fall back to all pixels
    scale_clip: tuple[float, float] = (0.25, 4.0)  # bounds on the per-channel std gain


@dataclass(frozen=True)
class DenoiseConfig:
    method: str = "bilateral"          # 'bilateral' | 'nlm' | 'none'
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    nlm_h: float = 5.0
    nlm_h_color: float = 5.0

    def __post_init__(self) -> None:
        if self.method not in ("bilateral", "nlm", "none"):
            raise ValueError(f"unknown denoise method {self.method!r}")


@dataclass(frozen=True)
class ResponseConfig:
    apc_sigma: float = 2.5
    apc_alpha: float = 0.6
    log_sigma: float = 4.0
    # Fixed (not per-image) gains applied to the sigma^2-normalised responses before clipping
    # to [0, 1]. Measured on stain-normalised data: p99 of APC ~0.04 and of LoG ~0.09.
    apc_gain: float = 12.0
    log_gain: float = 5.0
    canny_low_ratio: float = 0.5       # low threshold = ratio * (Otsu high threshold)


@dataclass(frozen=True)
class SegmentationConfig:
    # --- whole-cell (WBC) instance segmentation, on the Lab a* channel -------------------
    smooth_sigma: float = 1.0
    a_floor: float = 8.0               # a* below this is never WBC, whatever Otsu says
    min_cell_area: int = 120           # px; removes platelets, stain specks, label text
    open_radius: int = 1
    split_h: float = 1.5               # h-maxima depth (px) needed to split touching cells
    split_smooth: float = 1.0
    # --- per-cell nucleus segmentation, on Lab L* ---------------------------------------
    nucleus_smooth_sigma: float = 1.0
    # Edge pixels are blended with background by smoothing; exclude this many px of rim when
    # estimating the nucleus/cytoplasm split, or every cell looks 'bimodal' (bright rim).
    nucleus_rim_erode: int = 2
    nucleus_min_contrast: float = 8.0  # min L* gap between nucleus and cytoplasm classes
    nucleus_min_area: int = 30
    nucleus_min_cell_frac: float = 0.05
    nucleus_keep_frac: float = 0.2     # keep nucleus lobes >= this fraction of the largest


@dataclass(frozen=True)
class FeatureConfig:
    glcm_levels: int = 32
    glcm_distances: tuple[int, ...] = (1, 2, 3)
    glcm_angles_deg: tuple[int, ...] = (0, 45, 90, 135)
    min_texture_pixels: int = 30


@dataclass(frozen=True)
class QCConfig:
    min_sharpness: float = 20.0        # variance of the Laplacian of the grayscale image
    min_brightness: float = 40.0       # mean L* (0-100); darker fields are under-exposed
    max_wbc_fraction: float = 0.5      # a field that is mostly "WBC" is a segmentation failure
    max_border_cell_frac: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.max_wbc_fraction <= 1:
            raise ValueError("max_wbc_fraction must be in (0, 1]")


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "split": SplitConfig,
    "stain": StainConfig,
    "denoise": DenoiseConfig,
    "response": ResponseConfig,
    "segmentation": SegmentationConfig,
    "features": FeatureConfig,
    "qc": QCConfig,
}


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    stain: StainConfig = field(default_factory=StainConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    class_map: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(DEFAULT_CLASS_MAP))
    save_previews: bool = True

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build from a (possibly partial) dict. Unknown keys raise, so typos are caught."""
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            if name in _SECTIONS:
                kwargs[name] = _build_section(_SECTIONS[name], value)
            elif name == "class_map":
                kwargs[name] = {k: (int(v[0]), int(v[1])) for k, v in value.items()}
            else:
                kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path) -> PipelineConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _build_section(section_cls: type, values: dict[str, Any]):
    names = {f.name: f for f in dataclasses.fields(section_cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    default = section_cls()
    kwargs = {}
    for key, value in values.items():
        if isinstance(getattr(default, key), tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return section_cls(**kwargs)
