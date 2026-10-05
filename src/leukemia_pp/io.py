"""Image discovery and robust loading."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
MIN_SIDE = 16


class ImageReadError(RuntimeError):
    """Raised when a file cannot be decoded into a usable image."""


@dataclass(frozen=True)
class Sample:
    image_id: str
    path: Path
    rel_path: str      # posix path relative to the input root (portable across machines)
    class_name: str
    sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(input_dir: Path, class_names) -> list[Sample]:
    """Find images under `<input_dir>/<class_name>/**`. Deterministic order.

    Folders that are not a known class are reported and ignored. Duplicate image ids
    (file stems) are an error: they would silently overwrite each other's outputs.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    class_names = set(class_names)
    samples: list[Sample] = []
    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            log.warning("ignoring folder %s (not one of %s)", class_dir.name, sorted(class_names))
            continue
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append(Sample(
                    image_id=p.stem,
                    path=p.resolve(),
                    rel_path=p.relative_to(input_dir).as_posix(),
                    class_name=class_dir.name,
                    sha256=sha256_file(p),
                ))
    seen: dict[str, Sample] = {}
    for s in samples:
        if s.image_id in seen:
            raise ValueError(
                f"duplicate image id {s.image_id!r}: {seen[s.image_id].rel_path} and {s.rel_path}")
        seen[s.image_id] = s
    return samples


def read_image(path: Path, max_side: int | None = None) -> np.ndarray:
    """Decode to 8-bit BGR. Handles unicode paths on Windows; drops alpha; expands gray."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError as e:
        raise ImageReadError(f"cannot read {path}: {e}") from e
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        raise ImageReadError(f"cannot decode {path}")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise ImageReadError(f"{path} is too small ({w}x{h})")
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img
