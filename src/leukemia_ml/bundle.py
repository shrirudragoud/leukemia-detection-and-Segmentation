"""Pack everything a remote (GPU) machine needs into one .tar.gz, with a checksum manifest.

Contents: the tabular outputs of `leukemia-pp run` (manifest.csv, cells.csv, images.csv,
config.json, ...) and the crop cache(s) for the requested config, optionally embeddings.
Not included: raw images (third-party, regenerate with `leukemia-pp run`) and pretrained
weights (download from Zenodo 10908163; see the runbook).
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from .config import ExperimentConfig
from .data.crops import cache_key

TABLES = ("manifest.csv", "cells.csv", "images.csv", "config.json", "stain_reference.json",
          "run_summary.json", "audit.json", "audit.md")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def make_bundle(cfg: ExperimentConfig, out_path: str | Path, include_embeddings: bool = False,
                root_name: str = "run") -> dict:
    run_dir = Path(cfg.data.run_dir)
    run_hash = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config_hash"]
    crops = run_dir / "ml_cache" / f"crops_{cache_key(cfg.data, run_hash)}"
    if not (crops / "crops.npy").exists():
        raise FileNotFoundError(f"crop cache {crops} not built; run `leukemia-ml crops` first")
    files = [run_dir / t for t in TABLES if (run_dir / t).exists()]
    files += [crops / "crops.npy", crops / "meta.json"]
    if include_embeddings:
        files += sorted(crops.glob("emb_*.npy"))
    manifest = {"config_hash": run_hash, "crop_cache": crops.name, "files": {}}
    out_path = Path(out_path)
    with tarfile.open(out_path, "w:gz", compresslevel=3) as tar:
        for f in files:
            arc = f"{root_name}/{f.relative_to(run_dir)}"
            manifest["files"][arc] = {"bytes": f.stat().st_size, "sha256": _sha(f)}
            tar.add(f, arcname=arc)
        mpath = out_path.with_suffix(".manifest.json")
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tar.add(mpath, arcname=f"{root_name}/BUNDLE.json")
    manifest["tar_bytes"] = out_path.stat().st_size
    return manifest


def verify_extracted(root: str | Path) -> list[str]:
    """Check an extracted bundle against its BUNDLE.json; returns a list of problems."""
    root = Path(root)
    spec = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
    bad = []
    for arc, meta in spec["files"].items():
        p = root / Path(arc).relative_to(Path(arc).parts[0])
        if not p.exists():
            bad.append(f"missing {p}")
        elif p.stat().st_size != meta["bytes"] or _sha(p) != meta["sha256"]:
            bad.append(f"corrupt {p}")
    return bad
