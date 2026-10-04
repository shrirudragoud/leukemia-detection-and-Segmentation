"""leukemia-ml: crops | embed | probe | train | ablate"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig


def _cfg(args) -> ExperimentConfig:
    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    return cfg


def _cmd_crops(args) -> int:
    from .data.crops import build_crop_cache
    from .data.index import load_index
    cfg = _cfg(args)
    idx = load_index(cfg.data.run_dir, cfg.data.task, (), cfg.data.exclude_border_cells)
    print(build_crop_cache(cfg.data, idx, workers=args.workers, force=args.force))
    return 0


def _cmd_embed(args) -> int:
    from .data.crops import build_crop_cache
    from .data.datasets import load_crops
    from .data.index import load_index
    from .embed import extract_embeddings
    cfg = _cfg(args)
    idx = load_index(cfg.data.run_dir, cfg.data.task, (), cfg.data.exclude_border_cells)
    d = build_crop_cache(cfg.data, idx)
    emb, info = extract_embeddings(load_crops(d), cfg.model, d, cfg.train.device,
                                   limit=args.limit, force=args.force)
    print(json.dumps({"shape": list(emb.shape), **info}, indent=2))
    return 0


def _cmd_probe(args) -> int:
    from .data.crops import build_crop_cache
    from .data.datasets import load_crops
    from .data.index import load_index
    from .embed import extract_embeddings
    from .eval.probe import image_features, run_probe
    cfg = _cfg(args)
    idx = load_index(cfg.data.run_dir, cfg.data.task, cfg.data.feature_groups,
                     cfg.data.exclude_border_cells)
    d = build_crop_cache(cfg.data, idx)
    emb, _ = extract_embeddings(load_crops(d), cfg.model, d, cfg.train.device)
    X, img = image_features(idx, np.asarray(emb), idx.features if cfg.model.use_features else None)
    res = run_probe(idx, X, img, seeds=tuple(range(args.seeds)))
    print(json.dumps({"mean": res["mean"], "std": res["std"], "n_images": res["n_images"],
                      "n_groups": res["n_groups"]}, indent=2))
    return 0


def _cmd_train(args) -> int:
    from .train.loop import Trainer
    cfg = _cfg(args)
    out = Path(args.out) / f"{cfg.name}_{cfg.hash()}"
    res = Trainer(cfg, out).fit()
    t = res["test"]
    print(json.dumps({"out": str(out), "test_balanced_accuracy": t["balanced_accuracy"],
                      "test_macro_f1": t["macro_f1"], "epochs": res["epochs_run"],
                      "seconds": res["seconds"]}, indent=2))
    return 0


def _cmd_ablate(args) -> int:
    from .experiments import load_variants, run_probe_ablation
    base, variants = load_variants(args.spec)
    rep = run_probe_ablation(base, variants, args.out, seeds=tuple(range(args.seeds)),
                             n_boot=args.boot, device=args.device)
    print((Path(args.out) / "ablation.md").read_text(encoding="utf-8"))
    return 0 if rep["rows"] else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="leukemia-ml", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("crops", _cmd_crops), ("embed", _cmd_embed), ("probe", _cmd_probe),
                     ("train", _cmd_train)):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path)
        p.set_defaults(func=fn)
        if name == "crops":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--force", action="store_true")
        if name == "embed":
            p.add_argument("--limit", type=int)
            p.add_argument("--force", action="store_true")
        if name == "probe":
            p.add_argument("--seeds", type=int, default=3)
        if name == "train":
            p.add_argument("--out", type=Path, default=Path("runs"))
    ab = sub.add_parser("ablate", help="compare variants with grouped-CV probes + bootstrap stats")
    ab.add_argument("--spec", type=Path, required=True)
    ab.add_argument("--out", type=Path, required=True)
    ab.add_argument("--seeds", type=int, default=3)
    ab.add_argument("--boot", type=int, default=1000)
    ab.add_argument("--device", default="auto")
    ab.set_defaults(func=_cmd_ablate)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
