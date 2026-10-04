"""Command line interface:  leukemia-pp run | inspect"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import process_array, run
from .stain import ReinhardReference


def _load_cfg(path: Path | None) -> PipelineConfig:
    return PipelineConfig.from_json(path) if path else PipelineConfig()


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    summary = run(args.input, args.output, cfg, workers=args.workers,
                  dry_run=args.dry_run, force=args.force)
    return 3 if summary.get("n_failed") else 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Process one image and write its QC panel. Uses the stain reference of a previous run
    (`--reference`), or the image itself as its own reference (near-identity)."""
    from .io import read_image
    from .preview import save_preview
    from .stain import fit_reference
    cfg = _load_cfg(args.config)
    raw = read_image(args.image, cfg.data.max_side)
    if not cfg.stain.enabled:
        ref = None
    elif args.reference:
        ref = ReinhardReference.load(args.reference)
    else:
        ref = fit_reference([raw], cfg.stain)
    res = process_array(raw, cfg, ref)
    save_preview(args.out, raw, res, cfg, args.image.name)
    print(f"{res.seg.n_cells} cell(s); flags: {res.image_row['qc_flags'] or 'none'}; -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="leukemia-pp", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process a dataset of <input>/<Class>/*.jpg")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--config", type=Path, help="JSON file overriding PipelineConfig fields")
    r.add_argument("--workers", type=int, default=None, help="default: all cores")
    r.add_argument("--dry-run", action="store_true", help="only scan, split, write manifest")
    r.add_argument("--force", action="store_true",
                   help="reuse an output dir created with a different config")
    r.set_defaults(func=_cmd_run)

    i = sub.add_parser("inspect", help="write a QC panel for one image")
    i.add_argument("image", type=Path)
    i.add_argument("--out", type=Path, default=Path("inspect.png"))
    i.add_argument("--config", type=Path)
    i.add_argument("--reference", type=Path, help="stain_reference.json from a previous run")
    i.set_defaults(func=_cmd_inspect)

    args = ap.parse_args(argv)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
