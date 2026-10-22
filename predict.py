#!/usr/bin/env python3
"""Predict leukemia cell type from a blood smear image.

Usage:
    python predict.py <image_path> [--model-dir runs/cv_dinobloom_s_frozen_mil_*] [--fold 0]

Runs the full pipeline: stain normalisation -> segmentation -> cell cropping ->
DinoBloom-S encoder -> MIL attention head -> 4-class prediction.

Classes: Benign | Early (B-ALL) | Pre (B-ALL) | Pro (B-ALL)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent


def find_model_dir() -> Path:
    """Find the CV run directory with 5-fold models."""
    candidates = sorted(glob.glob(str(ROOT / "runs" / "cv_dinobloom_s_frozen_mil_*")))
    if not candidates:
        sys.exit("No trained model found in runs/. Run cross-validation first.")
    return Path(candidates[-1])


def find_run_dir() -> Path:
    """Find the preprocessing output directory."""
    for d in ["data/full_run_v2", "data/full_run"]:
        p = ROOT / d
        if (p / "config.json").exists():
            return p
    sys.exit("No preprocessing run found. Run `leukemia-pp run` first.")


def load_stain_reference(run_dir: Path):
    """Load the Reinhard stain reference from a completed preprocessing run."""
    from leukemia_pp.stain import ReinhardReference
    ref_path = run_dir / "stain_reference.json"
    if not ref_path.exists():
        return None
    data = json.loads(ref_path.read_text())
    return ReinhardReference(
        mean=np.array(data["mean"], np.float64),
        std=np.array(data["std"], np.float64),
        n_images=data.get("n_images", 20),
    )


def preprocess_image(image_path: str, run_dir: Path):
    """Run stain normalisation, denoising and segmentation on a single image."""
    from leukemia_pp.config import PipelineConfig
    from leukemia_pp.pipeline import process_array

    cfg_data = json.loads((run_dir / "config.json").read_text())
    inner = cfg_data.get("config", cfg_data)
    cfg = PipelineConfig.from_dict(inner)
    reference = load_stain_reference(run_dir)
    raw_bgr = cv2.imread(image_path)
    if raw_bgr is None:
        sys.exit(f"Cannot read image: {image_path}")
    result = process_array(raw_bgr, cfg, reference)
    return result, cfg


def cut_cell_crops(result, cfg, crop_size=128, margin=0.25, isolate=True):
    """Cut individual cell crops from the processed image."""
    from leukemia_ml.data.crops import cut_crop

    if cfg.neutralise.enabled and "rgb_clean" in result.arrays:
        img = result.arrays["rgb_clean"]
    else:
        img = result.arrays["rgb_norm"]

    labels = result.arrays["cell_labels"]
    crops = []
    cell_ids = []
    for row in result.cell_rows:
        cid = row["cell_id"]
        crop = cut_crop(img, labels, cid, crop_size, margin, isolate)
        crops.append(crop)
        cell_ids.append(cid)
    return crops, cell_ids


def load_model(model_dir: Path, fold: int = 0):
    """Build the model and load saved weights."""
    from leukemia_ml.config import ModelConfig
    from leukemia_ml.models.model import build_model

    cfg = ModelConfig(
        encoder="dinobloom_s",
        weights_path=str(ROOT / "data" / "weights" / "DinoBloom-S.pth"),
        input_size=112,
        freeze="frozen",
        head="mil",
        hidden=256,
        dropout=0.2,
        use_features=False,
    )
    model = build_model(cfg, n_classes=4)

    weights_path = model_dir / f"fold{fold}" / "model.pt"
    if not weights_path.exists():
        sys.exit(f"Model weights not found: {weights_path}")

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_image(model, crops: list[np.ndarray], input_size: int = 112):
    """Run MIL prediction on a bag of cell crops."""
    from leukemia_ml.data.datasets import to_tensor

    if not crops:
        return None, None, None

    tensors = [to_tensor(c, input_size) for c in crops]
    crop_batch = torch.stack(tensors)
    feats = torch.zeros(len(crops), 0)
    bag = torch.zeros(len(crops), dtype=torch.long)
    pos = torch.arange(len(crops))
    n = torch.tensor([len(crops)])

    with torch.no_grad():
        logits, attn = model(crop_batch, feats, bag, pos, n)

    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    attention = attn[0, :len(crops)].numpy()
    return probs, attention, logits.squeeze(0).numpy()


CLASS_NAMES = ["Benign", "Early (B-ALL)", "Pre (B-ALL)", "Pro (B-ALL)"]


def main():
    parser = argparse.ArgumentParser(description="Predict leukemia cell type from a blood smear image")
    parser.add_argument("image", help="Path to the blood smear image")
    parser.add_argument("--model-dir", help="Path to the CV run directory (default: auto-detect)")
    parser.add_argument("--fold", type=int, default=0, help="Which fold's model to use (0-4)")
    parser.add_argument("--all-folds", action="store_true", help="Ensemble all 5 folds")
    parser.add_argument("--save-crops", help="Directory to save cell crops to")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else find_model_dir()
    run_dir = find_run_dir()

    print(f"Image:     {args.image}")
    print(f"Model:     {model_dir.name}")
    print(f"Pipeline:  {run_dir}")
    print()

    # Step 1: Preprocess
    print("Preprocessing image (stain normalisation, segmentation)...")
    result, cfg = preprocess_image(args.image, run_dir)
    n_cells = len(result.cell_rows)
    print(f"  Found {n_cells} cells")
    if n_cells == 0:
        print("  No cells detected. Cannot make a prediction.")
        sys.exit(1)

    # Step 2: Cut crops
    print("Cutting cell crops...")
    crops, cell_ids = cut_cell_crops(result, cfg)
    print(f"  Cut {len(crops)} crops (128x128, isolated)")

    if args.save_crops:
        crop_dir = Path(args.save_crops)
        crop_dir.mkdir(parents=True, exist_ok=True)
        for i, (crop, cid) in enumerate(zip(crops, cell_ids)):
            cv2.imwrite(str(crop_dir / f"cell_{cid:03d}.png"), crop[..., ::-1])
        print(f"  Saved crops to {crop_dir}/")

    # Step 3: Load model and predict
    folds = range(5) if args.all_folds else [args.fold]
    all_probs = []

    for fold in folds:
        print(f"Loading model (fold {fold})...")
        model = load_model(model_dir, fold)
        probs, attention, logits = predict_image(model, crops)
        all_probs.append(probs)

        if not args.all_folds:
            # Show per-cell attention
            print(f"\n{'─' * 60}")
            print(f"  Cell attention weights (higher = more influential):")
            sorted_idx = np.argsort(attention.ravel())[::-1]
            for rank, idx in enumerate(sorted_idx[:min(10, len(sorted_idx))]):
                print(f"    Cell {cell_ids[idx]:3d}: attention = {attention.ravel()[idx]:.4f}")

    # Ensemble or single fold
    if args.all_folds:
        avg_probs = np.mean(all_probs, axis=0)
        print(f"\n  Ensembled {len(folds)} folds")
    else:
        avg_probs = all_probs[0]

    pred_class = int(np.argmax(avg_probs))
    confidence = avg_probs[pred_class]

    # Results
    print(f"\n{'═' * 60}")
    print(f"  PREDICTION: {CLASS_NAMES[pred_class]}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"{'═' * 60}")
    print(f"\n  Class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(avg_probs[i] * 40)
        marker = " ◄── predicted" if i == pred_class else ""
        print(f"    {name:15s} {avg_probs[i]:6.1%}  {bar}{marker}")

    # Malignant vs Benign
    p_malignant = avg_probs[1] + avg_probs[2] + avg_probs[3]
    print(f"\n  Benign vs Malignant:")
    print(f"    Benign:    {avg_probs[0]:6.1%}")
    print(f"    Malignant: {p_malignant:6.1%}  (Early + Pre + Pro)")

    print(f"\n  ⚠  Research prototype only — not for clinical diagnosis.")
    print(f"  Model: DinoBloom-S frozen encoder + MIL attention head")
    print(f"  Cells analysed: {len(crops)}")


if __name__ == "__main__":
    main()
