"""Leukemia preprocessing pipeline: raw JPEGs -> cached .npz + features.csv.

Run:
    cd ml_pipeline
    python preprocess.py --smoke              # synthetic self-test
    python preprocess.py                      # full run on ../data/raw
    python preprocess.py --input /path/raw    # custom input
    python preprocess.py --dry-run            # only build manifest, no HDS
"""
import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

import config as C
import stain_norm
from core import apc_response, bilateral_denoise, canny_binary, denoise_rgb, log_response

FEATURE_COLS = [
    "image_id", "class_name", "class_id_binary", "class_id_4way", "split",
    "nucleus_area", "nucleus_perim", "nucleus_eccentricity", "nucleus_solidity",
    "cell_area", "nc_ratio",
    "glcm_contrast", "glcm_energy", "glcm_homogeneity", "glcm_correlation",
    "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_images(raw_dir):
    items = []
    for class_dir in sorted(Path(raw_dir).iterdir()):
        if not class_dir.is_dir() or class_dir.name not in C.CLASS_MAP:
            continue
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, class_dir.name))
    return items


def stratified_split(items, seed):
    """Stratified train/val/test split. Guarantees >=1 train per class."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for p, c in items:
        by_class[c].append(p)
    split = {}
    for c, paths in by_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        n = len(paths)
        if n == 1:
            split[paths[0]] = "train"
            continue
        n_train = max(1, int(round(n * C.TRAIN_RATIO)))
        n_val = int(round(n * C.VAL_RATIO))
        # keep at least 1 in test when there are >=3 samples
        n_test = n - n_train - n_val
        if n >= 3 and n_test < 1:
            n_val = max(0, n_val - 1)
        for i, p in enumerate(paths):
            if i < n_train:
                split[p] = "train"
            elif i < n_train + n_val:
                split[p] = "val"
            else:
                split[p] = "test"
    return split


def _clean(binary, close_radius, min_area):
    binary = closing(binary, disk(close_radius))
    # skimage 0.26+ renamed min_size -> max_size AND flipped semantics: max_size=N drops <= N
    return remove_small_objects(binary, max_size=min_area)


def segment_cells(gray_float, close_radius, min_area):
    """Cells stain dark on light background — threshold INVERTED grayscale.
    Returns all cell components (multi-cell images are common in this dataset)."""
    u8 = (gray_float * 255).astype(np.uint8)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _clean(binary.astype(bool), close_radius, min_area)


def segment_nuclei(gray_float, cell_mask, close_radius, min_area):
    """Nuclei are the DARKEST pixels within cells. Otsu inside the cell mask only."""
    if not cell_mask.any():
        return np.zeros_like(cell_mask)
    u8 = (gray_float * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8[cell_mask], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nucleus = (u8 < thr) & cell_mask
    return _clean(nucleus, close_radius, min_area)


def morphology_features(mask):
    """Aggregate over all connected components (multi-cell images)."""
    if not mask.any():
        return (float('nan'),) * 4
    props = regionprops(label(mask.astype(np.uint8)))
    if not props:
        return (float('nan'),) * 4
    return (
        float(sum(r.area for r in props)),
        float(sum(r.perimeter for r in props)),
        float(np.mean([r.eccentricity for r in props])),
        float(np.mean([r.solidity for r in props])),
    )


def glcm_features(gray_u8, mask=None):
    img = gray_u8.copy()
    if mask is not None:
        img[~mask] = 0
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    return tuple(float(graycoprops(glcm, k)[0, 0])
                 for k in ("contrast", "energy", "homogeneity", "correlation"))


def save_preview(path, rgb, apc, logr, canny, nucleus_mask, cell_mask, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
    axes[0].imshow(rgb);                axes[0].set_title("denoised RGB")
    axes[1].imshow(apc, cmap='inferno'); axes[1].set_title(f"APC σ={C.APC_SIGMA}")
    axes[2].imshow(logr, cmap='inferno');axes[2].set_title(f"LOG σ={C.LOG_SIGMA}")
    axes[3].imshow(canny, cmap='gray');  axes[3].set_title("Canny")
    axes[4].imshow(nucleus_mask, cmap='gray'); axes[4].set_title("nucleus mask")
    axes[5].imshow(cell_mask, cmap='gray');    axes[5].set_title("cell mask")
    for a in axes:
        a.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80, bbox_inches='tight')
    plt.close(fig)


def process_one(args):
    orig_path, class_name, split, ref_mean, ref_std = args
    try:
        image_id = orig_path.stem
        bgr = cv2.imread(str(orig_path))
        if bgr is None:
            raise ValueError(f"cv2.imread returned None for {orig_path}")

        # [1] stain normalize
        bgr = stain_norm.normalize(bgr, ref_mean, ref_std)

        # [2] denoise (see config.DENOISE_METHOD)
        if C.DENOISE_METHOD == 'bilateral':
            denoised_bgr = bilateral_denoise(
                bgr, C.BILATERAL_D, C.BILATERAL_SIGMA_COLOR, C.BILATERAL_SIGMA_SPACE)
        elif C.DENOISE_METHOD == 'hds':
            denoised_bgr = denoise_rgb(
                bgr, C.HDS_ITERATIONS, C.HDS_LAMBDA, C.HDS_DT,
                C.HDS_G_PARAM, C.HDS_H_PARAM, C.HDS_EPSILON)
        elif C.DENOISE_METHOD == 'none':
            denoised_bgr = bgr
        else:
            raise ValueError(f"unknown DENOISE_METHOD={C.DENOISE_METHOD!r}")
        rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        gray_f = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

        # [3] edge / curvature layers
        apc  = apc_response(gray_f, C.APC_SIGMA, C.APC_ALPHA).astype(np.float32)
        logr = log_response(gray_f, C.LOG_SIGMA).astype(np.float32)
        cny  = canny_binary(gray_f, C.CANNY_LOW, C.CANNY_HIGH)

        # [4] segment masks from grayscale intensity (biology: cells stain dark)
        #     APC/LOG stay as CNN input channels; segmentation uses intensity for reliability
        cell_mask    = segment_cells(gray_f, C.CELL_CLOSE_RADIUS, C.MIN_MASK_AREA)
        nucleus_mask = segment_nuclei(gray_f, cell_mask, C.NUCLEUS_CLOSE_RADIUS, C.MIN_MASK_AREA)

        # [5] cache tensor
        cache_path = C.CACHE_DIR / split / f"{image_id}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, rgb=rgb, apc=apc, log=logr,
            canny=cny, nucleus_mask=nucleus_mask, cell_mask=cell_mask,
        )

        # [6] QC preview
        if C.SAVE_PREVIEW:
            prev = C.PREVIEW_DIR / split / f"{image_id}.png"
            prev.parent.mkdir(parents=True, exist_ok=True)
            save_preview(prev, rgb, apc, logr, cny, nucleus_mask, cell_mask,
                         f"{class_name} / {image_id} / {split}")

        # [7] morphology + texture features
        n_area, n_perim, n_ecc, n_sol = morphology_features(nucleus_mask)
        c_area, _, _, _ = morphology_features(cell_mask)
        nc_ratio = (n_area / c_area) if (c_area and c_area > 0) else float('nan')
        gray_u8 = (gray_f * 255).astype(np.uint8)
        glcm_c, glcm_e, glcm_h, glcm_corr = glcm_features(
            gray_u8, mask=cell_mask if cell_mask.any() else None
        )
        bin_id, four_id = C.CLASS_MAP[class_name]

        return {
            "image_id": image_id, "class_name": class_name,
            "class_id_binary": bin_id, "class_id_4way": four_id, "split": split,
            "nucleus_area": n_area, "nucleus_perim": n_perim,
            "nucleus_eccentricity": n_ecc, "nucleus_solidity": n_sol,
            "cell_area": c_area, "nc_ratio": nc_ratio,
            "glcm_contrast": glcm_c, "glcm_energy": glcm_e,
            "glcm_homogeneity": glcm_h, "glcm_correlation": glcm_corr,
            "mean_R": float(rgb[..., 0].mean()),
            "mean_G": float(rgb[..., 1].mean()),
            "mean_B": float(rgb[..., 2].mean()),
            "std_R":  float(rgb[..., 0].std()),
            "std_G":  float(rgb[..., 1].std()),
            "std_B":  float(rgb[..., 2].std()),
        }
    except Exception as e:
        print(f"[FAIL] {orig_path}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def write_manifest(items, split_map):
    C.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.MANIFEST_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["image_id", "orig_path", "class_name",
                    "class_id_binary", "class_id_4way", "split"])
        for path, cname in items:
            bin_id, four_id = C.CLASS_MAP[cname]
            w.writerow([path.stem, str(path), cname, bin_id, four_id, split_map[path]])


def run(raw_dir=None, dry_run=False, n_workers=None):
    raw_dir = Path(raw_dir or C.RAW_DIR)
    items = discover_images(raw_dir)
    if not items:
        print(f"No images found under {raw_dir}. "
              f"Expected subfolders named: {list(C.CLASS_MAP)}", file=sys.stderr)
        return 1

    split_map = stratified_split(items, C.SPLIT_SEED)
    counts = defaultdict(lambda: defaultdict(int))
    for p, c in items:
        counts[c][split_map[p]] += 1
    print(f"Found {len(items)} images across {len(counts)} classes:")
    for c, s in counts.items():
        print(f"  {c:8s}  train={s['train']:4d}  val={s['val']:4d}  test={s['test']:4d}")

    write_manifest(items, split_map)
    print(f"Manifest -> {C.MANIFEST_PATH}")
    if dry_run:
        return 0

    # Fit Reinhard reference. Prefer train Benign (no leakage), fall back progressively.
    ref_pool = [p for p, c in items if c == "Benign" and split_map[p] == "train"]
    if not ref_pool:
        print("WARN: no train Benign; falling back to all Benign")
        ref_pool = [p for p, c in items if c == "Benign"]
    if not ref_pool:
        print("WARN: no Benign at all; falling back to any train image")
        ref_pool = [p for p, _ in items if split_map[p] == "train"]
    if not ref_pool:
        ref_pool = [p for p, _ in items]
    rng = random.Random(C.SPLIT_SEED)
    ref_sample = rng.sample(ref_pool, min(C.STAIN_REF_SAMPLE_SIZE, len(ref_pool)))
    print(f"Fitting stain reference from {len(ref_sample)} image(s)...")
    ref_mean, ref_std = stain_norm.fit_reference(ref_sample)

    tasks = [(p, c, split_map[p], ref_mean, ref_std) for p, c in items]
    workers = n_workers if n_workers is not None else (C.N_WORKERS or os.cpu_count() or 1)
    print(f"Processing {len(tasks)} images with {workers} worker(s)...")

    C.FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(C.FEATURES_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        w.writeheader()
        if workers == 1:
            for t in tasks:
                r = process_one(t)
                if r:
                    w.writerow(r); n_ok += 1
        else:
            with mp.Pool(workers) as pool:
                for r in pool.imap_unordered(process_one, tasks, chunksize=1):
                    if r:
                        w.writerow(r); n_ok += 1

    print(f"\nDone: {n_ok}/{len(tasks)} succeeded")
    print(f"  Cache:    {C.CACHE_DIR}")
    print(f"  Features: {C.FEATURES_PATH}")
    print(f"  Manifest: {C.MANIFEST_PATH}")
    if C.SAVE_PREVIEW:
        print(f"  Previews: {C.PREVIEW_DIR}")
    return 0 if n_ok == len(tasks) else 3


def smoke_test():
    """Synthesize 1 cell-like image per class, run full pipeline, assert outputs."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw"
        for cname in C.CLASS_MAP:
            (raw / cname).mkdir(parents=True)
            img = np.full((128, 128, 3), (180, 130, 200), dtype=np.uint8)  # purple bg
            cv2.circle(img, (64, 64), 30, (100, 30, 110), -1)              # cell
            cv2.circle(img, (64, 64), 12, (50, 10, 60), -1)                # nucleus
            noise = np.random.RandomState(0).randint(-15, 15, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(raw / cname / f"{cname}_smoke.jpg"), img)

        # redirect config to tmp, shorten HDS for speed
        C.CACHE_DIR     = tmp / "cache"
        C.MANIFEST_PATH = tmp / "manifest.csv"
        C.FEATURES_PATH = tmp / "features.csv"
        C.PREVIEW_DIR   = tmp / "previews"
        C.HDS_ITERATIONS = 20

        rc = run(raw_dir=raw, n_workers=1)
        assert rc == 0, f"run() returned {rc}"
        assert C.MANIFEST_PATH.exists()
        assert C.FEATURES_PATH.exists()
        npz_files = list(C.CACHE_DIR.rglob("*.npz"))
        assert len(npz_files) == len(C.CLASS_MAP), \
            f"expected {len(C.CLASS_MAP)} .npz, got {len(npz_files)}"
        z = np.load(npz_files[0])
        for k in ("rgb", "apc", "log", "canny", "nucleus_mask", "cell_mask"):
            assert k in z.files, f"missing key {k}"
        assert z["rgb"].shape == (128, 128, 3) and z["rgb"].dtype == np.uint8
        assert z["apc"].dtype == np.float32
        assert z["nucleus_mask"].dtype == np.bool_
        with open(C.FEATURES_PATH) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 + len(C.CLASS_MAP), \
            f"features.csv rows: {len(rows)-1} (want {len(C.CLASS_MAP)})"
        print("\nSMOKE TEST PASSED")


def main():
    ap = argparse.ArgumentParser(
        description="Leukemia preprocessing: raw JPEGs -> cached .npz + features.csv")
    ap.add_argument("--input",   type=Path, default=C.RAW_DIR,
                    help=f"raw dir with per-class subfolders (default: {C.RAW_DIR})")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: all cores)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only scan + write manifest, skip HDS processing")
    ap.add_argument("--smoke",   action="store_true",
                    help="run synthetic self-test and exit")
    args = ap.parse_args()
    if args.smoke:
        smoke_test()
        return 0
    return run(raw_dir=args.input, dry_run=args.dry_run, n_workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
