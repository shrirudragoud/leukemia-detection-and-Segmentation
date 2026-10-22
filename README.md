# Leukemia Detection and Segmentation

An image-processing and feature-extraction project for microscopy images of white blood cells. The repository contains the current reproducible preprocessing pipeline, demo data and outputs, research documentation, and the earlier interactive GUI prototypes.

## What the project does

`leukemia_pp` turns raw peripheral-blood smear images into analysis-ready data:

| Stage | Method |
| --- | --- |
| Colour normalisation | Reinhard in float CIELAB, statistics from foreground (cell) pixels, reference fitted on the **train split only** and saved to disk |
| Denoising | Edge-preserving bilateral filter (non-local means optional) |
| Response layers | Scale-normalised APC (Hessian principal curvature), LoG and adaptive Canny, with fixed (not per-image) scaling |
| **WBC instance segmentation** | Lab **a\*** channel -> Otsu (with an absolute floor) -> hole filling -> distance-transform / h-maxima **watershed** to split touching cells |
| Nucleus segmentation | Per-cell Otsu on L\* of the cell *interior*, accepted only if nucleus and cytoplasm are genuinely separable |
| Features | Per-cell shape, colour, and masked multi-direction GLCM texture + entropy; image-level aggregates |
| QC | Sharpness, brightness, WBC area, flags such as `blurry`, `no_cells`, `stain_not_applied` |
| Data hygiene | Stratified, seeded splits that keep byte-identical duplicates (and optionally patients) together |

## Why colour-based segmentation

The v1 pipeline thresholded inverted grayscale, which merges every red blood cell with the leukocytes
(see `docs/paper.pdf`, "RBC contamination"). Leukocytes and blasts are strongly magenta/purple while RBCs
and background are pale grey-blue, so the Lab **a\*** axis separates them cleanly. On the three demo
images this finds 11, 4 and 3 cells with zero RBC false positives, and splits the touching pairs.

## Repository layout

```text
src/leukemia_pp/            Production package (v1.0)
  config.py                 Frozen, validated, serialisable configuration (hashed per run)
  io.py                     Discovery + robust image decoding (unicode paths, corrupt files)
  splits.py                 Deterministic stratified, leakage-aware splits
  stain.py                  Reinhard normalisation + saved reference
  denoise.py  responses.py  Denoising; APC / LoG / Canny layers
  segmentation.py           WBC instance + nucleus segmentation
  features.py  qc.py        Per-cell/per-image features; quality control
  pipeline.py  cli.py       Orchestration, multiprocessing, outputs; command line
  hds.py  emd.py            Stabilised HDS denoiser/edge indicator; BEEMD + pure-IMF selection
  representations.py        Optional HDS/BEEMD channels + per-cell stats of every response layer
  dataset.py                Channel presets, train-only normalisation stats, cell crops, feature groups
  preview.py  metrics.py    QC panels; Dice / IoU for when ground-truth masks exist
tests/                      94 pytest tests on synthetic fields with known geometry
configs/                    Example configs (all_representations.json = HDS + BEEMD on)
data/raw/                   Demo input (3 images)
data/processed/             Demo output of the production pipeline
ml_pipeline/                v1 implementation, frozen: the paper and docs/generate_figures.py use it
data/{cache,layers,previews,features.csv,manifest.csv,diagnose.png}   v1 outputs (paper figures)
docs/                       Technical report (v1 methodology)
APC Blood Vessal Block 3 (1)/, Lukiemiea block 2-3 combind (1)/      Original Tkinter prototypes
```

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                # 94 tests, ~30 s
```

## Run

Input layout: `<input>/<Class>/<image>.{jpg,png,...}` with class folders `Benign`, `Early`, `Pre`, `Pro`
(configurable via `class_map`). Image file names must be unique across classes.

```bash
leukemia-pp run --input data/raw --output data/processed            # all cores
leukemia-pp run --input data/raw --output out --dry-run              # scan + split + manifest only
leukemia-pp run --input data/raw --output out --config my.json       # override any config field
leukemia-pp inspect data/raw/Pro/WBC-Malignant-Pro-003.jpg --out qc.png \
    --reference data/processed/stain_reference.json                  # QC panel for one image
```

`my.json` may contain any subset of the config, e.g. `{"segmentation": {"min_cell_area": 80}}`.
Unknown keys are rejected. Exit codes: `0` ok, `2` bad input/config, `3` some images failed.

An output directory is bound to the config that created it (hash in `config.json`); running with a
different config requires a new directory or `--force`.

## Outputs (`<output>/`)

| File | Content |
| --- | --- |
| `cache/<split>/<id>.npz` | `rgb_norm`, `rgb` (denoised), `apc`, `log`, `canny`, `a_score`, `cell_labels`, `nucleus_labels` (instance ids), `cell_mask`, `nucleus_mask` |
| `cells.csv` | One row per segmented cell: shape, colour, texture, nucleus features, `touches_border` |
| `images.csv` | One row per image: QC measurements/flags and aggregated cell features (interior cells preferred) |
| `manifest.csv` | image id, relative path, class, labels, split, sha256 |
| `stain_reference.json`, `config.json` | Everything needed to reproduce / to normalise new images identically |
| `failures.csv`, `run_summary.json` | Failed images (never fatal to the run); counts and QC-flag totals |
| `previews/` | Eight-panel QC image per input |

Runs are deterministic: identical inputs and config give byte-identical CSVs regardless of worker count.
All lengths and areas are in pixels (pixel size is not recorded in this dataset).

## Optional representations (HDS, BEEMD / pure IMF) and how to decide if they help

Both are **off by default** (`hds.enabled`, `beemd.enabled`); `configs/all_representations.json` turns them
on (~0.9 s/image). Everything is computed on the stain-normalised, *un-denoised* luminance.

| Stage | What it adds | Verified so far |
| --- | --- | --- |
| **HDS** (`hds.py`) | Denoised luminance (`hds`) and a fixed-scale edge indicator (`hds_edge`) | The prototype's scheme diverges (PSNR 22 -> 5 dB on additive noise); the stabilised rewrite (normalised diffusivities, additive fidelity term, max-principle) gains +12 dB and is bounded after 3000 iterations. Parameters were chosen on a *synthetic* benchmark. |
| **APC / LoG / Canny** | Ridge, blob and edge layers (always on) | Border artefact fixed; per-cell mean/std/rim statistics are now features. Value for classification: unproven. |
| **BEEMD** (`emd.py`) | `imfs` (K x H x W), `imf_residue`, per-cell IMF energies | Exact reconstruction; separates 4 px texture / 20 px structure / drift on synthetic data; seeded and deterministic. |
| **Pure-IMF selection** | `imf_selected`, `imf_pure` | Rule = spatial-period band [3, 48] px + minimum energy; white noise lands at 2.4 px so it is rejected. This is *my* interpretable criterion: swap `select_pure_imfs` if your paper defines another. |
| **Morphology** | Per-cell shape (area, circularity, solidity, aspect, ...) | Works. Nucleus morphology is unavailable on this data (see limitations). |

**None of these has been shown to improve a classifier.** Pretrained ResNet/ViT/EfficientNet backbones
already learn edge and texture filters, so hand-made channels frequently give no gain, can disturb transfer
learning, and cost preprocessing time; they tend to help most on small datasets, as regularisation, or in a
fusion branch. Decide with data, using `leukemia_pp.dataset` and the same patient-/slide-level splits throughout:

1. `rgb` only (ImageNet-pretrained backbone) - the baseline everything must beat;
2. `rgb+apc` / `rgb+edges` / `rgb+hds` / `rgb+imf` / `full` (`dataset.PRESETS`), one at a time;
3. RGB + tabular late fusion, adding one `FEATURE_GROUPS` group at a time (shape, colour, texture, layer stats);
4. Keep a representation only if it improves macro-F1 on validation across several seeds / folds.

Extra input channels: keep the first three channels as RGB so pretrained weights stay valid, and initialise the
additional first-conv / patch-embedding weights from the mean RGB filter scaled by ~0.1 (or use a separate
light branch with late fusion). Use `compute_channel_stats` on the **train** split only, and `crop_cell` to feed
single cells instead of whole fields.

## Known limitations (read before modelling)

* **Nucleus masks are usually unresolved on this data.** The blasts have thin cytoplasm and a nucleus filling
  almost the whole cell at 224 px; within the cell interior the nucleus/cytoplasm contrast is 2-7 L\* units, below
  the acceptance threshold (8). The pipeline therefore reports `nucleus_found = False` and leaves nucleus
  features and `nc_ratio` as NaN rather than inventing a boundary. (The v1 nucleus masks and N/C ratios were
  largely edge-rim artefacts.) Cell-level shape/colour/texture features are unaffected. A learned nucleus
  segmenter, or higher-resolution data, is the proper fix; calibrate `nucleus_min_contrast` on the full dataset.
* Thresholds (`a_floor`, `min_cell_area`, `split_h`, ...) were set on three demo images. Validate on the full
  dataset, ideally against ground-truth masks (`metrics.dice` / `metrics.iou`).
* The Reinhard reference is a single global colour model; fields with very little foreground are left
  unnormalised and flagged (`stain_not_applied`).
* If images carry no patient id, the split is image-level; supply `split.group_regex` when ids exist.

## v1 (`ml_pipeline/`)

The original scripts are kept unchanged because the paper and `docs/generate_figures.py` depend on them:
`python ml_pipeline/preprocess.py` (v1 usage and outputs under `data/cache`, `data/features.csv`, ...).
New work should use `leukemia_pp`.

## Interactive prototypes

The two folders with historical names contain standalone Tkinter applications created during development. They are preserved for reference and experimentation; the reusable implementation is in `ml_pipeline/`.

On Linux, macOS, or Windows with Python and Tkinter installed:

```bash
python "APC Blood Vessal Block 3 (1)/APC Blood Vessal Block 3/app.py"
python "Lukiemiea block 2-3 combind (1)/Lukiemiea block 2-3 combind/main_combined_app.py"
```

## Data and reproducibility

The checked-in `data/raw/` directory is a small demonstration subset. The generated outputs and report figures correspond to that local subset. For a full experiment, place the complete licensed dataset under `data/raw/` using the expected class-folder structure and rerun the pipeline.

`data/manifest.csv` records portable relative input paths. Generated `.npz` caches can be recreated at any time and are included here as reference/demo outputs.

## Research report

The report in `docs/paper.pdf` explains the pipeline design, the choice of bilateral filtering over HDS for this microscopy data, the segmentation approach, and the limitations of intensity-based cell masks in fields containing red blood cells. Its LaTeX source and figure-generation script are included for reproducibility.

## License and medical disclaimer

No license is currently declared. Add the appropriate license before redistributing the repository or dataset. This is a research and image-processing project, not a clinical diagnostic tool; outputs must not be used as medical advice or as a substitute for qualified clinical review.
