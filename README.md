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
  preview.py  metrics.py    QC panels; Dice / IoU for when ground-truth masks exist
tests/                      63 pytest tests on synthetic fields with known geometry
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
pytest                                                # 63 tests, ~15 s
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
