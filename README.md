# Leukemia Detection and Segmentation

An image-processing and feature-extraction project for microscopy images of white blood cells. The repository contains the current reproducible preprocessing pipeline, demo data and outputs, research documentation, and the earlier interactive GUI prototypes.

## What the project does

The main pipeline converts raw microscopy images into:

1. Stain-normalized and denoised RGB images.
2. Adaptive Principal Curvature (APC), Laplacian of Gaussian (LOG), and Canny response layers.
3. Whole-cell and nucleus segmentation masks.
4. Six-channel compressed `.npz` tensors for downstream neural networks.
5. Morphology, texture, and color features in `data/features.csv` for classical ML models.

The default production path uses Reinhard stain normalization followed by an edge-preserving bilateral filter. The older Hybrid Diffusion-Steered (HDS) method remains available for comparison and is documented in the paper.

## Repository layout

```text
ml_pipeline/                         Current reusable preprocessing pipeline
  config.py                          All paths and algorithm parameters
  core.py                            Denoising and image-response functions
  stain_norm.py                      Reinhard LAB stain normalization
  preprocess.py                      Raw images -> tensors/features/manifest
  diagnose.py                        Diagnostic visualization for one image
  export_layers.py                   Export cached tensor layers as PNG files

data/
  raw/                               Input microscopy images by class
  cache/                             Generated compressed tensor outputs
  previews/                          Generated six-panel quality-control images
  layers/                            PNG exports of cached layers
  manifest.csv                       Image metadata and train/val/test split
  features.csv                      Extracted morphology/texture/color features
  diagnose.png                      Example diagnostic output

docs/
  paper.tex                          Technical report source
  paper.pdf                          Rendered technical report
  generate_figures.py                Rebuilds report figures
  figures/                           Figures used by the report
  figures.7z                         Archived figure bundle

APC Blood Vessal Block 3 (1)/       Earlier standalone APC edge-detection GUI
Lukiemiea block 2-3 combind (1)/    Earlier HDS + edge-detection GUI
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the current pipeline

Run the synthetic self-test:

```bash
python ml_pipeline/preprocess.py --smoke
```

Process the images under `data/raw/`:

```bash
python ml_pipeline/preprocess.py
```

Useful options:

```bash
python ml_pipeline/preprocess.py --dry-run
python ml_pipeline/preprocess.py --workers 1
python ml_pipeline/preprocess.py --input /path/to/raw-images
```

The input directory should contain class subdirectories named `Benign`, `Early`, `Pre`, and `Pro`. The pipeline writes its manifest, cached tensors, previews, and feature CSV beneath `data/`.

Export each cached tensor layer as a standalone PNG:

```bash
cd ml_pipeline
python export_layers.py
cd ..
```

Generate a diagnostic image for one raw sample:

```bash
cd ml_pipeline
python diagnose.py ../data/raw/Early/WBC-Malignant-Early-012.jpg
cd ..
```

Rebuild the technical-report figures:

```bash
python docs/generate_figures.py
```

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
