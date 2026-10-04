from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR       = ROOT / "data" / "raw"
CACHE_DIR     = ROOT / "data" / "cache"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
FEATURES_PATH = ROOT / "data" / "features.csv"
PREVIEW_DIR   = ROOT / "data" / "previews"

# Denoising method: 'bilateral' | 'hds' | 'none'
# HDS is the Block 2 paper method but it diverges on non-speckle (natural microscopy)
# noise — use only for methodology comparisons, not production preprocessing.
DENOISE_METHOD = 'bilateral'

# Bilateral filter (default). d=diameter, sigmas control colour/spatial smoothing.
BILATERAL_D           = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# HDS denoising (from Lukiemiea/hds.py — paper defaults, kept for opt-in)
HDS_ITERATIONS = 200
HDS_LAMBDA     = 0.05
HDS_DT         = 0.15
HDS_G_PARAM    = 1.3
HDS_H_PARAM    = 1.0
HDS_EPSILON    = 1e-8

# Edge params — sigmas tuned for WBC nuclei (~30-80 px), not 1-px edges
APC_SIGMA   = 2.5
APC_ALPHA   = 0.6
LOG_SIGMA   = 4.0
CANNY_LOW   = 0.1
CANNY_HIGH  = 0.25

# Segmentation
NUCLEUS_CLOSE_RADIUS = 3
CELL_CLOSE_RADIUS    = 5
MIN_MASK_AREA        = 100  # px

# Stratified split (per class)
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SPLIT_SEED  = 42

# Stain normalization reference (Reinhard) — averaged over N train Benign images
STAIN_REF_SAMPLE_SIZE = 20

# class_name -> (binary_id, fourway_id)   binary: 0=Benign, 1=Malignant
CLASS_MAP = {
    "Benign": (0, 0),
    "Early":  (1, 1),
    "Pre":    (1, 2),
    "Pro":    (1, 3),
}

N_WORKERS    = None   # None = os.cpu_count()
SAVE_PREVIEW = True
