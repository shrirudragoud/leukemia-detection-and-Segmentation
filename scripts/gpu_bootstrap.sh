#!/usr/bin/env bash
# One-time setup on a GPU machine (Kaggle notebook / Lightning AI studio / any Linux box).
# Usage:  bash scripts/gpu_bootstrap.sh /path/to/dataset/Original
#   where the argument is the folder containing Benign/ Early/ Pre/ Pro/ (Kaggle: attach the dataset
#   "mehradaria/leukemia" to the notebook; it is usually /kaggle/input/leukemia/Original).
set -euo pipefail
RAW="${1:?give the path to the dataset Original folder (contains Benign Early Pre Pro)}"
cd "$(dirname "$0")/.."

echo "== 1. install =="
pip install -q -e ".[ml]"
python - <<'PY'
import torch, timm
print("torch", torch.__version__, "| timm", timm.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
PY

echo "== 2. dataset link =="
mkdir -p data/dataset data/weights
[ -e data/dataset/Original ] || ln -s "$RAW" data/dataset/Original
for c in Benign Early Pre Pro; do test -d "data/dataset/Original/$c" || { echo "missing class folder $c"; exit 1; }; done
echo "images: $(find -L data/dataset/Original -name '*.jpg' | wc -l) (expect 3256)"

echo "== 3. DinoBloom weights (Zenodo 10908163, CC-BY-4.0) =="
for v in S B; do
  [ -s "data/weights/DinoBloom-$v.pth" ] || curl -L --retry 3 -o "data/weights/DinoBloom-$v.pth" \
     "https://zenodo.org/records/10908163/files/DinoBloom-$v.pth?download=1"
done
ls -la data/weights

echo "== 4. preprocessing (deterministic; ~10-15 min on 4 cores) =="
if [ ! -f data/full_run_v2/cells.csv ]; then
  leukemia-pp run --input data/dataset/Original --output data/full_run_v2 \
                  --config configs/full_dataset.json --workers "$(nproc)"
fi
python - <<'PY'
import json
s = json.load(open("data/full_run_v2/run_summary.json"))
print(s)
assert s["n_ok"] == 3256 and s["n_failed"] == 0, "preprocessing did not process all 3256 images"
assert s["n_cells"] > 36000, "unexpected number of cells"
PY

echo "== 5. crop caches =="
for cfg in g01_lora_dinobloom_s g02_lora_dinobloom_s_gray g08_lora_dinobloom_s_raw_background; do
  leukemia-ml crops --config "configs/ml/gpu/$cfg.json" --workers "$(nproc)"
done

echo "== 6. quick tests + preflight =="
python -m pytest -q -x tests/test_ml_models.py tests/test_ml_eval.py
leukemia-ml preflight --config configs/ml/gpu/g01_lora_dinobloom_s.json
echo "bootstrap OK"
