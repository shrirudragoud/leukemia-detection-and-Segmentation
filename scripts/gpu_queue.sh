#!/usr/bin/env bash
# Runs the experiment queue in priority order. A failure in one job does NOT stop the others;
# finished folds are skipped on re-run and interrupted folds resume, so it is safe to re-launch.
# Usage:  bash scripts/gpu_queue.sh [g01 g02 ...]      (default: all, in this order)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results_for_paper
ALL=(g01_lora_dinobloom_s g02_lora_dinobloom_s_gray g04_full_resnet50 g03_lora_dinobloom_s_hybrid \
     g06_lora_dinov2_s g08_lora_dinobloom_s_raw_background g05_full_efficientnet_b0 g07_lora_dinobloom_b)
if [ "$#" -gt 0 ]; then
  SEL=(); for p in "$@"; do for n in "${ALL[@]}"; do [[ "$n" == "$p"* ]] && SEL+=("$n"); done; done
else SEL=("${ALL[@]}"); fi

for name in "${SEL[@]}"; do
  echo "================ $name  $(date +%H:%M:%S)"
  leukemia-ml preflight --config "configs/ml/gpu/$name.json" > "logs/$name.preflight.json" 2>&1 || true
  leukemia-ml cv --config "configs/ml/gpu/$name.json" --out runs > "logs/$name.log" 2>&1
  rc=$?
  d=$(ls -d runs/${name}_* 2>/dev/null | head -1)
  if [ $rc -eq 0 ] && [ -f "$d/cv_summary.json" ]; then
    mkdir -p "results_for_paper/$name"
    cp "$d/cv_summary.json" "$d/cv_summary.md" "results_for_paper/$name/"
    cp "configs/ml/gpu/$name.json" "results_for_paper/$name/config.json"
    for f in "$d"/fold*/model.pt; do cp "$f" "results_for_paper/$name/$(basename $(dirname $f))_model.pt"; done   # best-epoch trainable weights (adapter/head; base encoder is reloaded from its own weights)
    for f in "$d"/fold*/train_log.jsonl; do cp "$f" "results_for_paper/$name/$(basename $(dirname $f))_train_log.jsonl"; done
    for f in "$d"/fold*/predictions.csv; do cp "$f" "results_for_paper/$name/$(basename $(dirname $f))_predictions.csv"; done
    echo "DONE $name"
  else
    echo "FAILED $name (rc=$rc) - see logs/$name.log ; re-run:  bash scripts/gpu_queue.sh ${name%%_*}"
  fi
done
tar czf results_for_paper.tar.gz results_for_paper logs 2>/dev/null && echo "packed results_for_paper.tar.gz"
echo "finished queue at $(date +%H:%M:%S)"; ls results_for_paper
