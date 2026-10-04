# HANDOFF: GPU run + paper finalisation

Branch `claude/sweet-hawking-lrormj` of `shrirudragoud/leukemia-detection-and-Segmentation`.
Everything below is tested on CPU; the CUDA path has NOT been run yet (no GPU was available), so
`preflight` is the first thing to run on the GPU machine.

## 0. Rules (please keep)
1. **Never type a result into the paper by hand.** Paper numbers come from `paper/results_macros.tex`
   and `paper/generated/*`, which `scripts/finalize_paper.py` builds from result JSON files.
2. Do **not** change a config mid-run (the config hash binds outputs to the exact settings; a changed
   config starts a new run). Do not edit result files. Do not tune on the test fold.
3. If a job fails, fix the *bug*, not the number; re-run the queue (finished folds are skipped,
   interrupted folds resume from `last.pt`).

## 1. Start the GPU run (Kaggle or Lightning AI)
Kaggle: new notebook, GPU on, **Internet on**, "Add data" -> `mehradaria/leukemia`.
Lightning AI: new studio with a GPU; upload/clone the repo; the dataset can be downloaded from Kaggle.

```bash
git clone -b claude/sweet-hawking-lrormj https://github.com/shrirudragoud/leukemia-detection-and-Segmentation.git
cd leukemia-detection-and-Segmentation
# Kaggle:   bash scripts/gpu_bootstrap.sh /kaggle/input/leukemia/Original
# Elsewhere: bash scripts/gpu_bootstrap.sh /path/to/Original     (folder containing Benign Early Pre Pro)
bash scripts/gpu_bootstrap.sh <path-to-Original>      # install, weights, preprocessing (10-15 min), crops, tests, preflight
nohup bash scripts/gpu_queue.sh > queue.log 2>&1 &    # all experiments, priority order
tail -f queue.log
```
Priority (the queue already runs in this order; stop whenever time runs out):
1. `g01_lora_dinobloom_s`  (headline fine-tuned model)
2. `g02_lora_dinobloom_s_gray` (no stain colour: shortcut control)
3. `g04_full_resnet50` (CNN baseline)
4. `g03_lora_dinobloom_s_hybrid` (curvature/morphology features fused)
5. `g06_lora_dinov2_s` (generic self-supervised baseline)
6. `g08_lora_dinobloom_s_raw_background` (what fine-tuning does when shortcuts are left in)
7. `g05_full_efficientnet_b0`, 8. `g07_lora_dinobloom_b`
Run a single one: `bash scripts/gpu_queue.sh g01`.  Time projection is in `logs/<name>.preflight.json`
(`est_total_hours_all_folds`; it is an upper bound).

## 2. What to bring back
`results_for_paper.tar.gz` (made by the queue; also written after each job) containing, per experiment,
`cv_summary.json`, `cv_summary.md`, `config.json`, per-fold `predictions.csv`, plus `logs/`.
Partial results are fine: the paper adapts to whichever experiments finished.

## 3. Finalise the paper (with me or alone)
```bash
tar xzf results_for_paper.tar.gz -C .          # puts results_for_paper/ in the repo root
python scripts/finalize_paper.py --report      # which experiments are present / missing
python scripts/finalize_paper.py               # builds macros, tables, sentences, figures
python scripts/finalize_paper.py --check       # FAILS if any placeholder / unresolved block remains
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
To have me do it: give me `results_for_paper.tar.gz` and say **"finalize"**. I will (a) run the three
commands above, (b) read the generated sentences and tables for truthfulness (e.g. if fine-tuning did
*not* beat the frozen model, the generated text says so), (c) delete every `GPU-DEPENDENT` block whose
experiment is missing, and (d) confirm `--check` passes.

## 4. If a job fails (playbook)
| Symptom | Fix |
| --- | --- |
| CUDA out of memory | lower `train.batch_size` (16 -> 8) or `data.max_cells` (16 -> 12), or set `model.grad_checkpointing: true`; keep the change in a *new* config copy |
| `DinoBloom checkpoint mismatch` | re-download the `.pth` (curl was cut off); size S=265 MB, B=528 MB |
| Kaggle cannot download weights | turn Internet on in notebook settings (needs phone verification) |
| `empty train/val/test split` | embargo too large for a subset run; use the default config |
| Session killed | just re-run `bash scripts/gpu_queue.sh`; it resumes |
| Very slow data loading | `train.num_workers: 4` (default); HED colour jitter is CPU-bound, set `aug.colour: hsv` only in a new config and say so in the paper |
| NaN loss with fp16 | `train.amp_dtype: bf16` if supported, else lower `lr_lora` |

## 5. What is already measured and final (CPU, in `docs/results/`)
Shortcut audit; stabilised-HDS benchmark; leakage analysis (random blocks vs contiguous folds + embargo,
distance curve); frozen DinoBloom-S attention-MIL 5-fold CV (balanced accuracy 0.968, 95% CI
0.955-0.978); honest ablations of image variants and of encoders. The paper's core stands on these.

## 6. Known limitations to keep in the text (do not remove)
Classes were imaged in separate sessions so within-dataset accuracy cannot separate biology from batch
style; no patient identifiers (patient-proxy folds from capture order); segmentation not validated
against ground truth; nucleus segmentation unresolved on this data; DinoBloom pretraining overlap with
this dataset not verified; no external validation yet.
