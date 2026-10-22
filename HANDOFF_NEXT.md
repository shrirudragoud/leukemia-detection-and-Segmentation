# Hand-off for the next agent (written 2026-10-04 ~21:40 UTC)

**Budget: only ~$2 of credits are left. Be frugal: no sub-agents, no polling loops, no re-running heavy jobs, short commands. Do only what is listed here.**
Branch: `claude/sweet-hawking-lrormj` (all work is pushed). Repo: shrirudragoud/leukemia-detection-and-segmentation.

## 1. What this is
A journal-style paper (Bates "How to Write a Paper in Scientific Journal Style and Format" guide) about 4-class ALL smear classification with a frozen hematology foundation model (DinoBloom-S),
with a leakage-aware evaluation, a shortcut audit and stage-by-stage preprocessing figures. The paper is generated: `paper/build_paper.py` (+ `paper/ext_*.py`) reads result files in `docs/results/` and writes
`paper/content.json`; `paper/render_docx.js` makes `paper/paper.docx`; LibreOffice makes `paper/paper.pdf`. **Every number comes from a result file. Never type a number into the text.**
Current build: ~130 pages, in `paper/paper.docx` and `paper/paper.pdf`.

## 2. What is DONE
- Preprocessing, tests (189 pass), CPU results (headline balanced accuracy 0.968, 95% CI 0.955-0.978), shortcut audit, leakage/purge/gap analyses, stage figures and stage-contribution analysis, XAI attention figure, references verified (60/62, see `paper/REFERENCE_CHECKLIST.md`), contents + lists + citation appendices, abbreviations spelled out.
- Audit of the manuscript against code/results done (`/tmp` copy is gone; fixes already applied).

## 3. What is RUNNING / PENDING (Kaggle GPU, account `shridharrudragoud`)
| Kernel | URL | Runs | Started | Expected end (UTC) |
|---|---|---|---|---|
| leukemia-gpu-a | https://www.kaggle.com/code/shridharrudragoud/leukemia-gpu-a | g01, g02, g08 | 20:00 | 22:30-23:15 |
| leukemia-gpu-b | https://www.kaggle.com/code/shridharrudragoud/leukemia-gpu-b | g04, g05, g06, g03 | 20:03 | 00:00-01:00 |
| leukemia-gpu-c (NOT started) | - | g07 (DinoBloom-B, ~3 h) | - | start only when a slot is free (Kaggle allows 2 GPU sessions at once) |

Code dataset: https://www.kaggle.com/datasets/shridharrudragoud/leuk-code (private; latest version also returns model weights). Templates for the kernels are in `scripts/kaggle/` (`run.py` + `kernel-metadata.json`).
Setup inside a kernel: copies code, downloads DinoBloom weights (Zenodo, internet on), preprocesses all images (~10-15 min), runs `scripts/gpu_queue.sh`, writes `results_for_paper.tar.gz`, `results_for_paper/` and `logs/` to `/kaggle/working`.
Cost estimate: ~45-60 min per run on a T4/P100 (not measured; the first real GPU run). **Kaggle shows logs only in the web UI (Logs tab) while running; `kaggle kernels logs` is empty until the job ends.**

## 4. Credentials (NOT in the repo)
Set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY` (the user pasted a key into the chat earlier; they will revoke it). Never write the key to a file or commit it. If auth fails, ask the user to add the two variables in the environment settings and open a new session.
Network: Kaggle API works. Crossref/arXiv/Zenodo/PMLR/NeurIPS/JMLR work; dblp/Semantic Scholar/OpenAlex block bots.

## 5. How to check and fetch results (cheap commands)
```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...
kaggle kernels status shridharrudragoud/leukemia-gpu-a      # RUNNING / COMPLETE / ERROR
kaggle kernels status shridharrudragoud/leukemia-gpu-b
kaggle kernels logs   shridharrudragoud/leukemia-gpu-a | tail -40   # works after the job ends
kaggle kernels files  shridharrudragoud/leukemia-gpu-a             # list outputs BEFORE downloading (a/b left big caches in /kaggle/working/repo; do not download everything)
mkdir -p /tmp/ko_a && kaggle kernels output shridharrudragoud/leukemia-gpu-a -p /tmp/ko_a --file-pattern "results_for_paper"   # if --file-pattern is unsupported, download the single tar: results_for_paper.tar.gz
```
If a kernel ended in ERROR: read `logs`, fix the bug in code (not the numbers), bump the code dataset (`kaggle datasets version -p <dir> -m msg --dir-mode zip`, dir = copy of src/configs/scripts/tests/pyproject/README), re-push the kernel (`kaggle kernels push -p scripts/kaggle/kernel_a`). The queue resumes finished folds.
The first real CUDA run was never tested before; typical risks: AMP dtype, DataLoader workers, out-of-memory for g07 (batch 8 already).

## 6. Where the results go
Unpack so that this layout exists in the repo root (it is git-ignored? check `git check-ignore results_for_paper`; do NOT commit big files):
```
results_for_paper/<run_name>/cv_summary.json        # run_name = g01_lora_dinobloom_s, g02_lora_dinobloom_s_gray, g03_lora_dinobloom_s_hybrid, g04_full_resnet50,
results_for_paper/<run_name>/fold{0..4}_predictions.csv   #   g05_full_efficientnet_b0, g06_lora_dinov2_s, g07_lora_dinobloom_b, g08_lora_dinobloom_s_raw_background
results_for_paper/<run_name>/fold{0..4}_train_log.jsonl   # (a/b: logs may be missing; only summaries + predictions are guaranteed)
```
Commit only `results_for_paper/*/cv_summary.json` and `*_predictions.csv`/`*_train_log.jsonl` (small). Weights (`*_model.pt`) stay out of git.

## 7. Finish the paper (cheap, deterministic)
```bash
python scripts/finalize_paper.py --build --check      # makes figures (real FT-CURVES/FT-BARS/FT-CM replace the SAMPLE images), fills the fine-tuning table, rebuilds docx+pdf, lists unfinished parts
```
Unfinished parts are marked `[[GPU...]]`, `SAMPLE IMAGE`, `FILL IN`, `[[PENDING`. After the table fills, write (from the numbers only) in `paper/ext_results.py` (section "Fine-tuning and encoder comparison") and `paper/ext_discussion.py` (section "Fine-tuning"):
1. g01 vs frozen reference (0.968): difference, paired-bootstrap CI (use `leukemia_ml.eval.stats.paired_bootstrap` on the predictions), Holm p.
2. colour effect (g01 vs g02); 3. feature branch (g03 vs g01); 4. encoder ordering (g01, g04, g05, g06, g07); 5. shortcut control (g08 vs g01).
Be honest if fine-tuning does NOT beat the frozen head. Keep the guide's rules: no duplicated data in a table and a figure, do not restate every value.
Then remove the two `[[GPU: ...]]` paragraphs (they disappear automatically when all 8 result files exist; with fewer, rewrite them).
**If the GPU results never arrive:** set `"gpu_placeholders": "cut"` in `paper/meta.json` and rebuild (removes the table, 3 figures and GPU paragraphs; the paper then states fine-tuning was not performed).

## 8. Things only the USER can finish
- `paper/meta.json`: `authors`, `affiliation`, `correspondence`, `acknowledgments` (currently `FILL IN`).
- Spot-check two references that could not be verified online: Holm 1979 (Scand J Stat 6(2):65-70) and Bradski 2000 (Dr. Dobb's J 25(11):120-125); Roberts 2021 lists 10 of 54 authors + "and others".
- Decide GPU placeholders keep/cut (section 7). Open question from the guide audit: the guide says not to put the same data in both a table and a figure and not to restate table values; the paper has a few duplicated pairs (denoising, ablation, purge/gap, confusion matrix) - the user was asked whether to fix; no answer yet.
- Figure placement: figures are grouped at the end of Results (guide allows it); the user was asked whether to move them next to first mention.
- Revoke the Kaggle key afterwards.

## 9. Known limits (already stated in the paper; do not "fix" by adding claims)
In-dataset accuracy is optimistic (class and session are confounded; background colour alone gives 0.823 balanced accuracy); no external validation; DinoBloom may have seen this dataset; Grad-CAM gave uniform maps and is excluded; CPU-only encoder comparison failed (no model-hub downloads) so encoder comparison depends on the GPU runs; nucleus segmentation fails for ~92% of cells.

## 10. Cheap sanity checks before the final push
`python -m pytest -q -x tests/test_ml_data.py` (fast), `python scripts/finalize_paper.py --check`, `git status` clean, push to `claude/sweet-hawking-lrormj`. Do not open a PR unless the user asks.
