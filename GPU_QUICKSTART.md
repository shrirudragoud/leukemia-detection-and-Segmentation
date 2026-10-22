# GPU quickstart (for the agent/person running the GPU step)

**Goal:** run the fine-tuning experiments and bring back `results_for_paper.tar.gz`. Nothing else.
Full details and the failure playbook are in `HANDOFF.md`.

1. Machine: Kaggle notebook (GPU on, Internet on, add dataset `mehradaria/leukemia`) or Lightning AI GPU studio.
2. Run:
   ```bash
   git clone -b claude/sweet-hawking-lrormj https://github.com/shrirudragoud/leukemia-detection-and-Segmentation.git
   cd leukemia-detection-and-Segmentation
   bash scripts/gpu_bootstrap.sh /kaggle/input/leukemia/Original     # or the path to the folder with Benign/Early/Pre/Pro
   nohup bash scripts/gpu_queue.sh > queue.log 2>&1 &
   ```
3. Watch `queue.log`. Time projections: `logs/<name>.preflight.json`.
4. If something fails: fix the bug (not the numbers), then re-run `bash scripts/gpu_queue.sh` (it resumes).
5. Bring back: `results_for_paper.tar.gz` (partial results are fine).

Do not: edit configs mid-run, edit result files, tune on the test fold, or type numbers into the paper.
The CUDA path was never run on a GPU before; `preflight` inside the bootstrap is the first real check.
