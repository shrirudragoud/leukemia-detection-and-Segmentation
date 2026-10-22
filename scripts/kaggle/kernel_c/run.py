"""Kaggle GPU job: bootstrap + fine-tuning queue. Outputs: /kaggle/working/results_for_paper.tar.gz and logs."""
import glob, os, shutil, subprocess, sys, time

RUNS = os.environ.get("LEUK_RUNS", "g07").split()
t0 = time.time()
src = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True))
raw = next(p for p in glob.glob("/kaggle/input/**/Original", recursive=True) if os.path.isdir(os.path.join(p, "Benign")))
print("code:", src, "| images:", raw, flush=True)
repo = "/kaggle/working/repo"
shutil.copytree(src, repo, dirs_exist_ok=True)
os.chdir(repo)


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash")
    if check and r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


sh("nvidia-smi || true", check=False)
sh(f"bash scripts/gpu_bootstrap.sh {raw}")
print(f"bootstrap done after {(time.time() - t0) / 60:.1f} min", flush=True)
sh(f"bash scripts/gpu_queue.sh {' '.join(RUNS)}", check=False)
for f in ("results_for_paper.tar.gz",):
    if os.path.exists(f):
        shutil.copy(f, "/kaggle/working/")
if os.path.isdir("results_for_paper"):
    shutil.copytree("results_for_paper", "/kaggle/working/results_for_paper", dirs_exist_ok=True)
for d in ("logs",):
    if os.path.isdir(d):
        shutil.copytree(d, "/kaggle/working/logs", dirs_exist_ok=True)
# keep the output small: drop the preprocessing caches and crop caches, keep results + logs only
for sub in ("data", "runs"):
    shutil.rmtree(os.path.join(repo, sub), ignore_errors=True)
print(f"all done after {(time.time() - t0) / 60:.1f} min", flush=True)
