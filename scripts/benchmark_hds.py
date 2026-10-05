"""Stabilised vs legacy HDS on additive noise -> docs/results/hds_benchmark.json"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from leukemia_pp.config import HDSConfig
from leukemia_pp.hds import hds_denoise

OUT = Path("docs/results/hds_benchmark.json")
spec = importlib.util.spec_from_file_location("legacy_core", "ml_pipeline/core.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
psnr = lambda a, b: float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))  # noqa: E731


def scene(sigma, seed):
    clean = np.full((128, 128), 0.2, np.float32)
    cv2.circle(clean, (64, 64), 30, 0.8, -1)
    cv2.rectangle(clean, (10, 10), (40, 40), 0.5, -1)
    noisy = clean + np.random.RandomState(seed).randn(*clean.shape).astype(np.float32) * sigma
    return clean, noisy.astype(np.float32)


rows = []
for sigma, seed in [(0.05, 0), (0.08, 1), (0.12, 2)]:
    clean, noisy = scene(sigma, seed)
    u, _, n_it = hds_denoise(noisy, HDSConfig())
    old = legacy.hybrid_denoise(noisy, 200, 0.05, 0.15, 1.3, 1.0, 1e-8)
    bil = cv2.bilateralFilter(noisy, 9, 0.2, 5)
    gau = cv2.GaussianBlur(noisy, (0, 0), 2)
    rows.append({"sigma": sigma, "noisy": psnr(noisy, clean), "legacy_hds": psnr(old, clean),
                 "stabilised_hds": psnr(u, clean), "bilateral": psnr(bil, clean),
                 "gaussian": psnr(gau, clean), "stabilised_iterations": int(n_it)})
# boundedness after exhaustive iteration
_, noisy = scene(0.3, 0)
u, e, n = hds_denoise(noisy, HDSConfig(iterations=3000, tol=0.0))
bounded = {"iterations": int(n), "finite": bool(np.isfinite(u).all()),
           "within_input_range": bool(u.min() >= noisy.min() - 1e-6 and u.max() <= noisy.max() + 1e-6)}
# texture preservation (chromatin-like pattern inside a disc)
rng = np.random.RandomState(5)
tex = cv2.GaussianBlur((rng.randn(128, 128) * 0.04).astype(np.float32), (0, 0), 1.2) * 2.5
c = np.full((128, 128), 0.3, np.float32)
cv2.circle(c, (64, 64), 40, 0.7, -1)
n_img = (c + tex * (c > 0.5) + rng.randn(128, 128) * 0.03).astype(np.float32)
core = cv2.erode((c > 0.5).astype(np.uint8), np.ones((9, 9))) > 0


def tcorr(u):
    return float(np.corrcoef((u - cv2.GaussianBlur(u, (0, 0), 6))[core], tex[core])[0, 1])


texture = {"noisy": tcorr(n_img), "stabilised_hds": tcorr(hds_denoise(n_img, HDSConfig())[0]),
           "bilateral": tcorr(cv2.bilateralFilter(n_img, 9, 0.2, 5))}
out = {"rows": rows, "boundedness": bounded, "texture_correlation": texture,
       "mean_gain_db": {k: float(np.mean([r[k] - r["noisy"] for r in rows]))
                        for k in ("legacy_hds", "stabilised_hds", "bilateral", "gaussian")},
       "hds_config": HDSConfig().__dict__}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out["mean_gain_db"], indent=2), out["boundedness"], texture)
