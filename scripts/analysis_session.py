"""Do consecutive image numbers share acquisition conditions? -> docs/results/session_structure.json

For each class: median distance of the background colour (Lab) between images with adjacent
numbers versus randomly paired images of the same class."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/dataset/Original")
OUT = Path("docs/results/session_structure.json")


def background(path):
    bgr = cv2.imread(str(path))
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    bright = lab[lab[:, 0] >= np.percentile(lab[:, 0], 70)]
    return np.median(bright, axis=0)


res = {}
for cls in ("Benign", "Early", "Pre", "Pro"):
    files = sorted((ROOT / cls).glob("*.jpg"), key=lambda p: int(re.findall(r"(\d+)\.jpg", p.name)[0]))
    f = np.array([background(p) for p in files])
    adj = np.linalg.norm(np.diff(f, axis=0), axis=1)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(f), 4000), rng.integers(0, len(f), 4000)
    rand = np.linalg.norm(f[i] - f[j], axis=1)
    res[cls] = {"n": len(f), "median_adjacent": float(np.median(adj)), "median_random": float(np.median(rand)),
                "ratio": float(np.median(adj) / np.median(rand))}
    print(cls, res[cls])
OUT.write_text(json.dumps(res, indent=2))
