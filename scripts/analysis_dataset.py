"""Dataset and preprocessing facts for the paper -> docs/results/dataset_stats.json"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RUN = Path("data/full_run_v2")
OUT = Path("docs/results/dataset_stats.json")


def rows(name):
    with open(RUN / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


man, img, cells = rows("manifest.csv"), rows("images.csv"), rows("cells.csv")
summary = json.loads((RUN / "run_summary.json").read_text())
cfg = json.loads((RUN / "config.json").read_text())

classes = ["Benign", "Early", "Pre", "Pro"]
per_class = {}
for c in classes:
    ii = [r for r in img if r["class_name"] == c]
    n = np.array([int(r["n_cells"]) for r in ii])
    frac = np.array([float(r["wbc_area_frac"]) for r in ii])
    cc = [r for r in cells if r["class_name"] == c]
    area = np.array([float(r["cell_area"]) for r in cc])
    per_class[c] = {
        "images": len(ii), "cells": int(n.sum()),
        "cells_per_image_mean": float(n.mean()), "cells_per_image_sd": float(n.std(ddof=1)),
        "cells_per_image_median": float(np.median(n)), "cells_per_image_max": int(n.max()),
        "wbc_area_frac_median": float(np.median(frac)),
        "cell_area_median": float(np.median(area)), "cell_area_p5": float(np.percentile(area, 5)),
        "cell_area_p95": float(np.percentile(area, 95)),
        "nucleus_found_frac": float(np.mean([r["nucleus_found"] == "True" for r in cc])),
        "border_cell_frac": float(np.mean([r["touches_border"] == "True" for r in cc])),
    }
all_area = np.array([float(r["cell_area"]) for r in cells])
sha = Counter(r["sha256"] for r in man)
dups = {h: n for h, n in sha.items() if n > 1}
split_counts = defaultdict(lambda: defaultdict(int))
for r in man:
    split_counts[r["class_name"]][r["split"]] += 1
flags = Counter(f for r in img for f in r["qc_flags"].split(";") if f)

out = {
    "n_images": len(man), "n_cells": len(cells), "run_seconds": summary["seconds"],
    "n_failed": summary["n_failed"], "config_hash": summary["config_hash"],
    "per_class": per_class,
    "cell_area_median": float(np.median(all_area)), "cell_area_p5": float(np.percentile(all_area, 5)),
    "cell_area_p95": float(np.percentile(all_area, 95)),
    "nucleus_found_frac": float(np.mean([r["nucleus_found"] == "True" for r in cells])),
    "duplicate_groups": len(dups), "images_in_duplicate_groups": int(sum(dups.values())),
    "qc_flag_counts": dict(flags),
    "image_height": sorted({int(r["height"]) for r in img}), "image_width": sorted({int(r["width"]) for r in img}),
    "sharpness_median": float(np.median([float(r["sharpness"]) for r in img])),
    "pipeline_config": cfg["config"],
}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("n_images", "n_cells", "duplicate_groups", "nucleus_found_frac")}, indent=2))
