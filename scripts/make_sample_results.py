"""Write SYNTHETIC results in exactly the format of results_for_paper/ (cv_summary.json, fold*_predictions.csv, fold*_train_log.jsonl).
They exist only to test the figure scripts end to end and to draw the SAMPLE placeholder figures. The numbers are random and mean nothing."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/sample_results")
NAMES = ["g01_lora_dinobloom_s", "g02_lora_dinobloom_s_gray", "g03_lora_dinobloom_s_hybrid", "g04_full_resnet50",
         "g05_full_efficientnet_b0", "g06_lora_dinov2_s", "g07_lora_dinobloom_b", "g08_lora_dinobloom_s_raw_background"]
rng = np.random.default_rng(123)
sizes = [504, 985, 963, 804]
for gi, name in enumerate(NAMES):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    skill = 0.93 + 0.05 * rng.random()
    y_all, p_all, per_fold = [], [], []
    for k in range(5):
        y = np.concatenate([np.full(s // 5, c) for c, s in enumerate(sizes)])
        logit = rng.normal(0, 1, (len(y), 4)) + 4.0 * skill * np.eye(4)[y] * (1 + 0.3 * rng.random((len(y), 1)))
        p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        pred = p.argmax(1)
        cm = np.zeros((4, 4), int); np.add.at(cm, (y, pred), 1)
        rec = np.diag(cm) / cm.sum(1)
        per_fold.append({"fold": k, "test": {"n": len(y), "balanced_accuracy": float(rec.mean()), "macro_f1": float(rec.mean()), "auroc_ovr": 0.99,
                                             "ece": 0.1, "per_class_recall": rec.tolist(), "confusion": cm.tolist(), "accuracy": float(rec.mean())},
                         "epochs": 12, "seconds": 100.0, "temperature": 0.5, "split": {"n_train_images": 1500, "n_val_images": 650, "n_test_images": len(y)}})
        y_all.append(y); p_all.append(p)
        with open(d / f"fold{k}_predictions.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image_id", "group", "label", "pred", "p_0", "p_1", "p_2", "p_3", "n_cells", "top_attention"])
            for i in range(len(y)):
                w.writerow([f"S-{k}-{i}", f"g{i // 37}", int(y[i]), int(pred[i])] + [f"{v:.5f}" for v in p[i]] + [8, 0.3])
        with open(d / f"fold{k}_train_log.jsonl", "w") as fh:
            for e in range(12):
                fh.write(json.dumps({"epoch": e, "train_loss": float(1.0 * np.exp(-0.25 * e) + 0.3 + 0.02 * rng.random()),
                                     "val_macro_f1": float(0.8 + 0.15 * (1 - np.exp(-0.4 * e)) * skill + 0.01 * rng.random()),
                                     "val_bal_acc": float(0.8 + 0.15 * (1 - np.exp(-0.4 * e)) * skill + 0.01 * rng.random()),
                                     "lr": [float(2e-4 * (0.5 * (1 + np.cos(np.pi * e / 12))))], "train_seconds": 10.0}) + "\n")
    y = np.concatenate(y_all); P = np.concatenate(p_all); pr = P.argmax(1)
    cm = np.zeros((4, 4), int); np.add.at(cm, (y, pr), 1); rec = np.diag(cm) / cm.sum(1)
    ba = float(rec.mean())
    summ = {"name": name, "config_hash": "SAMPLE", "folds": [0, 1, 2, 3, 4], "n_images": int(len(y)), "n_groups": 90,
            "pooled": {"n": int(len(y)), "balanced_accuracy": ba, "macro_f1": ba, "auroc_ovr": 0.99, "ece": 0.01, "per_class_recall": rec.tolist(), "confusion": cm.tolist(),
                       "sensitivity": 0.99, "specificity": float(rec[0]), "accuracy": ba},
            "pooled_ci": {"balanced_accuracy": {"point": ba, "lo": ba - 0.01, "hi": ba + 0.01}, "macro_f1": {"point": ba, "lo": ba - 0.01, "hi": ba + 0.01}},
            "per_fold": per_fold, "SYNTHETIC": True}
    (d / "cv_summary.json").write_text(json.dumps(summ, indent=1))
(OUT / "README.txt").write_text("SYNTHETIC test data in the format of results_for_paper/. Random numbers. Never cite.\n")
print("wrote", OUT)
