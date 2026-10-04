"""Dump each layer of every cached .npz as a standalone PNG for viewing."""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/cache"
OUT = ROOT / "data/layers"

def to_u8_colour(arr, cmap=cv2.COLORMAP_INFERNO):
    u8 = (arr * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)

for npz_path in sorted(CACHE.rglob("*.npz")):
    z = np.load(npz_path)
    stem = npz_path.stem
    dst = OUT / npz_path.parent.name
    dst.mkdir(parents=True, exist_ok=True)
    # RGB (stored as RGB, cv2 wants BGR)
    cv2.imwrite(str(dst / f"{stem}_1_rgb.png"),
                cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2BGR))
    # Response maps as colour-mapped PNGs
    cv2.imwrite(str(dst / f"{stem}_2_apc.png"), to_u8_colour(z["apc"]))
    cv2.imwrite(str(dst / f"{stem}_3_log.png"), to_u8_colour(z["log"]))
    # Binary masks as plain white-on-black
    cv2.imwrite(str(dst / f"{stem}_4_canny.png"),
                (z["canny"].astype(np.uint8) * 255))
    cv2.imwrite(str(dst / f"{stem}_5_nucleus.png"),
                (z["nucleus_mask"].astype(np.uint8) * 255))
    cv2.imwrite(str(dst / f"{stem}_6_cell.png"),
                (z["cell_mask"].astype(np.uint8) * 255))
    print(f"exported {stem}")

print(f"\ndone -> {OUT.resolve()}")
