"""Diagnostic: run one image through each pipeline stage in isolation
and dump a side-by-side PNG so we can SEE where the damage happens."""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as C
import stain_norm
from core import denoise_rgb, apc_response, log_response, canny_binary

ROOT = Path(__file__).resolve().parent.parent
img_path = Path(sys.argv[1] if len(sys.argv) > 1
                else ROOT / "data/raw/Early/WBC-Malignant-Early-012.jpg")
out = ROOT / "data/diagnose.png"

raw_bgr = cv2.imread(str(img_path))
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
print(f"raw:   mean={raw_rgb.mean(axis=(0,1))}  std={raw_rgb.std(axis=(0,1))}")

# stain norm using this image as its own reference — should be near-identity
ref_mean, ref_std = stain_norm.fit_reference([img_path])
stained_bgr = stain_norm.normalize(raw_bgr, ref_mean, ref_std)
stained_rgb = cv2.cvtColor(stained_bgr, cv2.COLOR_BGR2RGB)
print(f"stain: mean={stained_rgb.mean(axis=(0,1))}  std={stained_rgb.std(axis=(0,1))}")

# HDS on raw (no stain) at short/full iterations
hds20_bgr  = denoise_rgb(raw_bgr, 20,  C.HDS_LAMBDA, C.HDS_DT, C.HDS_G_PARAM, C.HDS_H_PARAM, C.HDS_EPSILON)
hds200_bgr = denoise_rgb(raw_bgr, 200, C.HDS_LAMBDA, C.HDS_DT, C.HDS_G_PARAM, C.HDS_H_PARAM, C.HDS_EPSILON)
hds20_rgb  = cv2.cvtColor(hds20_bgr,  cv2.COLOR_BGR2RGB)
hds200_rgb = cv2.cvtColor(hds200_bgr, cv2.COLOR_BGR2RGB)
print(f"hds20:  mean={hds20_rgb.mean(axis=(0,1))}  std={hds20_rgb.std(axis=(0,1))}")
print(f"hds200: mean={hds200_rgb.mean(axis=(0,1))}  std={hds200_rgb.std(axis=(0,1))}")

# APC on grayscale of each
def apc_of(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    return apc_response(g, C.APC_SIGMA, C.APC_ALPHA)

fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for a in ax.flat: a.axis('off')
ax[0,0].imshow(raw_rgb);     ax[0,0].set_title("RAW")
ax[0,1].imshow(stained_rgb); ax[0,1].set_title("stain-norm only")
ax[0,2].imshow(hds20_rgb);   ax[0,2].set_title("HDS 20 iters (no stain)")
ax[0,3].imshow(hds200_rgb);  ax[0,3].set_title("HDS 200 iters (no stain)")
ax[1,0].imshow(apc_of(raw_rgb),    cmap='inferno'); ax[1,0].set_title("APC(raw)")
ax[1,1].imshow(apc_of(stained_rgb),cmap='inferno'); ax[1,1].set_title("APC(stain)")
ax[1,2].imshow(apc_of(hds20_rgb),  cmap='inferno'); ax[1,2].set_title("APC(hds20)")
ax[1,3].imshow(apc_of(hds200_rgb), cmap='inferno'); ax[1,3].set_title("APC(hds200)")
fig.suptitle(img_path.name, fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=90, bbox_inches='tight')
print(f"\nsaved -> {out.resolve()}")
