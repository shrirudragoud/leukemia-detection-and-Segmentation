"""Keep only the informative panels of fig_xai.png (most-attended cell, attention per cell).
The Grad-CAM column came out as an almost uniform map for the frozen encoder, and the parameter-randomisation correlations were all
zero because of it, so Grad-CAM is NOT used as evidence (see paper, Results). The raw output stays in paper/figures/fig_xai.png."""
from PIL import Image

im = Image.open("paper/figures/fig_xai.png").convert("RGB")
w, h = im.size
a = im.crop((int(0.016 * w), 0, int(0.31 * w), h))
c = im.crop((int(0.66 * w), 0, w, h))
out = Image.new("RGB", (a.width + c.width + 40, h), "white")
out.paste(a, (0, 0)); out.paste(c, (a.width + 40, 0))
out.thumbnail((1400, 1800))
out.save("paper/figures/fig_attention.png")
print(out.size)
