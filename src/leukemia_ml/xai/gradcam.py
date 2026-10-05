"""Grad-CAM for the cell encoder (CNN feature maps or ViT patch tokens) + faithfulness checks.

Explains the image-level logit of a one-cell bag w.r.t. the encoder's last spatial features.
Saliency maps can look plausible while being unrelated to the model's weights (Adebayo et al.
2018), so this module ships the checks with the method: parameter randomisation
(`randomisation_sanity`) and deletion faithfulness (`deletion_curve`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.model import LeukemiaModel


def _single_bag(x: torch.Tensor, feats: torch.Tensor | None):
    n = torch.ones(1, dtype=torch.long)
    return (n.new_zeros(1), n.new_zeros(1), n,
            feats if feats is not None else torch.zeros(1, 0))


def _target_layer(model: LeukemiaModel) -> torch.nn.Module:
    enc = model.encoder
    if enc.blocks is not None:
        return enc.blocks[-1]
    for name in ("layer4", "features", "conv_head", "stages"):
        layer = getattr(enc.backbone, name, None)
        if layer is not None:
            return layer
    raise ValueError("cannot find a spatial feature layer for Grad-CAM")


def grad_cam(model: LeukemiaModel, x: torch.Tensor, class_idx: int | None = None,
             feats: torch.Tensor | None = None) -> tuple[np.ndarray, int]:
    """Returns (cam in [0,1] at input resolution, class index explained). x: (1,3,H,W)."""
    model.eval()
    layer = _target_layer(model)
    acts: dict[str, torch.Tensor] = {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            h = model.encoder(x)
            if model.branch is not None:
                f = feats if feats is not None else torch.zeros(1, model.branch.n_features)
                h = torch.cat([h, model.branch(f.to(h.dtype))], dim=-1)
            bag, pos, n, _ = _single_bag(x, feats)
            from ..models.model import pad_bags
            padded, mask = pad_bags(h, bag, pos, 1, 1)
            logits, _ = model.head(padded, mask)
            cls = int(logits.argmax(1).item()) if class_idx is None else class_idx
            a = acts["a"]
            grad = torch.autograd.grad(logits[0, cls], a)[0]
    finally:
        h1.remove()
    if a.dim() == 3:                                    # ViT tokens (1, 1+g*g, D)
        g = int(round((a.shape[1] - 1) ** 0.5))
        a, grad = a[:, 1:], grad[:, 1:]
        weights = grad.mean(dim=1, keepdim=True)         # (1,1,D)
        cam = F.relu((a * weights).sum(-1)).reshape(1, 1, g, g)
    else:                                               # CNN (1, C, h, w)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((a * weights).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-12)
    return cam.detach().cpu().numpy(), cls


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(spearmanr(a.ravel(), b.ravel())[0])


def randomisation_sanity(model: LeukemiaModel, x: torch.Tensor, seed: int = 0) -> list[dict]:
    """Cascading parameter randomisation (Adebayo et al.): re-initialise layers from the head
    down to the first encoder block and track rank correlation with the original map. A
    faithful explanation method must decorrelate; a high correlation means the map does not
    depend on what the model learned."""
    import copy
    base_cam, cls = grad_cam(model, x)
    clone = copy.deepcopy(model)
    torch.manual_seed(seed)
    stages: list[tuple[str, list[torch.nn.Module]]] = [("head", [clone.head])]
    if clone.branch is not None:
        stages.append(("branch", [clone.branch]))
    enc = clone.encoder
    blocks = list(enc.blocks) if enc.blocks is not None else [enc.backbone]
    for k, blk in enumerate(reversed(blocks)):
        stages.append((f"encoder_block_-{k + 1}", [blk]))
    out = []
    for name, mods in stages:
        for m in mods:
            for sub in m.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
        cam, _ = grad_cam(clone, x, class_idx=cls)
        out.append({"randomised_through": name, "spearman_with_original": _spearman(base_cam, cam)})
    return out


@torch.no_grad()
def deletion_curve(model: LeukemiaModel, x: torch.Tensor, cam: np.ndarray, cls: int,
                   fracs=(0.0, 0.1, 0.2, 0.4, 0.6), fill: float = 232 / 255.0,
                   feats: torch.Tensor | None = None, seed: int = 0) -> dict[str, list[float]]:
    """Class probability as the most-salient (CAM) vs random pixels are replaced by neutral grey.
    A faithful map makes the probability fall faster than random deletion."""
    from ..models.model import pad_bags
    model.eval()
    flat = cam.ravel()
    order = np.argsort(-flat)
    rng = np.random.default_rng(seed)
    rand_order = rng.permutation(len(flat))
    bag, pos, n, f = _single_bag(x, feats)

    def prob(img):
        h = model.encoder(img)
        if model.branch is not None:
            h = torch.cat([h, model.branch(f.to(h.dtype))], dim=-1)
        padded, mask = pad_bags(h, bag, pos, 1, 1)
        return float(torch.softmax(model.head(padded, mask)[0], 1)[0, cls])

    out = {"fractions": list(fracs), "cam": [], "random": []}
    for fr in fracs:
        k = int(fr * len(flat))
        for key, ordr in (("cam", order), ("random", rand_order)):
            m = np.zeros(len(flat), bool)
            m[ordr[:k]] = True
            mask = torch.from_numpy(m.reshape(cam.shape))[None, None].expand_as(x)
            out[key].append(prob(torch.where(mask, torch.full_like(x, fill), x)))
    return out
