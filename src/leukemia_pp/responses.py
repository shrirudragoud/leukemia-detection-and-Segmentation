"""Edge / curvature response layers used as extra network input channels.

All responses are *scale-normalised*, multiplied by a fixed gain and clipped to [0, 1]
rather than min-max scaled per image. Per-image min-max would stretch noise in an empty field to look
as strong as a real cell edge, and makes layers incomparable across images.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from .stain import otsu


def gray_float(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def apc_response(gray: np.ndarray, sigma: float, alpha: float, gain: float = 1.0) -> np.ndarray:
    """Adaptive Principal Curvature: alpha*|l2| + (1-alpha)*|l1| of the Hessian, sigma^2-
    normalised. `mode='reflect'` avoids the bright frame the zero-padded default creates."""
    hess = hessian_matrix(gray, sigma=sigma, order="rc", mode="reflect",
                          use_gaussian_derivatives=False)
    l1, l2 = hessian_matrix_eigvals(hess)   # l1 >= l2
    resp = (alpha * np.abs(l2) + (1.0 - alpha) * np.abs(l1)) * sigma ** 2 * gain
    return np.clip(resp, 0.0, 1.0).astype(np.float32)


def log_response(gray: np.ndarray, sigma: float, gain: float = 1.0) -> np.ndarray:
    """Scale-normalised |Laplacian of Gaussian| (blob detector)."""
    resp = np.abs(ndi.gaussian_laplace(gray.astype(np.float64), sigma=sigma, mode="reflect", truncate=6.0))
    resp = resp * sigma ** 2 * gain
    return np.clip(resp, 0.0, 1.0).astype(np.float32)


def canny_edges(gray: np.ndarray, low_ratio: float) -> np.ndarray:
    """Canny with an image-adaptive high threshold (Otsu) and low = ratio * high."""
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    high = otsu(u8)
    return cv2.Canny(u8, max(1.0, low_ratio * high), max(2.0, high)) > 0
