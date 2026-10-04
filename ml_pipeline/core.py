"""HDS denoising + edge/curvature responses. Extracted from hds.py + app.py,
GUI-free, no module-level state — safe for multiprocessing workers."""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import laplace


def _norm01(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-10)


def _grad_and_coeff(u, h, g, eps):
    padded = np.pad(u, 1, mode='edge')
    gW = u - padded[1:-1, :-2]
    gN = u - padded[:-2, 1:-1]
    gS = padded[2:, 1:-1] - u
    gE = padded[1:-1, 2:] - u
    def c(gk):
        return g * (1.0 / (1.0 + (gk ** 2) / (h ** 2)) + 1.0 / (np.abs(gk) + eps))
    return gW, gN, gS, gE, c(gN), c(gS), c(gW), c(gE)


def hybrid_denoise(noisy, iterations, lam, dt, g, h, eps):
    """HDS denoising on one float [0,1] channel. Returns denoised in [0,1]."""
    u = noisy.astype(np.float64).copy()
    f = noisy.astype(np.float64).copy()
    for _ in range(iterations):
        gW, gN, gS, gE, cN, cS, cW, cE = _grad_and_coeff(u, h, g, eps)
        div = cN * gN + cS * gS + cW * gW + cE * gE
        log_prior = lam * ((u / (f + eps)) - 1.0) * (1.0 / (u + eps))
        u_next = u + dt * (div - log_prior)
        if np.isnan(u_next).any() or np.isinf(u_next).any():
            return np.clip(u, 0.0, 1.0)
        u = np.clip(u_next, 0.0, 1.0)
    return u


def denoise_rgb(rgb_uint8, iterations, lam, dt, g, h, eps):
    """Run HDS on each channel independently. Returns uint8, same layout as input."""
    out = np.zeros_like(rgb_uint8, dtype=np.float64)
    for c in range(3):
        out[..., c] = hybrid_denoise(
            rgb_uint8[..., c].astype(np.float64) / 255.0,
            iterations, lam, dt, g, h, eps,
        )
    return (out * 255.0).astype(np.uint8)


def apc_response(gray_float, sigma, alpha):
    """Adaptive Principal Curvature — ridge/nucleus-rim detector."""
    hess = hessian_matrix(gray_float, sigma=sigma, order='rc',
                          use_gaussian_derivatives=False)
    l1, l2 = hessian_matrix_eigvals(hess)
    return _norm01(alpha * np.abs(l2) + (1 - alpha) * np.abs(l1))


def log_response(gray_float, sigma):
    """Laplacian of Gaussian — blob detector (whole-cell prior at larger sigma)."""
    return _norm01(np.abs(laplace(gaussian_filter(gray_float, sigma=sigma))))


def canny_binary(gray_float, low, high):
    u8 = (gray_float * 255).astype(np.uint8)
    return cv2.Canny(u8, int(low * 255), int(high * 255)) > 0


def bilateral_denoise(bgr_uint8, d=9, sigma_color=75, sigma_space=75):
    """Edge-preserving smoothing. Suits additive Gaussian noise in stained
    microscopy — unlike HDS, which is a multiplicative-speckle model and
    diverges on this data."""
    return cv2.bilateralFilter(bgr_uint8, d, sigma_color, sigma_space)
