import numpy as np

from leukemia_pp.responses import apc_response, canny_edges, log_response


def test_uniform_image_has_no_response_anywhere_including_border():
    flat = np.full((64, 64), 0.6, np.float32)
    assert apc_response(flat, 2.5, 0.6, gain=12).max() < 1e-4   # v1 had a bright border frame
    assert log_response(flat, 4.0, gain=5).max() < 1e-4
    assert not canny_edges(flat, 0.5).any()


def test_blob_responds_and_range_is_fixed():
    img = np.full((96, 96), 0.9, np.float32)
    yy, xx = np.mgrid[:96, :96]
    img[(yy - 48) ** 2 + (xx - 48) ** 2 < 15 ** 2] = 0.4
    for resp in (apc_response(img, 2.5, 0.6, 12), log_response(img, 4.0, 5)):
        assert resp.dtype == np.float32
        assert 0.0 <= resp.min() and resp.max() <= 1.0
        # |LoG| is zero exactly on a step edge and peaks ~sigma either side: use an edge band
        assert resp[40:56, 24:42].max() > 0.2
        assert resp[:20, :20].max() < 0.02                  # flat background stays quiet
    assert canny_edges(img, 0.5).any()


def test_response_does_not_rescale_with_image_content():
    """Empty-ish noisy field must not be stretched to look like a strong edge (no min-max)."""
    rng = np.random.RandomState(0)
    noise = (0.9 + rng.randn(96, 96) * 0.002).astype(np.float32)
    assert apc_response(noise, 2.5, 0.6, 12).max() < 0.1
