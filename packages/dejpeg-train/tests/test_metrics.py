"""Phase 0 Task 0.2 -- eval metrics tests (ported FBCNN metrics sanity)."""
from __future__ import annotations

import numpy as np
import pytest

from dejpeg_train.eval.metrics import compute_all, psnr, psnr_b, ssim


def _gradient(size=64):
    idx = np.arange(size, dtype=np.uint8) * (255 // (size - 1))
    g = (idx[:, None] + idx[None, :]) // 2  # (H, W)
    return np.stack([g, g, g], axis=-1).astype(np.uint8)  # HWC RGB


def _blocky(size=64, block=8):
    img = _gradient(size)
    out = img.copy()
    for br in range(0, size, block):
        for bc in range(0, size, block):
            out[br : br + block, bc : bc + block] = img[br : br + block, bc : bc + block].mean(axis=(0, 1))
    return out.astype(np.uint8)


def test_self_match_is_perfect():
    img = _gradient()
    assert psnr(img, img) == float("inf")
    # PSNR-B adds a blocking-effect factor computed from the image itself, so even
    # a self-match is finite when the image has block-grid structure; it is large.
    assert np.isfinite(psnr_b(img, img)) and psnr_b(img, img) > 100.0
    assert abs(ssim(img, img) - 1.0) < 1e-6


def test_psnr_decreases_with_noise():
    rng = np.random.RandomState(0)
    gt = _gradient()
    noisy = np.clip(gt.astype(int) + rng.randint(-20, 21, gt.shape), 0, 255).astype(np.uint8)
    assert psnr(gt, noisy) < psnr(gt, gt)
    assert 0 <= ssim(gt, noisy) < 1.0


def test_psnrb_penalizes_blocking_more_than_psnr():
    """PSNR-B adds a blocking-effect factor; for a blocky image it must drop below PSNR."""
    gt = _gradient()
    comp = _blocky()
    # both metrics finite for a non-identical pair
    p = psnr(gt, comp)
    pb = psnr_b(gt, comp)
    assert np.isfinite(p) and np.isfinite(pb)
    assert pb < p, f"PSNR-B ({pb:.2f}) should be < PSNR ({p:.2f}) on a blocky image"


def test_compute_all_keys_without_lpips():
    img = _gradient()
    out = compute_all(img, img)
    assert set(out) == {"psnr", "psnr_b", "ssim"}
    assert out["psnr"] == float("inf")
    assert abs(out["ssim"] - 1.0) < 1e-6


def test_compute_all_includes_lpips_when_provided():
    try:
        from dejpeg_train.eval.metrics import LPIPSAlex

        metric = LPIPSAlex(device="cpu")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"LPIPS-Alex weights unavailable: {e}")
    img = _gradient()
    out = compute_all(img, img, metric)
    assert "lpips_alex" in out
    assert out["lpips_alex"] < 1e-3  # self-distance ~ 0
