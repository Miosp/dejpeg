"""PSNR-B parity test for the ported FBCNN grayscale variant on Classic5.

Skipped unless FBCNN_TESTSETS_DIR points at a directory containing the upstream
'testsets/Classic5' folder (5 grayscale BMP images: 1.bmp .. 5.bmp).

Classic5 is a grayscale benchmark. The FBCNN paper (https://arxiv.org/abs/2109.14573,
Table 1) reports FBCNN (gray) results on Classic5 QF=30 as PSNR=33.54, SSIM=0.894,
PSNRB=32.78. This test reproduces the PSNRB number to prove the port is
behaviorally faithful to upstream.

Pipeline mirrors `main_test_fbcnn_gray.py` from the upstream repo: read grayscale,
JPEG-compress at QF=30, run FBCNN with qf=30, compute PSNR-B between restored and
original. The PSNR-B metric (blocking-effect-aware PSNR) is ported verbatim from
`utils/utils_image.py:calculate_psnrb`.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from fbcnn import GRAY, run

from conftest import skip_without_testsets, skip_without_weights


# Paper-reported Classic5 PSNRB for FBCNN (grayscale), QF=30.
# Source: FBCNN paper Table 1 (https://arxiv.org/abs/2109.14573).
PAPER_CLASSIC5_QF30_PSNRB = 32.78  # dB
PSNR_TOLERANCE_DB = 0.2


def _read_bmp_gray(path: Path) -> np.ndarray:
    """Read BMP as HxW uint8 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img


def _jpeg_compress_gray(img: np.ndarray, qf: int) -> np.ndarray:
    """JPEG-compress an HxW uint8 grayscale image at the given quality factor."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _compute_bef(img: np.ndarray) -> float:
    """Blocking Effect Factor on a single-channel float64 image. Mirrors upstream."""
    block = 8
    height, width = img.shape[:2]

    h_b = list(range(block - 1, width - 1, block))
    h_bc = [i for i in range(width - 1) if i not in h_b]
    v_b = list(range(block - 1, height - 1, block))
    v_bc = [j for j in range(height - 1) if j not in v_b]

    d_b = 0.0
    d_bc = 0.0
    for i in h_b:
        diff = img[:, i] - img[:, i + 1]
        d_b += float(np.sum(diff ** 2))
    for i in h_bc:
        diff = img[:, i] - img[:, i + 1]
        d_bc += float(np.sum(diff ** 2))
    for j in v_b:
        diff = img[j, :] - img[j + 1, :]
        d_b += float(np.sum(diff ** 2))
    for j in v_bc:
        diff = img[j, :] - img[j + 1, :]
        d_bc += float(np.sum(diff ** 2))

    n_hb = height * (width // block - 1)
    n_hbc = height * (width - 1) - n_hb
    n_vb = width * (height // block - 1)
    n_vbc = width * (height - 1) - n_vb
    d_b /= (n_hb + n_vb)
    d_bc /= (n_hbc + n_vbc)
    eta = math.log2(block) / math.log2(min(height, width)) if d_b > d_bc else 0.0
    return eta * (d_b - d_bc)


def _psnrb_gray(ground_truth: np.ndarray, restored: np.ndarray) -> float:
    """PSNR-B between two HxW uint8 grayscale images. Mirrors upstream calculate_psnrb."""
    if ground_truth.shape != restored.shape:
        raise ValueError("shape mismatch")
    gt = ground_truth.astype(np.float64)
    rs = restored.astype(np.float64)
    bef = _compute_bef(rs)
    mse = float(np.mean((gt - rs) ** 2))
    mse_b = mse + bef
    if mse_b == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse_b))


def test_psnrb_classic5_qf30(testsets_dir: Path | None, weights_dir: Path | None):
    skip_without_testsets(testsets_dir)
    skip_without_weights(weights_dir)

    classic5 = testsets_dir / "Classic5"  # type: ignore[arg-type]
    if not classic5.exists():
        pytest.skip(f"Classic5 not found under {testsets_dir}")

    image_paths = sorted(classic5.glob("*.bmp"))
    assert len(image_paths) == 5, f"Expected 5 BMPs in {classic5}, found {len(image_paths)}"

    psnrbs = []
    for p in image_paths:
        original = _read_bmp_gray(p)
        compressed = _jpeg_compress_gray(original, qf=30)
        # Port's run() expects HxWxC; grayscale variant uses C=1.
        restored_c, _ = run(compressed[..., None], qf=30, variant=GRAY, weights_dir=weights_dir)  # type: ignore[arg-type]
        restored = restored_c[..., 0]
        psnrbs.append(_psnrb_gray(original, restored))

    mean_psnrb = float(np.mean(psnrbs))
    delta = abs(mean_psnrb - PAPER_CLASSIC5_QF30_PSNRB)
    per_image = ", ".join(f"{p.name}={x:.2f}" for p, x in zip(image_paths, psnrbs))
    assert delta <= PSNR_TOLERANCE_DB, (
        f"PSNR-B drift: got {mean_psnrb:.3f} dB, expected {PAPER_CLASSIC5_QF30_PSNRB} dB "
        f"(delta {delta:.3f} > tolerance {PSNR_TOLERANCE_DB}). Per-image: {per_image}"
    )
