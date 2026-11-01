"""Noise-pattern counterweight losses (2026-10-15 SUPPLEMENT §7.3).

The shipped model emits a fixed fine grain with spectral peaks at
(±1/4, ±1/2) and (±1/2, ±1/4) cycles/px; LPIPS-VGG (~94% of the training
gradient) prefers that pattern over its absence. These terms counteract it.

All terms must run in fp32, OUTSIDE bf16 autocast (D6 / supplement §7.3).
Patch sizes must stay multiples of 4 so 1/4 and 1/2 cycles/px land on FFT
bins (256 px: bins 64 and 128).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FLAT_STD_UNIT = 2.0 / 255  # flatness threshold in [0,1] units (7x7 std)
SSIM_C1, SSIM_C2 = 0.01 ** 2, 0.03 ** 2

PEAK_FREQS = ((0.25, 0.5), (0.5, 0.25), (-0.25, 0.5), (0.5, -0.25))  # (fy, fx)


def ssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """1 - SSIM, 11x11 Gaussian window, sigma 1.5, per channel, valid padding.

    Weight 0.5 helped as a fine-tune; weight 1.0 from scratch was -5.65 dB
    (phase 0.7.4) -- keep the from-scratch weight at 0.02-0.2.
    """
    g = torch.exp(-(torch.arange(11.0, device=pred.device) - 5) ** 2 / (2 * 1.5 ** 2))
    k = (g[:, None] * g[None, :] / g.sum() ** 2).view(1, 1, 11, 11).repeat(pred.shape[1], 1, 1, 1)
    f = lambda t: F.conv2d(t, k, groups=pred.shape[1])
    mp, mg = f(pred), f(gt)
    vp, vg, cov = f(pred * pred) - mp ** 2, f(gt * gt) - mg ** 2, f(pred * gt) - mp * mg
    s = ((2 * mp * mg + SSIM_C1) * (2 * cov + SSIM_C2)) / (
        (mp ** 2 + mg ** 2 + SSIM_C1) * (vp + vg + SSIM_C2))
    return 1 - s.mean()


def highpass(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=x.device) / 16
    k = k.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return x - F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=x.shape[1])


def flat_hf_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """High-pass L1, only where GT is flat (7x7 std < 2 levels). Weight 5 in the probe."""
    y = gt.mean(1, keepdim=True)
    m = F.avg_pool2d(y, 7, 1, 3)
    std = (F.avg_pool2d(y * y, 7, 1, 3) - m * m).clamp_min(0).sqrt()
    mask = (std < FLAT_STD_UNIT).float()
    return (mask * (highpass(pred) - highpass(gt)).abs()).sum() / mask.sum().clamp_min(1) / pred.shape[1]


def peak_bin_loss(pred: torch.Tensor, gt: torch.Tensor, radius_bins: int = 1) -> torch.Tensor:
    """Mean error power at the known pattern bins (absolute, not share-based:
    a ratio could be gamed by adding error elsewhere)."""
    e = pred - gt
    h, w = e.shape[-2:]
    spec = torch.fft.fft2(e, norm="ortho").abs() ** 2  # N,C,H,W
    mask = torch.zeros(h, w, dtype=torch.bool, device=e.device)
    for py, px in PEAK_FREQS:
        for sy, sx in ((py, px), (-py, -px)):
            by, bx = int(round(sy * h)) % h, int(round(sx * w)) % w
            for i in range(-radius_bins, radius_bins + 1):
                for j in range(-radius_bins, radius_bins + 1):
                    mask[(by + i) % h, (bx + j) % w] = True
    return spec[..., mask].mean()


def fft_l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Tested on CPU: no effect at weight 1. Kept for reference."""
    return torch.fft.rfft2(pred - gt, norm="ortho").abs().mean()
