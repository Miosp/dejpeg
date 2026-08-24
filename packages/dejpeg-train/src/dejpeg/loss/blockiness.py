"""Relative blockiness loss (spec §3.1).

Penalizes only EXCESS 8x8 block-boundary discontinuity, relative to the target's
own boundary/interior ratio. Zero when the prediction's ratio is at or below the
target's -- so a clean image with strong legitimate edges falling on the 8x8 grid
is NOT penalized (an absolute penalty would fail that). The grid offset comes
from the degradation record (post-resize grid phase).
"""
from __future__ import annotations

import torch

BLOCK = 8


def _boundary_interior_means(img: torch.Tensor, offset: tuple[int, int], block: int = BLOCK):
    """Return (D_boundary, D_interior) averaged abs inter-pixel differences.

    ``D_boundary`` averages diffs at lines BETWEEN 8x8 blocks; ``D_interior``
    averages diffs at block-midpoint lines. With grid offset (gx, gy), a boundary
    line after block-row r sits at index (gy + r*block - 1) in the diff tensor.
    """
    dv = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()  # along H, index i = edge after row i
    dh = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()  # along W

    gx, gy = offset
    h_edges = dv.shape[2]  # == H-1
    w_edges = dh.shape[3]  # == W-1

    def mask(num_edges: int, o: int):
        idx = torch.arange(num_edges, device=dv.device)
        bnd = (idx % block) == ((o + block - 1) % block)
        intr = (idx % block) == ((o + block // 2 - 1) % block)
        return bnd, intr

    bnd_h, intr_h = mask(h_edges, gy)
    bnd_w, intr_w = mask(w_edges, gx)

    d_b = (dv[:, :, bnd_h, :].mean() + dh[:, :, :, bnd_w].mean()) / 2
    d_i = (dv[:, :, intr_h, :].mean() + dh[:, :, :, intr_w].mean()) / 2
    return d_b, d_i


def blockiness_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    offset: tuple[int, int] = (0, 0),
    block: int = BLOCK,
    eps: float = 1e-6,
) -> torch.Tensor:
    """relu(ratio_pred - ratio_target). Zero unless pred is blockier than target."""
    pb, pi = _boundary_interior_means(pred, offset, block)
    tb, ti = _boundary_interior_means(target, offset, block)
    ratio_p = pb / (pi + eps)
    ratio_t = tb / (ti + eps)
    return torch.relu(ratio_p - ratio_t)
