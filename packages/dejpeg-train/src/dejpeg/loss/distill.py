"""Distillation losses (spec §3.3) -- WRITTEN BUT UNUSED IN v1 (Phase 5 only).

  * spatial-affinity: match pairwise-similarity structure of feature maps rather
    than raw activations. Width-independent (no projection): affinity is computed
    over spatial tokens with channel as the feature axis, then resized to a common
    token count, so student (C=32) and teacher (C=64) maps can be compared.
  * pseudo-label: L1 between student output and teacher pseudo-labels.

Enabled only if Phase 5 (domain-gap trigger) fires.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _affinity(feat: torch.Tensor, num_tokens: int = 64) -> torch.Tensor:
    # feat: (N, C, H, W) -> (N, T, T) cosine affinity over spatial tokens.
    n, c, h, w = feat.shape
    side = int(round(num_tokens ** 0.5))
    t = F.interpolate(feat, size=(side, side), mode="bilinear", align_corners=False)
    t = t.reshape(n, c, side * side)             # (N, C, T)
    t = F.normalize(t, dim=1)                     # unit length over C
    aff = torch.einsum("nct,ndt->ncd", t, t)      # (N, C, C) -- channel affinity
    return F.normalize(aff, dim=[1, 2])


def spatial_affinity_loss(
    student_feats: list[torch.Tensor], teacher_feats: list[torch.Tensor], num_tokens: int = 64
) -> torch.Tensor:
    """Match channel-wise affinity structure at each level. Width-independent.

    student_feats/teacher_feats: lists of (N,C,H,W) feature maps, one per level,
    paired by index. Channel affinity (CxC) is reduced to a scalar per level by
    FROBENIUS distance only when C matches; otherwise we compare the affinity
    eigen-spectra (width-invariant).
    """
    losses = []
    for s, tfeats in zip(student_feats, teacher_feats):
        a_s = _affinity(s, num_tokens)
        a_t = _affinity(tfeats, num_tokens)
        if a_s.shape == a_t.shape:
            losses.append(F.mse_loss(a_s, a_t))
        else:
            # singular-value spectrum is dimension-comparable across widths
            spec_s = torch.linalg.svdvals(a_s)  # (N, C_s)
            spec_t = torch.linalg.svdvals(a_t)  # (N, C_t)
            k = min(spec_s.shape[1], spec_t.shape[1])
            losses.append(F.mse_loss(spec_s[:, :k], spec_t[:, :k]))
    return torch.stack(losses).mean()


def pseudo_label_loss(student_out: torch.Tensor, teacher_pseudo: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(student_out, teacher_pseudo)
