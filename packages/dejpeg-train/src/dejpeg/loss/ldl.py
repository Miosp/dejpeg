"""LDL-style local artifact-map loss (spec §3.2).

Spatially-targeted penalty: weight the residual by an artifact map that is high
where the residual's local variance is anomalous AND the target is locally smooth
(no real texture to explain the residual). Lets us raise the adversarial weight
at lower hallucination risk. Demonstrated on larger backbones; stability at ~1.5M
params is unverified -> Phase 2 on/off comparison.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _local_var(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    xp = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    mu = F.avg_pool2d(xp, k, stride=1)
    return (F.avg_pool2d(xp * xp, k, stride=1) - mu * mu).clamp(min=0)


def ldl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 7,
    tau: float = 5e-3,
    eps: float = 1e-6,
) -> torch.Tensor:
    resvar = _local_var(pred - target, k).mean(dim=1, keepdim=True)
    artmap = resvar / (resvar.amax(dim=[2, 3], keepdim=True) + eps)
    tvar = _local_var(target, k).mean(dim=1, keepdim=True)
    gate = 1.0 / (1.0 + tvar / tau)  # ~1 where target smooth, ~0 on real texture
    return (gate * artmap * (pred - target).abs()).mean()
