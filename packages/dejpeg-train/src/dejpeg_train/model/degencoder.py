"""DeJPEGNet-E: degradation-embedding encoder.

Small conv net run ONCE per image (downscaled to 256x256), producing a 32-D
embedding used as conditioning by the restore net. Because it runs once per
image (not per tile), a global pool here does NOT violate tile invariance -- the
embedding is a per-image constant broadcast to every tile.

Trained (Phase 1) with MoCo contrastive (1.0) + LQ reconstruction (0.5) +
auxiliary QF regression (0.1). This file only defines the architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_INPUT_SIZE = 256
OUT_DIM = 32


class DeJPEGNetE(nn.Module):
    """4 stride-2 conv blocks (3->16->32->64->64) -> global pool -> MLP -> 32-D.

    Input is resized to 256x256 upstream; at that size the feature map collapses
    to 16x16 after the four stride-2 blocks, then is globally pooled. ~70k params.
    """

    def __init__(self, out_dim: int = OUT_DIM, in_ch: int = 3):
        super().__init__()
        self.out_dim = out_dim
        self.blocks = nn.ModuleList(
            [
                self._block(in_ch, 16),
                self._block(16, 32),
                self._block(32, 64),
                self._block(64, 64),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).flatten(1)
        return self.mlp(x)
