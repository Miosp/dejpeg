"""DeJPEGNet-T: teacher (DEFINE ONLY, never trained in v1).

Larger NAFNet (C=64, enc/dec [4,4,4], bottleneck 8, ~12-15M params) conditioned on
the GROUND-TRUTH degradation record (which is never available at deploy time, so
this model is never shipped). Plain blocks (no prompt). Phase 5 only: trains on
synthetic data with known degradation, pseudo-labels scraped real-web JPEGs, and
distills into the v1 student.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import NAFBlock
from .student import Downsample, Upsample


class DeJPEGNetT(nn.Module):
    """Teacher backbone. ctx (GT degradation record) is accepted for API symmetry;
    plain blocks ignore it. Not exercised in Phase 0 beyond shape checks."""

    def __init__(
        self,
        in_ch: int = 3,
        c0: int = 64,
        enc_blocks=(4, 4, 4),
        mid_blocks: int = 8,
        dec_blocks=(4, 4, 4),
        attention: str = "lsca",
        window: int = 32,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        widths = [c0, c0 * 2, c0 * 4]
        mid_w = c0 * 4

        self.shallow = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for lvl, (w, nb) in enumerate(zip(widths, enc_blocks)):
            self.enc.append(
                nn.ModuleList(
                    [NAFBlock(w, attention, prompt=None, window=window) for _ in range(nb)]
                )
            )
            if lvl < len(widths) - 1:
                self.downs.append(Downsample(widths[lvl], widths[lvl + 1]))
        self.downs.append(Downsample(widths[-1], mid_w))
        self.mid = nn.ModuleList(
            [NAFBlock(mid_w, attention, prompt=None, window=window) for _ in range(mid_blocks)]
        )
        dec_widths = list(reversed(widths))
        up_from = [mid_w] + dec_widths[:-1]
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for cin, w, nb in zip(up_from, dec_widths, dec_blocks):
            self.ups.append(Upsample(cin, w))
            self.dec.append(
                nn.ModuleList(
                    [NAFBlock(w, attention, prompt=None, window=window) for _ in range(nb)]
                )
            )
        self.head = nn.Conv2d(widths[0], in_ch, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tile: torch.Tensor, ctx=None) -> torch.Tensor:
        x = self.shallow(tile)
        skips = []
        for lvl, blocks in enumerate(self.enc):
            for blk in blocks:
                x = blk(x)
            skips.append(x)
            if lvl < len(self.enc) - 1:
                x = self.downs[lvl](x)
        x = self.downs[-1](x)
        for blk in self.mid:
            x = blk(x)
        for i, (up, blocks) in enumerate(zip(self.ups, self.dec)):
            x = up(x)
            x = x + skips[len(skips) - 1 - i]
            for blk in blocks:
                x = blk(x)
        return tile + self.head(x)
