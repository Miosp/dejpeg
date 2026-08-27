"""DeJPEGNet-S: the shipped student restore net.

Inputs: tile RGB (N,3,H,W) + conditioning vector ctx (N, 97) = [q_table(65),
deg_emb(32)]. Output: tile + residual (full RGB restore).

Architecture: shallow 3x3 conv 3->C; U-Net with 3 enc levels (widths C, 2C, 4C),
a bottleneck at 4C, and 3 dec levels; PixelShuffle upsampling (no transpose
conv); NAFBlocks (activation-free, prompt-conditioned). Tile-invariant by
construction: no global op anywhere (LSCA uses a local 32x32 pool).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from .blocks import NAFBlock
from .prompt import COND_DIM, build_conditioning


class Downsample(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """1x1 conv -> PixelShuffle(2). Channel cout at 2x spatial resolution."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout * 4, 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class DeJPEGNetS(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        c0: int = 30,
        enc_blocks=(2, 2, 4),
        mid_blocks: int = 4,
        dec_blocks=(2, 2, 2),
        attention: str = "lsca",
        use_prompt: bool = True,
        window: int = 32,
        num_prompts: int = 5,
        cond_dim: int = COND_DIM,
        grad_checkpoint: bool = False,
        cond_mode: str = "prompt",
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.use_prompt = use_prompt
        self.cond_mode = cond_mode if cond_mode else ("prompt" if use_prompt else "none")
        self.grad_checkpoint = grad_checkpoint
        widths = [c0, c0 * 2, c0 * 4]
        mid_w = c0 * 4

        def mk_prompt(c):
            return build_conditioning(self.cond_mode, c, cond_dim, num_prompts)

        self.shallow = nn.Conv2d(in_ch, widths[0], 3, padding=1)

        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for lvl, (w, nb) in enumerate(zip(widths, enc_blocks)):
            self.enc.append(
                nn.ModuleList([NAFBlock(w, attention, mk_prompt(w), window) for _ in range(nb)])
            )
            if lvl < len(widths) - 1:
                self.downs.append(Downsample(widths[lvl], widths[lvl + 1]))
        self.downs.append(Downsample(widths[-1], mid_w))  # into bottleneck

        self.mid = nn.ModuleList(
            [NAFBlock(mid_w, attention, mk_prompt(mid_w), window) for _ in range(mid_blocks)]
        )

        dec_widths = list(reversed(widths))  # 4C, 2C, C
        up_from = [mid_w] + dec_widths[:-1]   # channels feeding each upsample
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for cin, w, nb in zip(up_from, dec_widths, dec_blocks):
            self.ups.append(Upsample(cin, w))
            self.dec.append(
                nn.ModuleList([NAFBlock(w, attention, mk_prompt(w), window) for _ in range(nb)])
            )

        self.head = nn.Conv2d(widths[0], in_ch, 3, padding=1)
        # zero-init head so the untrained model outputs the input (residual identity)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _blk(self, blk, x, ctx):
        """NAFBlock call, optionally under gradient checkpointing (training only).
        Checkpointing each block trades ~30% extra compute for a large activation-
        memory saving, enabling substantially larger batches -> better GPU utilization."""
        if self.grad_checkpoint and self.training:
            return ckpt.checkpoint(blk, x, ctx, use_reentrant=False)
        return blk(x, ctx)

    def forward(self, tile: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.shallow(tile)
        skips = []
        for lvl, blocks in enumerate(self.enc):
            for blk in blocks:
                x = self._blk(blk, x, ctx)
            skips.append(x)
            if lvl < len(self.enc) - 1:
                x = self.downs[lvl](x)
        x = self.downs[-1](x)
        for blk in self.mid:
            x = self._blk(blk, x, ctx)
        for i, (up, blocks) in enumerate(zip(self.ups, self.dec)):
            x = up(x)
            x = x + skips[len(skips) - 1 - i]
            for blk in blocks:
                x = self._blk(blk, x, ctx)
        return tile + self.head(x)


def dropout_qtable(qtable: torch.Tensor, p: float, generator=None) -> torch.Tensor:
    """Zero the 64 quant values and clear the validity flag for a fraction ``p``.

    qtable: (N, 65) with layout [64 quant values | 1 validity flag]. Returns a new
    tensor; rows selected for dropout become all-zero with validity 0, training
    the non-JPEG / unknown-table path.
    """
    out = qtable.clone()
    if p <= 0:
        return out
    n = out.shape[0]
    mask = torch.rand(n, generator=generator) < p
    out[mask, :64] = 0.0
    out[mask, 64] = 0.0
    return out


def build_ctx(qtable: torch.Tensor, deg_emb: torch.Tensor) -> torch.Tensor:
    """Concat q_table(65) and deg_emb(32) into the (N, 97) conditioning vector."""
    return torch.cat([qtable, deg_emb], dim=1)
