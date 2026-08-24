"""PromptCIR-style conditioning: learnable prompt bank + local soft weights + FiLM.

Fully convolutional -- no global pooling anywhere -- so tile-invariant by
construction. The conditioning vector is ``[q_table(65), deg_emb(32)] = 97-D``.
``q_table`` may be zeroed with its validity flag cleared (non-JPEG path); the
module consumes whatever vector it is given, transparently.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import simple_gate_last

COND_DIM = 97  # 65 (q_table: 64 quant values + 1 validity flag) + 32 (deg_emb)


class PromptModule(nn.Module):
    """Per-level prompt bank with position-local soft selection over P prompts.

    prompt_bank: (P, 2C) -> produces (gamma, beta) via FiLM.
    ctx projection is activation-free: Linear -> SimpleGate (elementwise).
    Selection weights are computed by a 1x1 conv over (feat + per-image ctx),
    softmax over P -- local per position, no pooling.
    """

    def __init__(self, channels: int, cond_dim: int = COND_DIM, num_prompts: int = 5):
        super().__init__()
        self.channels = channels
        self.num_prompts = num_prompts
        self.prompt_bank = nn.Parameter(torch.empty(num_prompts, 2 * channels))
        nn.init.normal_(self.prompt_bank, std=0.02)
        self.ctx_fc = nn.Linear(cond_dim, 2 * channels)
        self.weight_conv = nn.Conv2d(channels, num_prompts, 1)

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor):
        # feat: (N, C, H, W)  [LayerNorm output]; ctx: (N, cond_dim)
        g = simple_gate_last(self.ctx_fc(ctx))        # (N, C)
        f = feat + g.unsqueeze(-1).unsqueeze(-1)       # broadcast per-image ctx
        w = self.weight_conv(f).softmax(dim=1)         # (N, P, H, W) local weights
        cond = torch.einsum("nphw,pc->nchw", w, self.prompt_bank)  # (N, 2C, H, W)
        gamma, beta = cond.chunk(2, dim=1)
        return gamma, beta


class ScalarFiLM(nn.Module):
    """Simpler conditioning baseline: ctx -> one (gamma, beta) per channel, broadcast
    across all spatial positions (no per-position variation, no prompt bank).

    P3 ablation variant. Same forward signature as PromptModule so it drops into
    NAFBlock. Activation-free (Linear + chunk, no nonlinearity). Tile-invariant
    (the (gamma, beta) is a per-image constant)."""

    def __init__(self, channels: int, cond_dim: int = COND_DIM):
        super().__init__()
        self.channels = channels
        self.ctx_fc = nn.Linear(cond_dim, 2 * channels)

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor):
        # feat: (N, C, H, W); ctx: (N, cond_dim)
        gb = self.ctx_fc(ctx)                          # (N, 2C)
        gamma, beta = gb.chunk(2, dim=-1)              # each (N, C)
        n = gamma.shape[0]
        return gamma.view(n, -1, 1, 1), beta.view(n, -1, 1, 1)


def build_conditioning(mode: str, channels: int, cond_dim: int = COND_DIM, num_prompts: int = 5):
    """Factory for the three P3 conditioning variants. Returns a module or None."""
    if mode == "prompt":
        return PromptModule(channels, cond_dim, num_prompts)
    if mode == "scalar":
        return ScalarFiLM(channels, cond_dim)
    if mode == "none":
        return None
    raise ValueError(f"unknown cond_mode {mode!r}; expected one of prompt/scalar/none")
