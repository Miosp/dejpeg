"""NAFNet-derived building blocks for DeJPEGNet.

Activation-free by construction: LayerNorm + SimpleGate + elementwise ops only.
No ReLU/GELU/Sigmoid in the default restore_net path (the SPAN ablation uses
Sigmoid internally as an opt-in attention variant, never on the default path).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTIONS = ("lsca", "span", "none", "global")


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW (NAFNet convention)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        d = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(d + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split channels in half, elementwise multiply. Activation-free nonlinearity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


def simple_gate_last(x: torch.Tensor) -> torch.Tensor:
    """SimpleGate along the last dimension (for 2-D tensors)."""
    a, b = x.chunk(2, dim=-1)
    return a * b


class RepDWConv3x3(nn.Module):
    """Depthwise reparameterized conv: parallel {3x3, 1x1, identity} summed.

    Fuses to a single depthwise 3x3 at deploy time (see :meth:`fuse`).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw3(x) + self.dw1(x) + x

    def fuse(self) -> nn.Conv2d:
        from .reparam import fuse_rep_dwconv

        return fuse_rep_dwconv(self)


class LSCA(nn.Module):
    """Local Simple Channel Attention.

    Fixed-window avg-pool -> 1x1 conv -> nearest-upsample -> multiplicative gate.
    The pool window is local (default 32x32), never global, so the block stays
    tile-invariant. Identical train/test behaviour; no export-time swap.
    """

    def __init__(self, channels: int, window: int = 32):
        super().__init__()
        self.window = window
        self.pool = nn.AvgPool2d(window, stride=window, ceil_mode=True, count_include_pad=False)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.pool(x)
        p = self.conv(p)
        p = F.interpolate(p, size=(h, w), mode="nearest")
        return x * p


class GlobalSCA(nn.Module):
    """GLOBAL-pool SCA. Breaks tile invariance by construction.

    Exists only as the negative control for the tile-invariance test: swapping
    LSCA for this must make the test fail, proving the test actually checks something.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x.mean(dim=[2, 3], keepdim=True)
        return x * self.conv(p)


class SPANAttention(nn.Module):
    """SPAN parameter-free attention (arXiv 2311.12770).

    sigma_a(x) = sigmoid(a*x) - 0.5 (origin-symmetric odd, range (-0.5, 0.5)).
    No pooling of any kind, so tile-invariant by construction -- a stronger
    guarantee than LSCA's local 32x32 pool. ``a`` is a per-channel scale init 1.0;
    pass ``learnable_a=False`` for the strictly parameter-free paper baseline.
    """

    def __init__(self, channels: int, learnable_a: bool = True):
        super().__init__()
        self.a = nn.Parameter(torch.ones(channels)) if learnable_a else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a.view(1, -1, 1, 1) if self.a is not None else 1.0
        v = torch.sigmoid(a * x) - 0.5
        return x * (1.0 + v)


def build_attention(name: str, channels: int, window: int = 32) -> nn.Module:
    if name == "lsca":
        return LSCA(channels, window)
    if name == "span":
        return SPANAttention(channels)
    if name == "none":
        return nn.Identity()
    if name == "global":
        return GlobalSCA(channels)
    raise ValueError(f"unknown attention {name!r}; expected one of {ATTENTIONS}")


class NAFBlock(nn.Module):
    """DeJPEGNet NAF block (canonical two-branch form: attention + Simple-FFN).

    Attention branch:
        LayerNorm -> PromptFiLM -> 1x1(C->2C) -> RepDWConv(2C) -> SimpleGate ->
        Attention -> 1x1(C->C) -> residual * beta
    FFN branch (Simple-FFN, activation-free):
        LayerNorm -> 1x1(C->2C) -> SimpleGate -> 1x1(C->C) -> residual * gamma

    Both beta and gamma are zero-init (LayerScale): the block starts as identity
    and gradually "activates" (standard NAFNet stability trick). ``prompt`` is a
    per-level PromptModule (or None); ``ctx`` is forwarded into it.
    """

    def __init__(
        self,
        channels: int,
        attention: str = "lsca",
        prompt=None,
        window: int = 32,
        ffn_expand: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.attention_name = attention

        # --- attention branch ---
        self.norm1 = LayerNorm2d(channels)
        self.prompt = prompt
        self.conv1 = nn.Conv2d(channels, 2 * channels, 1)
        self.dwconv = RepDWConv3x3(2 * channels)
        self.sg1 = SimpleGate()
        self.attn = build_attention(attention, channels, window)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # --- Simple-FFN branch (activation-free: Conv-SimpleGate-Conv) ---
        ffn_c = ffn_expand * channels
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_c, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_c // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor, ctx=None) -> torch.Tensor:
        # attention branch
        x = self.norm1(inp)
        if self.prompt is not None:
            film_g, film_b = self.prompt(x, ctx)
            x = x * film_g + film_b
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg1(x)
        x = self.attn(x)
        x = self.conv2(x)
        y = inp + x * self.beta
        # Simple-FFN branch
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma
