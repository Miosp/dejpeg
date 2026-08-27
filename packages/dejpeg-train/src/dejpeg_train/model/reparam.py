"""Reparameterization utilities.

1. ``fuse_rep_dwconv`` -- RepDWConv3x3 multi-branch (dw3 + dw1 + identity) compacts
   to a single depthwise 3x3.
2. ``fuse_stacked_3x3_to_5x5`` -- two linear 3x3 convs with no nonlinearity between
   compact to a single 5x5 conv (2026 VARH-AI NTIRE winner). Applies widely here
   because NAFNet is activation-free. Implementation uses impulse probing so the
   kernel is recovered with no sign/flip ambiguity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def fuse_rep_dwconv(module) -> nn.Conv2d:
    """Fuse a RepDWConv3x3 into one depthwise 3x3 conv (bias-free)."""
    c = module.channels
    w3 = module.dw3.weight.detach()    # (c, 1, 3, 3)
    w1 = module.dw1.weight.detach()    # (c, 1, 1, 1)
    fused = w3.clone()
    fused[:, 0, 1, 1] += w1[:, 0, 0, 0]
    fused[:, 0, 1, 1] += 1.0           # identity branch
    out = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
    out.weight = nn.Parameter(fused)
    out.eval()
    return out


def fuse_stacked_3x3_to_5x5(conv_a: nn.Conv2d, conv_b: nn.Conv2d) -> nn.Conv2d:
    """Fuse two sequential linear 3x3 convs into one 5x5 conv.

    Requires ``conv_b(conv_a(x))`` semantics, both 3x3 with padding=1, compatible
    channels, no nonlinearity between, and ``conv_a`` bias-free (conv_b may carry
    bias). Output is a 5x5 conv (padding=2) that reproduces the cascade exactly.
    """
    assert conv_a.kernel_size == (3, 3) and conv_b.kernel_size == (3, 3), "both must be 3x3"
    assert conv_a.padding == (1, 1) and conv_b.padding == (1, 1), "both must use padding=1"
    assert conv_a.bias is None, "conv_a must be bias-free (fold into conv_b explicitly if needed)"
    assert conv_b.in_channels == conv_a.out_channels, "channel mismatch in cascade"

    i_a = conv_a.in_channels
    o_b = conv_b.out_channels

    # Probe with conv_b.bias temporarily zeroed (its spatially-uniform response
    # would otherwise corrupt the per-channel kernel extraction); restore after.
    has_bias = conv_b.bias is not None
    saved_bias = conv_b.bias.detach().clone() if has_bias else None
    if has_bias:
        conv_b.bias.data.zero_()

    # The composed cross-correlation kernel K[r] = sum_{p+q=r} wb[q]*wa[p]. A unit
    # impulse at input channel i produces response[n] = K[center-n] -- i.e. the
    # kernel mirrored about the centre -- so we flip the response to recover K.
    fused = torch.zeros(o_b, i_a, 5, 5)
    with torch.no_grad():
        for i in range(i_a):
            impulse = torch.zeros(1, i_a, 5, 5)
            impulse[0, i, 2, 2] = 1.0
            response = conv_b(conv_a(impulse))            # (1, o_b, 5, 5)
            fused[:, i] = torch.flip(response[0], dims=[1, 2])

    if saved_bias is not None:
        conv_b.bias.data.copy_(saved_bias)

    out = nn.Conv2d(i_a, o_b, 5, padding=2, bias=has_bias)
    out.weight = nn.Parameter(fused)
    if has_bias:
        out.bias = nn.Parameter(saved_bias.clone())
    out.eval()
    return out
