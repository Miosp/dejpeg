"""Building blocks for FBCNN — ported from upstream models/network_fbcnn.py.

Modernized: functional.relu instead of relu_, type hints, no other behaviour changes.
Reference: https://github.com/jiaxi-jiang/FBCNN/blob/main/models/network_fbcnn.py
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["sequential", "conv", "ResBlock", "downsample_strideconv", "upsample_convtranspose", "QFAttention"]


def sequential(*args: Union[nn.Module, nn.Sequential]) -> nn.Sequential:
    """Flatten *args into a single nn.Sequential.

    If a single nn.Sequential is passed, it is returned unchanged. If a single
    nn.Module is passed, it is wrapped. Multiple args are concatenated, with
    any nested nn.Sequential children promoted to the top level.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules: list[nn.Module] = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(
    in_channels: int = 64,
    out_channels: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    mode: str = "CR",
    negative_slope: float = 0.2,
) -> nn.Sequential:
    """Construct a sequential conv block from a mode string.

    Mode characters: C=Conv2d, T=ConvTranspose2d, B=BatchNorm2d, I=InstanceNorm2d,
    R=ReLU, r=LeakyReLU(negative_slope), L=Sigmoid, l=Tanh, M=MaxPool2d (kernel/stride from mode prefix),
    A=AvgPool2d (kernel/stride from mode prefix).
    """
    if mode[0] not in ["C", "T", "M", "A"]:
        raise AssertionError(f"Unsupported conv mode prefix: {mode[0]!r}")
    modes: list[nn.Module] = []
    pool_kernel_size = None
    pool_stride = None
    for ch in mode:
        if ch == "C":
            modes.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif ch == "T":
            modes.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif ch == "B":
            modes.append(nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        elif ch == "I":
            modes.append(nn.InstanceNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False))
        elif ch == "R":
            modes.append(nn.ReLU(inplace=False))  # never inplace — preserves export
        elif ch == "r":
            modes.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        elif ch == "L":
            modes.append(nn.Sigmoid())
        elif ch == "l":
            modes.append(nn.Tanh())
        elif ch == "M":
            # kernel/stride read from the digit prefix at mode[0]; this branch is unreachable
            # because pool blocks are constructed in downsample_maxpool; kept for parity.
            modes.append(nn.MaxPool2d(pool_kernel_size, pool_stride))
        elif ch == "A":
            modes.append(nn.AvgPool2d(pool_kernel_size, pool_stride))
        else:
            raise NotImplementedError(f"Unknown mode character: {ch!r}")
    return sequential(*modes)


class ResBlock(nn.Module):
    """Residual block: x + conv(relu(conv(x))) following the mode string.

    Identical to upstream except no in-place ops. Default mode 'CRC' = Conv-ReLU-Conv.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        mode: str = "CRC",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        assert in_channels == out_channels, "ResBlock requires in_channels == out_channels"
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res(x)


def downsample_strideconv(
    in_channels: int = 64,
    out_channels: int = 64,
    kernel_size: int = 2,
    stride: int = 2,
    padding: int = 0,
    bias: bool = True,
    mode: str = "2R",
    negative_slope: float = 0.2,
) -> nn.Sequential:
    """Stride-2 convolution downsample. Mode prefix digit is the stride."""
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], f"mode examples: 2, 2R, 2BR, 3, ..., 4BR. Got {mode!r}"
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)


def upsample_convtranspose(
    in_channels: int = 64,
    out_channels: int = 3,
    kernel_size: int = 2,
    stride: int = 2,
    padding: int = 0,
    bias: bool = True,
    mode: str = "2R",
    negative_slope: float = 0.2,
) -> nn.Sequential:
    """ConvTranspose upsample. Mode prefix digit is the stride."""
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], f"mode examples: 2, 2R, 2BR, 3, ..., 4BR. Got {mode!r}"
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)


class QFAttention(nn.Module):
    """Residual block modulated by per-sample (gamma, beta) from the QF predictor.

    forward(x, gamma, beta) returns x + gamma * conv_block(x) + beta.
    gamma and beta are scalars per sample (shape [N] or [N,1]); we unsqueeze
    them to [N,1,1,1] inside forward.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        mode: str = "CRC",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        assert in_channels == out_channels, "QFAttention requires in_channels == out_channels"
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x + gamma * self.res(x) + beta
