"""Standalone DISTS metric (Ding et al., IEEE TPAMI 2020, arXiv 2004.07728).

Faithful port of the official implementation (github.com/dingkeyan93/DISTS,
MIT License, (c) 2020 Keyan Ding) so gate numbers match the reference. VGG16
features at 5 stages (plus the raw RGB input as a 6th group), L2pooling between
stages, learned per-channel alpha/beta weighting (weights.pt shipped with the
official repo, bundled at ~/dejpeg-work/weights/DISTS_weights.pt).

DISTS(x, x) == 0; larger = worse. Inputs: float RGB in [0, 1], NCHW.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_WEIGHTS = Path.home() / "dejpeg-work/weights/DISTS_weights.pt"


class L2pooling(nn.Module):
    def __init__(self, filter_size: int = 5, stride: int = 2, channels: int = 3):
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer("filter", g[None, None, :, :].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x**2
        out = F.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return (out + 1e-12).sqrt()


class DISTS(nn.Module):
    chns = [3, 64, 128, 256, 512, 512]

    def __init__(self, weights_path: str | Path = _WEIGHTS):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.stage1 = nn.Sequential(*[vgg[x] for x in range(0, 4)])                       # -> relu1_2 (64)
        self.stage2 = nn.Sequential(L2pooling(channels=64), *[vgg[x] for x in range(5, 9)])    # -> relu2_2 (128)
        self.stage3 = nn.Sequential(L2pooling(channels=128), *[vgg[x] for x in range(10, 16)])  # -> relu3_3 (256)
        self.stage4 = nn.Sequential(L2pooling(channels=256), *[vgg[x] for x in range(17, 23)])  # -> relu4_3 (512)
        self.stage5 = nn.Sequential(L2pooling(channels=512), *[vgg[x] for x in range(24, 30)])  # -> relu5_3 (512)
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        w = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        self.alpha.data = w["alpha"]
        self.beta.data = w["beta"]
        self.eval()

    def _feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = (x - self.mean) / self.std
        f0 = h
        f1 = self.stage1(h)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return [f0, f1, f2, f3, f4, f5]

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (N, 3, H, W) in [0, 1]. Returns per-image DISTS scores (lower = better)."""
        fx = self._feats(x)
        fy = self._feats(y)
        c1 = c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        dist1 = dist2 = 0.0
        for k in range(len(self.chns)):
            mx = fx[k].mean([2, 3], keepdim=True)
            my = fy[k].mean([2, 3], keepdim=True)
            s1 = (2 * mx * my + c1) / (mx**2 + my**2 + c1)
            dist1 = dist1 + (alpha[k] * s1).sum(1, keepdim=True)
            vx = ((fx[k] - mx) ** 2).mean([2, 3], keepdim=True)
            vy = ((fy[k] - my) ** 2).mean([2, 3], keepdim=True)
            cov = (fx[k] * fy[k]).mean([2, 3], keepdim=True) - mx * my
            s2 = (2 * cov + c2) / (vx + vy + c2)
            dist2 = dist2 + (beta[k] * s2).sum(1, keepdim=True)
        return 1 - (dist1 + dist2).squeeze(-1).squeeze(-1)
