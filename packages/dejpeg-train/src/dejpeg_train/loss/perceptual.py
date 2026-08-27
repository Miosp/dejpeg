"""Perceptual losses.

Convention: inputs are float images in [0,1], NCHW. Lpips expects [-1,1] so we
remap internally.

Two strictly separate instances are used:
  * training perceptual core (anti-collapse): LPIPS-VGG, computed on a random
    128x128 crop (fp16 autocast + full-patch LPIPS is a NaN source).
  * evaluation gate metric: LPIPS-Alex, full image. NEVER train on Alex and never
    gate on VGG (metric leakage guard, spec §3.4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, net: str = "vgg", crop: int | None = None, pretrained: bool = True):
        super().__init__()
        import lpips  # imported lazily so test collection works offline

        self.net_name = net
        self.crop = crop
        self.model = lpips.LPIPS(net=net, pretrained=pretrained, verbose=False)
        # freeze feature extractor
        for p in self.model.parameters():
            p.requires_grad = False

    def _random_crop(self, *tensors: torch.Tensor):
        h, w = tensors[0].shape[-2:]
        assert tensors[0].shape[-2:] == tensors[1].shape[-2:]
        if self.crop is None or h <= self.crop or w <= self.crop:
            return tensors
        i = int(torch.randint(0, h - self.crop + 1, (1,)).item())
        j = int(torch.randint(0, w - self.crop + 1, (1,)).item())
        return tuple(t[:, :, i : i + self.crop, j : j + self.crop] for t in tensors)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._random_crop(pred, target)
        x = 2 * pred - 1
        y = 2 * target - 1
        return self.model(x, y).mean()
