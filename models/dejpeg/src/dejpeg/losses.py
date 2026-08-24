"""L1 + LPIPS(VGG) loss on a random 128x128 crop.

The crop is not just speed: full-patch LPIPS under autocast is a reliable NaN
source, while cropped VGG-LPIPS stays numerically stable. Train on VGG, evaluate
with LPIPS-Alex/DISTS -- never the same network for both (metric-leakage guard).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, net: str = "vgg", crop: int | None = 128):
        super().__init__()
        import lpips  # lazy so the core install works without training extras

        self.crop = crop
        self.model = lpips.LPIPS(net=net, verbose=False)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.crop and pred.shape[-1] > self.crop and pred.shape[-2] > self.crop:
            _, _, h, w = pred.shape
            i = int(torch.randint(0, h - self.crop + 1, (1,)).item())
            j = int(torch.randint(0, w - self.crop + 1, (1,)).item())
            pred = pred[:, :, i:i + self.crop, j:j + self.crop]
            target = target[:, :, i:i + self.crop, j:j + self.crop]
        return self.model(2 * pred - 1, 2 * target - 1).mean()
