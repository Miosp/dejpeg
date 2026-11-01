"""L1 + LPIPS(VGG) loss on a random 128x128 crop.

The crop keeps the perceptual term cheap and decorrelated across steps; the
numerics come from forcing fp32 outside autocast plus the patched
normalize_tensor (D6). Train on VGG, evaluate with LPIPS-Alex/DISTS -- never
the same network for both (metric-leakage guard).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def patch_lpips_normalization() -> None:
    """D6: eps inside the sqrt in lpips.normalize_tensor.

    Upstream normalizes as x / (sqrt(sum x^2) + eps): an all-zero feature
    vector sends an infinite gradient through sqrt. x / sqrt(sum x^2 + eps)
    is finite everywhere and differs only where features are ~0.
    """
    import lpips

    def _normalize_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)

    lpips.normalize_tensor = _normalize_tensor
    if hasattr(lpips, "lpips"):  # some versions alias the inner module
        lpips.lpips.normalize_tensor = _normalize_tensor


class PerceptualLoss(nn.Module):
    def __init__(self, net: str = "vgg", crop: int | None = 128):
        super().__init__()
        import lpips  # lazy so the core install works without training extras

        patch_lpips_normalization()
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
        x = (2 * pred - 1).float()
        y = (2 * target - 1).float()
        # D6: LPIPS runs in fp32, outside any surrounding autocast --
        # bf16/fp16 activations here were a historical NaN source.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.model(x, y).mean()
