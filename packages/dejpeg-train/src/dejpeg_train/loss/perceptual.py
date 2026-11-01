"""Perceptual losses.

Convention: inputs are float images in [0,1], NCHW. Lpips expects [-1,1] so we
remap internally.

  * training perceptual core (anti-collapse): LPIPS-VGG, computed on a random
    128x128 crop, forced fp32 (D6: autocast LPIPS is a NaN source;
    lpips.normalize_tensor patched to eps-inside-sqrt).
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
    def __init__(self, net: str = "vgg", crop: int | None = None, pretrained: bool = True):
        super().__init__()
        import lpips  # imported lazily so test collection works offline

        patch_lpips_normalization()
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
        x = (2 * pred - 1).float()
        y = (2 * target - 1).float()
        # D6: VGG/LPIPS runs in fp32, outside any surrounding autocast --
        # bf16/fp16 activations here were a historical NaN source.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.model(x, y).mean()
