"""PatchGAN adversarial loss (spec §3).

Discriminator is a 70x70 PatchGAN with spectral norm. It is NOT restore_net, so
BatchNorm/LeakyReLU are allowed here (the activation-free constraint applies only
to the restore net). Hinge loss (stable). adv weight 0.15 / disc 1.0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sn_conv(i, o, k, s, p):
    return nn.utils.spectral_norm(nn.Conv2d(i, o, k, s, p, bias=False))


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN: takes concat([input, image]) on in_ch channels."""

    def __init__(self, in_ch: int = 6, base: int = 64, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            _sn_conv(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        ]
        nf, prev = base, base
        for n in range(1, n_layers):
            nf = min(prev * 2, base * 8)
            stride = 1 if n == n_layers - 1 else 2
            layers += [_sn_conv(prev, nf, 4, stride, 1), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True)]
            prev = nf
        layers += [_sn_conv(prev, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
