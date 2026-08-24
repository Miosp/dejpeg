"""FBCNN main network class — ported from upstream models/network_fbcnn.py.

Modernized: type hints, no in-place ops, no data-dependent tensor control flow.
Module attribute names are kept identical to upstream so the original pretrained
state_dict loads with strict=True.
Reference: https://github.com/jiaxi-jiang/FBCNN/blob/main/models/network_fbcnn.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fbcnn.blocks import (
    QFAttention,
    ResBlock,
    conv,
    downsample_strideconv,
    sequential,
    upsample_convtranspose,
)

__all__ = ["FBCNN"]


class FBCNN(nn.Module):
    """FBCNN: Flexible Blind Convolutional Neural Network for JPEG artifact removal.

    Encoder→middle→decoder U-shape with a QF (quality factor) predictor branch
    whose outputs modulate the decoder via QFAttention blocks. Constructor and
    forward signatures match upstream exactly.

    Args:
        in_nc: input channel count (RGB=3).
        out_nc: output channel count.
        nc: list of channel widths per encoder level (length must be 4).
        nb: number of ResBlock / QFAttention blocks per level.
        act_mode: activation mode passed to ResBlock/QFAttention. 'R' = ReLU.
        downsample_mode: only 'strideconv' supported.
        upsample_mode: only 'convtranspose' supported.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nc: list[int] = [64, 128, 256, 512],
        nb: int = 4,
        act_mode: str = "R",
        downsample_mode: str = "strideconv",
        upsample_mode: str = "convtranspose",
    ) -> None:
        super().__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode="C")
        self.nb = nb
        self.nc = nc

        if act_mode != "R":
            raise NotImplementedError(f"act_mode [{act_mode!s}] is not supported")

        if downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(f"downsample mode [{downsample_mode!s}] is not supported")

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode="2"),
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode="2"),
        )
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=True, mode="2"),
        )

        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C") for _ in range(nb)]
        )

        self.m_body_decoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C") for _ in range(nb)]
        )

        if upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode!s}] is not supported")

        self.m_up3 = nn.ModuleList(
            [
                upsample_block(nc[3], nc[2], bias=True, mode="2"),
                *[QFAttention(nc[2], nc[2], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            ]
        )
        self.m_up2 = nn.ModuleList(
            [
                upsample_block(nc[2], nc[1], bias=True, mode="2"),
                *[QFAttention(nc[1], nc[1], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            ]
        )
        self.m_up1 = nn.ModuleList(
            [
                upsample_block(nc[1], nc[0], bias=True, mode="2"),
                *[QFAttention(nc[0], nc[0], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            ]
        )

        self.m_tail = conv(nc[0], out_nc, bias=True, mode="C")

        self.qf_pred = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.qf_embed = sequential(
            torch.nn.Linear(1, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.to_gamma_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Tanh())

    def forward(self, L: torch.Tensor, qf_input: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = L.size()[-2:]
        padding_bottom = int(np.ceil(h / 8) * 8 - h)
        padding_right = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(L)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        x = self.m_body_encoder(x4)
        qf = self.qf_pred(x)
        x = self.m_body_decoder(x)

        qf_embedding = self.qf_embed(qf_input) if qf_input is not None else self.qf_embed(qf)

        gamma_3 = self.to_gamma_3(qf_embedding)
        beta_3 = self.to_beta_3(qf_embedding)
        gamma_2 = self.to_gamma_2(qf_embedding)
        beta_2 = self.to_beta_2(qf_embedding)
        gamma_1 = self.to_gamma_1(qf_embedding)
        beta_1 = self.to_beta_1(qf_embedding)

        x = x + x4
        x = self.m_up3[0](x)
        for i in range(self.nb):
            x = self.m_up3[i + 1](x, gamma_3, beta_3)

        x = x + x3
        x = self.m_up2[0](x)
        for i in range(self.nb):
            x = self.m_up2[i + 1](x, gamma_2, beta_2)

        x = x + x2
        x = self.m_up1[0](x)
        for i in range(self.nb):
            x = self.m_up1[i + 1](x, gamma_1, beta_1)

        x = x + x1
        x = self.m_tail(x)
        x = x[..., :h, :w]

        return x, qf
