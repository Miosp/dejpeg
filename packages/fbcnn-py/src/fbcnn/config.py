"""Variant configurations for FBCNN.

Variants mirror upstream's separate weight files:
  - COLOR_REAL:   fbcnn_color.pth          (in_nc=3, out_nc=3)
  - GRAY:         fbcnn_gray.pth           (in_nc=1, out_nc=1)
  - GRAY_DOUBLE:  fbcnn_gray_double.pth    (in_nc=1, out_nc=1, trained on double-JPEG)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fbcnn.network import FBCNN

__all__ = ["Variant", "COLOR_REAL", "GRAY", "GRAY_DOUBLE", "VARIANTS", "VARIANTS_BY_ID", "build_model"]


@dataclass(frozen=True)
class Variant:
    id: str
    name: str
    in_nc: Literal[1, 3]
    out_nc: Literal[1, 3]
    nc: tuple[int, ...]
    nb: int
    weight_url: str
    weight_filename: str
    description: str


COLOR_REAL = Variant(
    id="fbcnn-color-real",
    name="FBCNN Color (Real-World)",
    in_nc=3,
    out_nc=3,
    nc=(64, 128, 256, 512),
    nb=4,
    weight_url="https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
    weight_filename="fbcnn_color.pth",
    description="Removes JPEG artifacts from real photographs with unknown compression history.",
)

GRAY = Variant(
    id="fbcnn-gray",
    name="FBCNN Grayscale",
    in_nc=1,
    out_nc=1,
    nc=(64, 128, 256, 512),
    nb=4,
    weight_url="https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray.pth",
    weight_filename="fbcnn_gray.pth",
    description="Removes JPEG artifacts from grayscale images.",
)

GRAY_DOUBLE = Variant(
    id="fbcnn-gray-double",
    name="FBCNN Grayscale (Double JPEG)",
    in_nc=1,
    out_nc=1,
    nc=(64, 128, 256, 512),
    nb=4,
    weight_url="https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray_double.pth",
    weight_filename="fbcnn_gray_double.pth",
    description="Removes artifacts from grayscale images that were JPEG-compressed twice (non-aligned).",
)


VARIANTS: tuple[Variant, ...] = (COLOR_REAL, GRAY, GRAY_DOUBLE)
VARIANTS_BY_ID: dict[str, Variant] = {v.id: v for v in VARIANTS}


def build_model(variant: Variant) -> FBCNN:
    """Instantiate an FBCNN network for the given variant. Does not load weights."""
    return FBCNN(
        in_nc=variant.in_nc,
        out_nc=variant.out_nc,
        nc=list(variant.nc),
        nb=variant.nb,
        act_mode="R",
    )
