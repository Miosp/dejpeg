"""Validation contact sheet (spec §7).

Fixed 16-image contact sheet PNG every 5k iters -- the single most useful
perceptual debugging artifact. Same 16 validation images every time so changes
across iterations are directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def contact_sheet(
    images: list[np.ndarray],
    cols: int = 4,
    thumb: int = 160,
    pad: int = 4,
    bg: int = 32,
    path: str | Path | None = None,
) -> np.ndarray:
    """Lay out up to ``cols*rows`` images into a grid PNG.

    images: list of (H,W,3) uint8 arrays. Extra slots are filled with bg.
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    W = cols * thumb + (cols + 1) * pad
    H = rows * thumb + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (bg, bg, bg))
    for i, arr in enumerate(images):
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = Image.fromarray(arr.astype(np.uint8)).resize((thumb, thumb), Image.BILINEAR)
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + pad)
        sheet.paste(img, (x, y))
    out = np.array(sheet)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(str(path))
    return out
