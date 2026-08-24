"""High-level inference helper for the ported FBCNN.

Used by the conversion smoke test and PSNR tests. Not used by the deployed
runtime (the deployed runtime uses the ONNX artifact).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fbcnn.config import Variant, build_model
from fbcnn.weights import load_pretrained

__all__ = ["run"]


def _uint8_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWxC uint8 (0-255) -> 1xCxHxW float32 (0-1)."""
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC, got shape {img.shape}")
    h, w, c = img.shape
    t = torch.from_numpy(img).float().permute(2, 0, 1).contiguous() / 255.0
    return t.unsqueeze(0)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """1xCxHxW float32 (0-1) -> HxWxC uint8 (0-255)."""
    t = t.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    return (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def run(
    image: np.ndarray,
    qf: int | None,
    variant: Variant,
    weights_dir: Path,
) -> tuple[np.ndarray, int]:
    """Run FBCNN on a single HxWxC uint8 image.

    Args:
        image: HxWxC uint8 input (C = variant.in_nc).
        qf: JPEG quality factor 10-100. If None, model predicts it.
        variant: which variant to run.
        weights_dir: directory containing the variant's .pth file.

    Returns:
        (restored_uint8, used_qf) where used_qf is the integer QF actually applied
        (predicted by the model if input qf was None).
    """
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")

    net = build_model(variant)
    load_pretrained(net, variant, weights_dir)
    net.eval()

    with torch.no_grad():
        x = _uint8_to_tensor(image)
        qf_input = None if qf is None else torch.tensor([[1.0 - qf / 100.0]])
        out_e, out_qf = net(x, qf_input)
        # QF convention: model outputs `1 - qf/100`; invert to get qf in [0,100]
        used_qf_internal = float(out_qf.squeeze().item())
        used_qf = round((1.0 - used_qf_internal) * 100.0)
        return _tensor_to_uint8(out_e), used_qf
