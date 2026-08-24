"""Inference: load the released weights and restore JPEG-decoded images.

The model is fully convolutional and tile-invariant, so any input size works:
the image is reflect-padded to a multiple of 32 (the attention window), restored
in a single forward pass, and cropped back.

An optional mild unsharp mask (``sharpness``) is applied after restoration. A
small amount measurably improves DISTS/LPIPS/no-reference scores beyond the raw
network output; 0.10 is the tuned default and 0 disables it.

Weights are distributed as a GitHub Release asset and downloaded to a local
cache on first use (see ``resolve_weights``).
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import DeJPEGNet

PAD_MULTIPLE = 32
DEFAULT_SHARPNESS = 0.10

WEIGHTS_FILENAME = "dejpeg-c40-fp16.pt"
# Download source for the shipped FP16 weights. Point this at the actual
# GitHub Release asset once the repository is published; overridable via
# DEJPEG_WEIGHTS_URL for mirrors or local hosting.
RELEASE_WEIGHTS_URL = "https://github.com/OWNER/dejpeg/releases/download/v1.0.0/dejpeg-c40-fp16.pt"


def cache_dir() -> Path:
    return Path(os.environ.get("DEJPEG_CACHE_DIR", Path.home() / ".cache" / "dejpeg"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dejpeg] downloading weights from {url}")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        with urllib.request.urlopen(url) as r:  # noqa: S310
            tmp.write(r.read())
        Path(tmp.name).replace(dest)


def resolve_weights() -> Path:
    """Locate the shipped weights file, downloading it on first use.

    Resolution order: ``$DEJPEG_WEIGHTS`` (explicit file path) > cache
    (``~/.cache/dejpeg/dejpeg-c40-fp16.pt``, override with ``DEJPEG_CACHE_DIR``).
    """
    explicit = os.environ.get("DEJPEG_WEIGHTS")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"DEJPEG_WEIGHTS points to a missing file: {path}")
        return path
    cached = cache_dir() / WEIGHTS_FILENAME
    if not cached.is_file():
        url = os.environ.get("DEJPEG_WEIGHTS_URL", RELEASE_WEIGHTS_URL)
        _download(url, cached)
    return cached


def load_model(weights_path: str | Path | None = None, device: str = "cpu") -> DeJPEGNet:
    """Build the network from the checkpoint metadata and load FP16 weights as FP32."""
    path = Path(weights_path) if weights_path else resolve_weights()
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = DeJPEGNet(**blob.get("config", {}))
    model.load_state_dict({k: v.float() for k, v in blob["state_dict"].items()})
    return model.to(device).eval()


def _pad_to_multiple(x: torch.Tensor, multiple: int = PAD_MULTIPLE) -> torch.Tensor:
    _, _, h, w = x.shape
    return torch.nn.functional.pad(
        x, (0, (multiple - w % multiple) % multiple, 0, (multiple - h % multiple) % multiple),
        mode="reflect",
    )


def unsharp_mask(x: torch.Tensor, amount: float, radius: float = 1.0) -> torch.Tensor:
    """out = x + amount * (x - gaussian(x)), per channel, on a [0,1] NCHW batch."""
    if amount <= 0:
        return x
    c = x.shape[1]
    k = int(2 * round(3 * radius) + 1)
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-coords.pow(2) / (2 * radius * radius))
    g = (g / g.sum())
    hk = g.view(1, 1, 1, k).expand(c, 1, 1, k)
    vk = g.view(1, 1, k, 1).expand(c, 1, k, 1)
    blurred = torch.nn.functional.conv2d(x, hk, padding=(0, k // 2), groups=c)
    blurred = torch.nn.functional.conv2d(blurred, vk, padding=(k // 2, 0), groups=c)
    return (x + amount * (x - blurred)).clamp(0, 1)


@torch.no_grad()
def restore_array(x: torch.Tensor, model: DeJPEGNet, sharpness: float = DEFAULT_SHARPNESS) -> torch.Tensor:
    """Restore a [0,1] float RGB tensor, NCHW or CHW. Returns the same shape."""
    squeeze = x.ndim == 3
    if squeeze:
        x = x.unsqueeze(0)
    _, _, h, w = x.shape
    out = model(_pad_to_multiple(x))
    out = unsharp_mask(out[:, :, :h, :w].clamp(0, 1), sharpness)
    return out.squeeze(0) if squeeze else out


def restore_image(
    src: str | Path | Image.Image,
    dst: str | Path | None = None,
    model: DeJPEGNet | None = None,
    sharpness: float = DEFAULT_SHARPNESS,
    device: str = "cpu",
) -> Image.Image:
    """Restore one image file (or PIL image). Writes to ``dst`` when given and returns the result."""
    if model is None:
        model = load_model(device=device)
    pil = src if isinstance(src, Image.Image) else Image.open(src)
    rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    out = restore_array(tensor, model, sharpness)
    restored = Image.fromarray((out.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    if dst is not None:
        restored.save(dst)
    return restored
