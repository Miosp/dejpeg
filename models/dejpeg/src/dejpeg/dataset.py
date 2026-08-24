"""Training data: an infinite stream of aligned (degraded, clean) patches.

Reads any folder of images, degrades on the fly (see :mod:`dejpeg.degrade`) and
emits fixed-size patches. Two mix-ins address failure modes we found the hard way:

* ``identity_frac`` -- a fraction of samples pass through un-degraded with a clean
  target. Without these the model over-sharpens near-clean inputs; with them it
  learns when *not* to act (near-identity behavior).
* ``gray_frac`` -- a fraction of pairs are converted to grayscale-replicated RGB.
  Without this anchor the model hallucinates color on out-of-distribution
  grayscale/flat-chroma content (it learned "JPEG artifacts are colorful" too
  well); with even a quarter of pairs grayed, chroma robustness holds everywhere.

Worker processes each get an independent seed and image subset, so the combined
stream is diverse and race-free.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

from .degrade import DegradationSampler

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class PatchStream(IterableDataset):
    def __init__(
        self,
        image_dir: str | Path,
        patch: int = 256,
        crop_size: int = 512,
        identity_frac: float = 0.10,
        gray_frac: float = 0.25,
        seed: int = 0,
    ):
        super().__init__()
        self.image_paths = sorted(
            p for p in Path(image_dir).rglob("*") if p.suffix.lower() in EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"no images under {image_dir}")
        self.patch = patch
        self.crop_size = crop_size
        self.identity_frac = identity_frac
        self.gray_frac = gray_frac
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed + wid * 7919)
        sampler = DegradationSampler(seed=self.seed + wid * 104729)
        while True:
            path = rng.choice(self.image_paths)
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
            if min(arr.shape[:2]) < self.crop_size:
                scale = self.crop_size / min(arr.shape[:2])
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
                arr = np.asarray(img)

            if rng.random() < self.identity_frac:
                jpeg = arr.astype(np.float32) / 255.0
                target = jpeg
            else:
                jpeg, target = sampler.sample(arr)
                if rng.random() < self.gray_frac:
                    jpeg = np.repeat(jpeg.mean(axis=2, keepdims=True), 3, axis=2)
                    target = np.repeat(target.mean(axis=2, keepdims=True), 3, axis=2)

            h, w = jpeg.shape[:2]
            dy, dx = rng.randrange(0, h - self.patch + 1), rng.randrange(0, w - self.patch + 1)
            crop = (slice(dy, dy + self.patch), slice(dx, dx + self.patch))
            yield (
                torch.from_numpy(jpeg[crop].transpose(2, 0, 1)),
                torch.from_numpy(target[crop].transpose(2, 0, 1)),
            )


def make_loader(image_dir: str | Path, batch_size: int = 8, num_workers: int = 6,
                patch: int = 256, **stream_kw) -> DataLoader:
    stream = PatchStream(image_dir, patch=patch, **stream_kw)
    return DataLoader(stream, batch_size=batch_size, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None, drop_last=True)
