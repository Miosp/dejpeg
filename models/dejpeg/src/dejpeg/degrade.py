"""Synthetic JPEG degradation for training pairs.

Starting from a clean image, sample a realistic degradation and emit an aligned
(degraded, clean) pair:

1. random flip/rotation;
2. with p=0.35, a pre-resize (simulates images that were resampled before saving);
3. sample a quality factor -- 60% in [30, 75], 20% in [1, 30), 20% in [75, 100] --
   plus chroma subsampling {4:2:0 50%, 4:2:2 25%, 4:4:4 25%} and p=0.10 progressive;
4. with p=0.40, 1-2 extra JPEG passes at nearby qualities (generation loss);
5. with p=0.50, a post-compression resize applied identically to both pair members;
6. with p=0.20, mild Gaussian noise on the degraded member only.

Both members are cropped at the same offset so the pair stays pixel-aligned.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

SUBSAMPLING_PIL = {"420": 2, "422": 1, "444": 0}
INTERP = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


@dataclass
class DegradationConfig:
    crop_size: int = 512
    p_pre_resize: float = 0.35
    pre_resize_range: tuple[float, float] = (0.5, 1.5)
    p_progressive: float = 0.10
    p_multipass: float = 0.40
    multipass_qf_delta: int = 15
    p_post_resize: float = 0.50
    post_resize_range: tuple[float, float] = (0.5, 2.0)
    p_noise: float = 0.20
    noise_sigma_max: float = 5.0 / 255.0
    subsampling_probs: dict = field(default_factory=lambda: {"420": 0.5, "422": 0.25, "444": 0.25})


def sample_qf(rng: np.random.Generator) -> int:
    r = rng.random()
    if r < 0.6:
        return int(rng.integers(30, 76))
    if r < 0.8:
        return int(rng.integers(1, 31))
    return int(rng.integers(75, 101))


def _resize(img: Image.Image, scale: float, interp: str) -> Image.Image:
    if abs(scale - 1.0) < 1e-6:
        return img
    w, h = img.size
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), INTERP[interp])


class DegradationSampler:
    """Stateful sampler holding the config and the RNG driving all randomness."""

    def __init__(self, config: DegradationConfig | None = None, seed: int | None = None):
        self.config = config or DegradationConfig()
        self.rng = np.random.default_rng(seed)

    def sample(self, clean_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg, rng = self.config, self.rng
        clean = Image.fromarray(clean_uint8)

        k = int(rng.integers(0, 4))
        if k:
            clean = clean.rotate(90 * k, expand=True)
        if rng.random() < 0.5:
            clean = clean.transpose(Image.FLIP_LEFT_RIGHT)

        if rng.random() < cfg.p_pre_resize:
            clean = _resize(clean, float(rng.uniform(*cfg.pre_resize_range)), "bicubic")

        qf = sample_qf(rng)
        subsampling = list(cfg.subsampling_probs)[
            int(rng.choice(len(cfg.subsampling_probs), p=np.array(list(cfg.subsampling_probs.values()))
                           / sum(cfg.subsampling_probs.values())))
        ]
        progressive = rng.random() < cfg.p_progressive
        arr = self._jpeg_roundtrip(clean, qf, subsampling, progressive)
        for _ in range(int(rng.integers(1, 3)) if rng.random() < cfg.p_multipass else 0):
            delta = int(rng.integers(-cfg.multipass_qf_delta, cfg.multipass_qf_delta + 1))
            qp = int(np.clip(qf + delta, 1, 100))
            arr = self._jpeg_roundtrip(Image.fromarray(arr), qp, subsampling, progressive)

        if rng.random() < cfg.p_post_resize:
            h, w = arr.shape[:2]
            need = cfg.crop_size
            scale = max(float(rng.uniform(*cfg.post_resize_range)), need / h, need / w)
            interp = str(rng.choice(list(INTERP)))
            arr = np.asarray(_resize(Image.fromarray(arr), scale, interp))
            clean = _resize(clean, scale, interp)

        # resizes can leave the pair smaller than the target crop; grow back
        h, w = arr.shape[:2]
        if h < cfg.crop_size or w < cfg.crop_size:
            up = max(cfg.crop_size / h, cfg.crop_size / w)
            arr = np.asarray(_resize(Image.fromarray(arr), up, "bicubic"))
            clean = _resize(clean, up, "bicubic")

        jpeg_arr = arr.astype(np.float32)
        if rng.random() < cfg.p_noise:
            sigma = float(rng.uniform(0, cfg.noise_sigma_max))
            jpeg_arr = np.clip(jpeg_arr + rng.normal(0, sigma * 255, jpeg_arr.shape), 0, 255)

        # aligned crop to exactly crop_size; the small [0,15] jitter varies the
        # 8x8 grid phase the network sees, like real-world crops do
        h, w = jpeg_arr.shape[:2]
        max_dy = max(0, min(15, h - cfg.crop_size))
        max_dx = max(0, min(15, w - cfg.crop_size))
        dy, dx = int(rng.integers(0, max_dy + 1)), int(rng.integers(0, max_dx + 1))
        box = (slice(dy, dy + cfg.crop_size), slice(dx, dx + cfg.crop_size))

        return jpeg_arr[box] / 255.0, np.asarray(clean, dtype=np.float32)[box] / 255.0

    @staticmethod
    def _jpeg_roundtrip(img: Image.Image, qf: int, subsampling: str, progressive: bool) -> np.ndarray:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(qf), subsampling=SUBSAMPLING_PIL[subsampling],
                 progressive=progressive)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))
