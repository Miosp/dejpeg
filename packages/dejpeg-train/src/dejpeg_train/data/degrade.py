"""Degradation sampler (spec §2.4).

Per sample: clean 640x640 patch -> (jpeg_patch 512x3, clean_target 512x3, record).

Steps: flip/rot90; p=0.35 pre-resize; QF sample; JPEG encode (variable subsampling
+ quant table + progressive); p=0.4 multi-pass; p=0.5 post-compression resize
(clean resized identically so the pair stays aligned); grid-phase crop at random
(dx,dy) in [0,15]^2; p=0.2 mild noise. Records the FULL degradation record so the
oracle (Phase 1.3) can condition on ground truth and blockiness can use the grid
phase. true_qf is for stratified sampling + auxiliary QF regression ONLY -- never
a conditioning signal.

Encoder: PIL (libjpeg-turbo), deterministic, supports progressive + custom qtables.
simplejpeg / nvjpeg fast paths are benchmarked separately; the sampler's CPU path
is the Phase-0 reference.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import simplejpeg
from PIL import Image

from .jpegmeta import parse_jpeg

SUBSAMPLING_PIL = {"420": 2, "422": 1, "444": 0}
INTERP_PIL = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


@dataclass
class DegradationRecord:
    qf: list[int]
    subsampling: str
    progressive: bool
    passes: int
    pre_resize_scale: float | None
    post_resize_scale: float | None
    post_resize_interp: str | None
    quant_tables: dict            # actual tables parsed from the final emitted JPEG
    grid_offset: tuple[int, int]  # (dx, dy) crop offset
    grid_phase: tuple[int, int]   # (dx%8, dy%8) -- exact when no post-resize
    noise_sigma: float
    true_qf: int                  # stratification + aux QF-reg only, NOT conditioning
    table_source: str = "annex_k"
    is_control: bool = False
    control_kind: str | None = None


@dataclass
class DegradationConfig:
    crop_size: int = 512
    p_pre_resize: float = 0.35
    pre_resize_range: tuple[float, float] = (0.5, 1.5)
    p_progressive: float = 0.1
    p_multipass: float = 0.4
    multipass_qf_delta: int = 15
    p_post_resize: float = 0.5
    post_resize_range: tuple[float, float] = (0.5, 2.0)
    p_noise: float = 0.2
    noise_sigma_max: float = 5.0 / 255.0
    subsampling_probs: dict = field(default_factory=lambda: {"420": 0.5, "422": 0.25, "444": 0.25})
    table_source_probs: dict = field(
        default_factory=lambda: {"annex_k": 0.6, "mozjpeg": 0.25, "flat": 0.15}
    )


def sample_qf(rng: np.random.Generator, lo: int | None = None, hi: int | None = None) -> int:
    if lo is not None and hi is not None:
        # ranged: uniform within [lo, hi] (for QF-stratified sourcing)
        return int(rng.integers(max(1, lo), min(101, hi + 1)))
    r = rng.random()
    if r < 0.6:
        return int(rng.integers(30, 76))     # majority 30-75
    if r < 0.8:
        return int(rng.integers(1, 31))      # low 1-30
    return int(rng.integers(75, 101))        # high 75-100


def _flat_qtable(qf: int) -> list[list[int]]:
    scale = 5000 / qf if qf < 50 else 200 - 2 * qf
    v = int(np.clip(round(50 * scale / 100), 1, 255))
    return [[v] * 64]  # single table applied to all components


def encode_jpeg(
    img: Image.Image, qf: int, subsampling: str, table_source: str, progressive: bool
) -> bytes:
    """JPEG-encode.

    Common path (annex_k + baseline, ~85% of samples) goes through simplejpeg
    (libjpeg-turbo direct binding) -- robust against the Pillow 12 tiled-JPEG
    regression that crashes on edge-case content. Exotic paths (custom flat
    qtables, progressive) use PIL, then a real temp file (Pillow's primary path
    has a fileno), then finally simplejpeg baseline. The emitted table is parsed
    back from bytes for the record, so a fallback that drops flat/progressive
    keeps the conditioning consistent.
    """
    arr = np.asarray(img)
    if table_source != "flat" and not progressive:
        return simplejpeg.encode_jpeg(
            arr, quality=int(qf), colorspace="RGB",
            colorsubsampling=subsampling, fastdct=True,
        )
    kw = dict(
        format="JPEG",
        quality=int(qf),
        subsampling=SUBSAMPLING_PIL[subsampling],
        progressive=bool(progressive),
        optimize=False,
    )
    if table_source == "flat":
        kw["qtables"] = _flat_qtable(qf)
    try:
        buf = io.BytesIO()
        img.save(buf, **kw)
        return buf.getvalue()
    except OSError:
        pass
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.close(fd)
        img.save(path, **kw)
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return simplejpeg.encode_jpeg(
            arr, quality=int(qf), colorspace="RGB",
            colorsubsampling=subsampling, fastdct=True,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def decode_jpeg(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def _resize_pil(img: Image.Image, scale: float, interp: str) -> Image.Image:
    if abs(scale - 1.0) < 1e-6:
        return img
    w, h = img.size
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), INTERP_PIL[interp])


def _resize_arr(arr: np.ndarray, scale: float, interp: str) -> np.ndarray:
    img = Image.fromarray(arr)
    out = _resize_pil(img, scale, interp)
    return np.array(out)


def _augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    k = int(rng.integers(0, 4))  # 0..3 rotations
    if k:
        img = img.rotate(90 * k, expand=True)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _sample_categorical(rng: np.random.Generator, probs: dict) -> str:
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=float)
    p = p / p.sum()
    return keys[int(rng.choice(len(keys), p=p))]


class DegradationSampler:
    def __init__(self, config: DegradationConfig | None = None, seed: int | None = None):
        self.config = config or DegradationConfig()
        self.rng = np.random.default_rng(seed)

    def sample(self, clean_uint8_hwc: np.ndarray, qf_range: tuple[int, int] | None = None):
        cfg = self.config
        rng = self.rng
        clean_img = Image.fromarray(clean_uint8_hwc)
        clean_img = _augment(clean_img, rng)

        pre_scale: float | None = None
        if rng.random() < cfg.p_pre_resize:
            pre_scale = float(rng.uniform(*cfg.pre_resize_range))
            clean_img = _resize_pil(clean_img, pre_scale, "bicubic")

        if qf_range is not None:
            qf = sample_qf(rng, qf_range[0], qf_range[1])
        else:
            qf = sample_qf(rng)
        qfs = [qf]
        subsampling = _sample_categorical(rng, cfg.subsampling_probs)
        table_source = _sample_categorical(rng, cfg.table_source_probs)
        progressive = rng.random() < cfg.p_progressive

        data = encode_jpeg(clean_img, qf, subsampling, table_source, progressive)
        arr = decode_jpeg(data)
        passes = 1

        if rng.random() < cfg.p_multipass:
            n_extra = int(rng.integers(1, 3))  # 1 or 2 extra passes
            for _ in range(n_extra):
                qp = int(
                    np.clip(
                        qf + int(rng.integers(-cfg.multipass_qf_delta, cfg.multipass_qf_delta + 1)),
                        1,
                        100,
                    )
                )
                qfs.append(qp)
                data = encode_jpeg(Image.fromarray(arr), qp, subsampling, table_source, progressive)
                arr = decode_jpeg(data)
                passes += 1

        meta = parse_jpeg(data)  # actual tables of the final emitted JPEG

        post_scale: float | None = None
        post_interp: str | None = None
        if rng.random() < cfg.p_post_resize:
            h, w = arr.shape[:2]
            need = cfg.crop_size + 15
            min_scale = max(need / h, need / w)
            post_scale = max(float(rng.uniform(*cfg.post_resize_range)), min_scale)
            post_interp = str(rng.choice(["bilinear", "bicubic", "lanczos"]))
            arr = _resize_arr(arr, post_scale, post_interp)
            clean_img = _resize_pil(clean_img, post_scale, post_interp)

        # guarantee the image is large enough for a crop_size + 15 patch (pre-resize
        # alone, or a small combined scale, can otherwise undershoot)
        h, w = arr.shape[:2]
        need = cfg.crop_size + 15
        if h < need or w < need:
            up = max((need + 0.0) / h, (need + 0.0) / w)
            arr = _resize_arr(arr, up, "bicubic")
            clean_img = _resize_pil(clean_img, up, "bicubic")

        h, w = arr.shape[:2]
        max_dx = max(0, min(15, w - cfg.crop_size))
        max_dy = max(0, min(15, h - cfg.crop_size))
        dx = int(rng.integers(0, max_dx + 1))
        dy = int(rng.integers(0, max_dy + 1))

        jpeg_patch = arr[dy : dy + cfg.crop_size, dx : dx + cfg.crop_size].astype(np.float32)
        clean_np = np.array(clean_img)
        clean_target = clean_np[dy : dy + cfg.crop_size, dx : dx + cfg.crop_size].astype(np.float32)

        noise_sigma = 0.0
        if rng.random() < cfg.p_noise:
            noise_sigma = float(rng.uniform(0, cfg.noise_sigma_max))
            jpeg_patch = jpeg_patch + rng.normal(0.0, noise_sigma, jpeg_patch.shape).astype(np.float32)
            jpeg_patch = np.clip(jpeg_patch, 0, 255)

        record = DegradationRecord(
            qf=qfs,
            subsampling=subsampling,
            progressive=progressive,
            passes=passes,
            pre_resize_scale=pre_scale,
            post_resize_scale=post_scale,
            post_resize_interp=post_interp,
            quant_tables={
                int(k): {"precision": v.precision, "values": list(v.values)}
                for k, v in meta.quant_tables.items()
            },
            grid_offset=(dx, dy),
            grid_phase=(dx % 8, dy % 8),
            noise_sigma=noise_sigma,
            true_qf=qf,
            table_source=table_source,
        )
        # normalize to [0,1]
        return jpeg_patch / 255.0, clean_target / 255.0, record
