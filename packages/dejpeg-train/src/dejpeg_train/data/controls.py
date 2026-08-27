"""Control samples (spec §2.4): 10% of every batch.

  * passthrough: clean PNG, target == input (teaches near-identity / no-op).
  * webp:        WebP-compressed input vs clean target (non-JPEG artifact family).
  * gray_jpeg:   grayscale JPEG replicated to 3 channels vs RGB target.

true_qf is fixed at 100 (controls are not stratified). The model must learn to do
nothing on clean/already-good inputs and to handle non-JPEG artifacts.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .degrade import DegradationRecord, _sample_categorical


@dataclass
class ControlConfig:
    probs: dict = field(default_factory=lambda: {"passthrough": 0.4, "webp": 0.3, "gray_jpeg": 0.3})
    webp_quality_range: tuple[int, int] = (40, 90)
    gray_qf_range: tuple[int, int] = (20, 80)


def _control_record(kind: str) -> DegradationRecord:
    return DegradationRecord(
        qf=[100], subsampling="444", progressive=False, passes=0,
        pre_resize_scale=None, post_resize_scale=None, post_resize_interp=None,
        quant_tables={}, grid_offset=(0, 0), grid_phase=(0, 0),
        noise_sigma=0.0, true_qf=100, is_control=True, control_kind=kind,
    )


class ControlSampler:
    def __init__(self, config: ControlConfig | None = None, seed: int | None = None):
        self.config = config or ControlConfig()
        self.rng = np.random.default_rng(seed)

    def sample(self, clean_uint8_hwc: np.ndarray):
        cfg = self.config
        rng = self.rng
        kind = _sample_categorical(rng, cfg.probs)
        target = clean_uint8_hwc.astype(np.float32) / 255.0

        if kind == "passthrough":
            arr = target.copy()
            return arr, target, _control_record("passthrough")

        if kind == "webp":
            img = Image.fromarray(clean_uint8_hwc)
            q = int(rng.integers(*cfg.webp_quality_range))
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=q)
            arr = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
            return arr, target, _control_record("webp")

        # gray_jpeg
        gray = Image.fromarray(clean_uint8_hwc).convert("L")
        q = int(rng.integers(*cfg.gray_qf_range))
        buf = io.BytesIO()
        gray.save(buf, format="JPEG", quality=q)
        dec = np.array(Image.open(buf).convert("L"))
        arr = np.stack([dec] * 3, axis=-1).astype(np.float32) / 255.0
        return arr, target, _control_record("gray_jpeg")
