"""Phase 0 Task 0.6 -- degradation sampler tests."""
from __future__ import annotations

import numpy as np

from dejpeg_train.data.controls import ControlConfig, ControlSampler
from dejpeg_train.data.degrade import DegradationConfig, DegradationSampler, sample_qf


def make_clean(seed: int = 0, size: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # smooth gradient + noise so JPEG has real structure to attack
    base = np.linspace(0, 255, size, dtype=np.float32)
    img = (base[None, :] + base[:, None]) / 2
    img = np.stack([img, img, img], axis=-1)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255)
    return img.astype(np.uint8)


def test_degrade_deterministic_under_seed():
    clean = make_clean()
    j1, c1, r1 = DegradationSampler(seed=42).sample(clean)
    j2, c2, r2 = DegradationSampler(seed=42).sample(clean)
    assert np.array_equal(j1, j2), "jpeg patch not reproducible under fixed seed"
    assert np.array_equal(c1, c2), "clean target not reproducible under fixed seed"
    assert r1.qf == r2.qf
    assert r1.subsampling == r2.subsampling
    assert r1.grid_offset == r2.grid_offset
    assert r1.passes == r2.passes


def test_record_fields_in_range_and_grid_recoverable():
    sampler = DegradationSampler(seed=1)
    for _ in range(12):
        jpeg, clean, r = sampler.sample(make_clean())
        assert all(1 <= q <= 100 for q in r.qf)
        assert 0 <= r.grid_offset[0] <= 15 and 0 <= r.grid_offset[1] <= 15
        assert r.grid_phase == (r.grid_offset[0] % 8, r.grid_offset[1] % 8)
        assert r.passes >= 1 and len(r.qf) == r.passes
        assert jpeg.shape == (512, 512, 3) and clean.shape == (512, 512, 3)
        assert jpeg.dtype == np.float32 and clean.dtype == np.float32
        assert float(jpeg.min()) >= 0.0 and float(jpeg.max()) <= 1.0


def test_quant_tables_recorded_from_emitted_jpeg():
    jpeg, clean, r = DegradationSampler(seed=2).sample(make_clean())
    assert len(r.quant_tables) >= 1, "actual quant table not parsed into record"
    first = next(iter(r.quant_tables.values()))
    assert len(first["values"]) == 64


def test_degradation_actually_changes_pixels():
    sampler = DegradationSampler(DegradationConfig(p_pre_resize=0, p_post_resize=0, p_multipass=0, p_noise=0), seed=5)
    jpeg, clean, r = sampler.sample(make_clean())
    assert not np.array_equal(jpeg, clean), "no degradation applied"
    assert r.is_control is False


def test_qf_distribution_covers_all_bins():
    rng = np.random.default_rng(0)
    qfs = [sample_qf(rng) for _ in range(3000)]
    low = sum(1 for q in qfs if q <= 30)
    mid = sum(1 for q in qfs if 30 < q < 75)
    high = sum(1 for q in qfs if q >= 75)
    assert low > 0 and mid > 0 and high > 0
    assert mid > low and mid > high  # majority in 30-75


def test_controls_passthrough_target_equals_input():
    sampler = ControlSampler(ControlConfig(probs={"passthrough": 1.0}), seed=3)
    arr, target, rec = sampler.sample(make_clean())
    assert np.array_equal(arr, target)
    assert rec.is_control and rec.control_kind == "passthrough" and rec.true_qf == 100


def test_controls_webp_and_gray_produce_distinct_inputs():
    for kind, probs in [("webp", {"webp": 1.0}), ("gray_jpeg", {"gray_jpeg": 1.0})]:
        sampler = ControlSampler(ControlConfig(probs=probs), seed=4)
        arr, target, rec = sampler.sample(make_clean())
        assert rec.control_kind == kind
        assert arr.shape == target.shape == (640, 640, 3)
        assert rec.is_control
