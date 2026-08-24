"""Phase 0 Task 0.5 -- synthetic generator tests.

Determinism + losslessness are the load-bearing properties: the generator must
reproduce byte-identically under a fixed seed and survive lossless shard storage.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dejpeg.data.synthetic import SyntheticGenerator


@pytest.fixture(scope="module")
def gen():
    return SyntheticGenerator(allow_html=False)


def test_shape_dtype(gen):
    arr = gen.generate(42)
    assert arr.shape == (640, 640, 3)
    assert arr.dtype == np.uint8


def test_deterministic_under_fixed_seed(gen):
    a = gen.generate(123)
    b = gen.generate(123)
    assert np.array_equal(a, b), "same seed must produce byte-identical output"


def test_different_seeds_differ(gen):
    a = gen.generate(1000)
    b = gen.generate(2000)
    assert not np.array_equal(a, b), "different seeds unexpectedly produced identical output"


def test_lossless_png_roundtrip(gen):
    arr = gen.generate(7)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    back = np.array(Image.open(buf).convert("RGB"))
    assert np.array_equal(back, arr), "PNG round-trip must be bit-identical"


def test_lossless_webp_roundtrip(gen):
    arr = gen.generate(8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="WEBP", lossless=True)
    buf.seek(0)
    back = np.array(Image.open(buf).convert("RGB"))
    assert np.array_equal(back, arr), "lossless WebP round-trip must be bit-identical"


def test_all_generators_produce_valid_nonblank_images(gen):
    for name, fn in gen._dispatch.items():
        rng = np.random.RandomState(99)
        arr = fn(rng)
        arr = gen._finalize(arr)
        assert arr.shape == (640, 640, 3), f"{name} wrong shape {arr.shape}"
        assert arr.dtype == np.uint8, f"{name} wrong dtype {arr.dtype}"
        assert arr.std() > 1.0, f"{name} output near-blank (std {arr.std():.2f})"


def test_html_generator_disabled_by_default():
    g = SyntheticGenerator(allow_html=False)
    assert "html" not in g.names
