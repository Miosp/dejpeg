"""JPEG metadata parser tests (Task 0.3).

Two properties matter:
- correctness: extracted tables/sampling/flags match what the encoder wrote.
- cross-language identity: the Python parser and the TypeScript twin in
  inference-core emit byte-identical canonical JSON on the same fixture. This is
  load-bearing — if the browser derives different conditioning than training
  did, the model is fed something it never saw.
"""

from __future__ import annotations

import io
import struct
import subprocess

import pytest

from dejpeg.data.jpegmeta import parse_jpeg

PIL = pytest.importorskip("PIL")  # only present inside the dejpeg venv
from PIL import Image  # noqa: E402

# Standard libjpeg (Annex K) luma quantization table at QF 50, natural order.
# libjpeg-turbo (what Pillow wraps) uses the identical table. Anchor for the
# de-zigzag + correctness check.
ANNEX_K_QF50_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

QFS = [10, 30, 50, 75, 95]
# Pillow subsampling id -> expected (h, v) per component (Y, Cb, Cr).
SUBSAMPLING_EXPECT = {
    0: [(1, 1), (1, 1), (1, 1)],  # 4:4:4
    1: [(2, 1), (1, 1), (1, 1)],  # 4:2:2
    2: [(2, 2), (1, 1), (1, 1)],  # 4:2:0
}


def _make_image(size: int = 96) -> Image.Image:
    """Deterministic RGB image with smooth gradients and high-frequency detail
    so JPEG at low QF still produces a valid, non-trivial stream."""
    import numpy as np

    xs = np.arange(size, dtype=np.float32)
    gx = np.tile(xs, (size, 1))
    gy = gx.T
    noise = np.sin(gx * 0.7) * 32 + np.cos(gy * 0.5) * 32
    r = ((gx * 2.5) + noise) % 255
    g = ((gy * 2.5) - noise) % 255
    b = (((gx + gy) * 1.5)) % 255
    arr = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


def _save_fixture(path, qf: int, subsampling: int, progressive: bool) -> None:
    img = _make_image()
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=qf,
        subsampling=subsampling,
        progressive=progressive,
    )
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    """Generate the libjpeg-turbo fixture matrix into a temp dir.

    Returns a dict keyed by (qf, subsampling, progressive) -> path.
    """
    d = tmp_path_factory.mktemp("jpeg_fixtures")
    out = {}
    for qf in QFS:
        for sub, _ in SUBSAMPLING_EXPECT.items():
            for prog in (False, True):
                p = d / f"q{qf:03d}_s{sub}_{'prog' if prog else 'base'}.jpg"
                _save_fixture(str(p), qf, sub, prog)
                out[(qf, sub, prog)] = str(p)
    return out


def test_annex_k_qf50_luma_table_is_ground_truth(fixtures):
    """QF-50 baseline fixture: parser's natural-order luma table == standard
    Annex-K table. Validates de-zigzag + extraction against an absolute,
    encoder-convention-independent ground truth."""
    path = fixtures[(50, 2, False)]
    with open(path, "rb") as f:
        meta = parse_jpeg(f.read())
    luma_qt_id = meta.components[0].qt_id
    table = meta.quant_tables[luma_qt_id].values
    assert len(table) == 64
    assert table == ANNEX_K_QF50_LUMA


@pytest.mark.parametrize("sub", list(SUBSAMPLING_EXPECT))
def test_subsampling_and_component_count(fixtures, sub):
    path = fixtures[(50, sub, False)]
    with open(path, "rb") as f:
        meta = parse_jpeg(f.read())
    assert len(meta.components) == 3
    expect = SUBSAMPLING_EXPECT[sub]
    for comp, (h, v) in zip(meta.components, expect):
        assert (comp.h, comp.v) == (h, v), (comp, (h, v))


@pytest.mark.parametrize("prog", [False, True])
def test_progressive_flag(fixtures, prog):
    path = fixtures[(50, 2, prog)]
    with open(path, "rb") as f:
        meta = parse_jpeg(f.read())
    assert meta.progressive is prog
    assert meta.sof_marker == (0xC2 if prog else 0xC0)


def test_dimensions_and_precision(fixtures):
    path = fixtures[(75, 0, False)]
    with open(path, "rb") as f:
        meta = parse_jpeg(f.read())
    assert meta.precision == 8
    assert meta.width == 96 and meta.height == 96


def test_each_fixture_has_two_quant_tables(fixtures):
    """Color JPEG defines a luma (id 0) and chroma (id 1) quant table."""
    for key, path in fixtures.items():
        with open(path, "rb") as f:
            meta = parse_jpeg(f.read())
        tids = sorted(meta.quant_tables)
        assert tids == [0, 1], (key, tids)
        for qt in meta.quant_tables.values():
            assert len(qt.values) == 64
            assert qt.precision == 0  # 8-bit at these QFs


def _ts_canonical(bun: str, path: str) -> str:
    import os

    runner = os.path.join(os.path.dirname(__file__), "_jpegmeta_runner.ts")
    proc = subprocess.run(
        [bun, runner, path], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def test_cross_language_byte_identical(fixtures, bun):
    """Load-bearing: Python and TypeScript parsers emit byte-identical canonical
    JSON on every fixture. A divergence here means the browser would feed the
    model a conditioning vector it never saw in training."""
    failures = []
    for key, path in fixtures.items():
        with open(path, "rb") as f:
            py = parse_jpeg(f.read()).to_canonical_json()
        ts = _ts_canonical(bun, path)
        if py != ts:
            failures.append((key, py, ts))
    assert not failures, (
        f"{len(failures)} fixture(s) diverged; first:\n"
        f"key={failures[0][0]}\npy ={failures[0][1]}\nts ={failures[0][2]}"
    )


def test_not_a_jpeg_rejected():
    from dejpeg.data.jpegmeta import JpegParseError

    with pytest.raises(JpegParseError):
        parse_jpeg(b"PNG\r\n\x1a\n" + b"\x00" * 32)


def test_parser_stops_at_sos_does_not_touch_entropy():
    """A truncated stream (header intact, entropy cut) still parses metadata;
    only the scan body is missing."""
    img = _make_image(48)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40, subsampling=2)
    data = buf.getvalue()
    # Find SOS (FF DA) and cut shortly after it.
    sos = data.find(b"\xff\xda")
    assert sos > 0
    truncated = data[: sos + 12]
    meta = parse_jpeg(truncated)
    assert len(meta.components) == 3
