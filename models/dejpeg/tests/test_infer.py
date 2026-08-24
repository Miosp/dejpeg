"""End-to-end inference on synthetic JPEGs, plus the degradation pipeline."""
import io

import numpy as np
import torch
from PIL import Image

from dejpeg.degrade import DegradationSampler
from dejpeg.infer import restore_image


def synthetic_photo(seed: int = 0, size: int = 96) -> Image.Image:
    """Gradient + shapes: enough structure for JPEG to chew on."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.stack([
        (xx * 255 / size), (yy * 255 / size),
        127 + 100 * np.sin(xx / 7) * np.cos(yy / 9),
    ]).astype(np.uint8).transpose(1, 2, 0)
    for _ in range(4):
        x0, y0 = rng.integers(0, size - 20, 2)
        img[y0:y0 + 15, x0:x0 + 15] = rng.integers(0, 256, 3)
    return Image.fromarray(img)


def jpeg_roundtrip(img: Image.Image, qf: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=qf, subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_restore_improves_or_preserves_psnr_on_easy_case():
    """At QF95 the artifacts are tiny; restoration must not make things worse by much."""
    model = _tiny_model()
    gt = synthetic_photo()
    degraded = jpeg_roundtrip(gt, 95)
    out = restore_image(degraded, model=model, sharpness=0.10)
    a = np.asarray(gt, dtype=np.float64)
    d = np.asarray(degraded, dtype=np.float64)
    o = np.asarray(out, dtype=np.float64)

    def psnr(x):
        return 10 * np.log10(1.0 / max(np.mean((a - x) ** 2 / 255**2), 1e-12))

    assert psnr(o) > psnr(d) - 0.5


def test_restore_preserves_size_and_range():
    model = _tiny_model()
    for size in [(61, 87), (128, 128)]:
        img = synthetic_photo().resize(size)
        out = restore_image(jpeg_roundtrip(img, 30), model=model)
        assert out.size == size
        arr = np.asarray(out)
        assert arr.dtype == np.uint8 and arr.min() >= 0 and arr.max() <= 255


def test_sharpness_changes_output():
    model = _tiny_model()
    degraded = jpeg_roundtrip(synthetic_photo(), 40)
    mild = np.asarray(restore_image(degraded, model=model, sharpness=0.10), dtype=np.float32)
    off = np.asarray(restore_image(degraded, model=model, sharpness=0.0), dtype=np.float32)
    assert np.abs(mild - off).mean() > 0.2  # uint8 scale: visible but gentle difference


def test_degradation_sampler_produces_aligned_pairs():
    sampler = DegradationSampler(seed=3)
    clean = np.asarray(synthetic_photo(size=512))
    jpeg, target = sampler.sample(clean)
    assert jpeg.shape == target.shape == (512, 512, 3)
    assert jpeg.dtype == target.dtype == np.float32
    assert 0.0 <= jpeg.min() and jpeg.max() <= 1.0
    # pairs are aligned but distinct: degradation must change the image
    assert np.abs(jpeg - target).mean() > 1e-4
    # deterministic under the same seed
    again = DegradationSampler(seed=3).sample(clean)
    np.testing.assert_array_equal(jpeg, again[0])


def _tiny_model():
    from dejpeg.model import DeJPEGNet

    model = DeJPEGNet(c0=8).eval()
    with torch.no_grad():  # give the head a tiny non-zero output so it "does something"
        model.head.weight.normal_(0, 1e-3)
    return model
