# DeJPEGNet

Compact perceptual JPEG artifact removal. A 2.63M-parameter activation-free U-Net
that removes blocking, ringing and banding from JPEG-compressed images while
keeping texture natural, trained with nothing but a folder of clean images.

Shipped weights (`dejpeg-c40-fp16.pt`, ~5 MB FP16) are distributed as a
GitHub Release asset; `load_model()` downloads them to `~/.cache/dejpeg/` on
first use, so inference works out of the box.

## Results

Restoration scored against pristine originals on the standard Classic5 and
LIVE1 benchmarks (JPEG re-encoded at each quality factor with 4:2:0 chroma
subsampling; metrics via `scripts/evaluate.py`; FBCNN re-run under the identical
protocol). Lower is better for DISTS/LPIPS, higher for PSNR.

DISTS (perceptual distance):

| Suite | QF | Input | FBCNN (35.7M) | **DeJPEGNet (2.6M)** |
|---|---|---|---|---|
| Classic5 | 10 | 0.2412 | 0.1938 | **0.1191** |
| Classic5 | 20 | 0.1536 | 0.1597 | **0.0911** |
| Classic5 | 30 | 0.1215 | 0.1405 | **0.0743** |
| Classic5 | 40 | 0.1054 | 0.1268 | **0.0642** |
| LIVE1 | 10 | 0.2387 | 0.1855 | **0.1292** |
| LIVE1 | 20 | 0.1653 | 0.1372 | **0.0895** |
| LIVE1 | 30 | 0.1315 | 0.1126 | **0.0699** |
| LIVE1 | 40 | 0.1126 | 0.0973 | **0.0602** |

LPIPS follows the same pattern (e.g. Classic5 QF10: 0.1804 vs FBCNN's 0.2594).
PSNR is where FBCNN keeps a ~1 dB distortion-metric lead (Classic5 QF10:
29.80 vs our 28.81); this model optimizes for how images *look*, not pixel
averages. A built-in unsharp post-process (exposed as `--sharpness`, default
on) buys back some of that punch at negligible perceptual cost.

## Weights

The released FP16 checkpoint lives as a GitHub Release asset
(`dejpeg-c40-fp16.pt`, ~5 MB). Resolution order in `load_model()`:

1. explicit `weights_path=` argument
2. `$DEJPEG_WEIGHTS` (path to a local file)
3. `~/.cache/dejpeg/dejpeg-c40-fp16.pt` (downloaded from the release on
   first use; override the URL with `$DEJPEG_WEIGHTS_URL` or the cache
   directory with `$DEJPEG_CACHE_DIR`)

## Install

```bash
pip install -e .                 # inference only (torch, numpy, pillow)
pip install -e ".[train]"        # + training (lpips)
pip install -e ".[export]"       # + ONNX export
```

## Usage

Python:

```python
from dejpeg import load_model, restore_image

model = load_model()                       # release weights (auto-downloaded), CPU or CUDA
restore_image("input.jpg", "restored.png", model)
```

CLI:

```bash
dejpeg restore photo.jpg restored.png --sharpness 0.1
dejpeg restore photos_folder/ restored_folder/
```

Train from your own clean images (any folder of jpg/png; degradations are
synthesized on the fly):

```bash
python scripts/train.py --data ~/datasets/clean-images --out runs/v1 --iters 150000
```

Export to FP16 ONNX (reparameterized, simplified):

```bash
python scripts/export_onnx.py --out dejpeg.onnx --dynamic
```

Benchmark a checkpoint:

```bash
python scripts/evaluate.py --suite data/benchmarks/classic5 --qf 10 20 30 40
# expects ground-truth images in the suite folder; add .[eval] for LPIPS/DISTS
```

## How it works

The network predicts a *residual* on top of its input (zero-initialized output
head), so it learns only corrections and never degrades what it doesn't
understand. Architecture:

- **NAFNet-style blocks**: LayerNorm → 1×1 conv → depthwise 3×3 → SimpleGate
  (channel split + multiply as the only nonlinearity) → channel attention.
- **Local channel attention**: gates pool over fixed 32px windows, never the
  whole image. Changing pixels far away cannot affect nearby outputs
  (enforced by test), which is what makes tiled large-image inference
  consistent.
- **Reparameterizable depthwise convs**: parallel {3×3, 1×1, identity} branches
  during training fold into one depthwise conv for deployment.
- Strided convs down, PixelShuffle up, skip connections throughout.
  Inputs are reflect-padded to a multiple of 32; any image size works.

## Training recipe notes

Things the recipe gets right beyond the standard loop (each discovered the
hard way):

- **Identity anchor** (`--identity-frac 0.1`): a fraction of pairs are left
  undegraded so the model learns when *not* to act.
- **Grayscale anchor** (`--gray-frac 0.25`): without it the model hallucinates
  color on grayscale / flat-chroma content; it had learned "JPEG artifacts
  are colorful". Anchoring on desaturated pairs removes chroma hallucination
  entirely while leaving color restoration intact elsewhere.
- **LPIPS on a random 128px crop**: full-patch VGG-LPIPS under autocast is a
  NaN source; the crop fixes both memory and stability.
- **EMA weights** (`decay 0.999`) are evaluated and shipped, never raw weights.
- bf16 autocast (never fp16: fp16 + LPIPS NaNs), AdamW β=(0.9, 0.9),
  grad-clip 1.0, warmup + cosine to zero.

## Development

```bash
pip install -e ".[dev]"
pytest        # 13 tests: architecture locks, fusion parity, tile behavior,
              # degradation determinism, end-to-end restore
ruff check src scripts tests
```

CI runs both on CPU-only PyTorch.

## License

Apache-2.0 (see the repository root `LICENSE`). The Classic5/LIVE1 benchmark
images are not included; obtain them from their original sources if you want
to reproduce the tables above.
