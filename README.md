# dejpeg

JPEG artifact removal that runs entirely in your browser. Photos never leave
the device: inference runs client-side via ONNX Runtime Web (WebGPU
primary, WASM fallback), backed by a compact custom-trained model.

**Try it: <https://dejpeg.ludwina.top>**

## Models

| Model | Size | Best for |
|---|---|---|
| DeJPEGNet v2 (default) | 15.4 MB | Staying close to the original: beats FBCNN on PSNR, LPIPS and DISTS at every tested quality |
| DeJPEGNet v2 + Texture | 15.3 MB | v2 with a slider for added grain, which is synthetic and not recovered detail |
| DeJPEGNet v1 | 5.4 MB | Smaller download, tuned for perceptual quality |

LIVE1, PSNR (dB) / LPIPS / DISTS, one pass, quality unknown to the model:

| | QF10 | QF20 | QF30 | QF40 |
|---|---|---|---|---|
| FBCNN | 27.77 / 0.250 / 0.179 | 30.11 / 0.158 / 0.132 | 31.43 / 0.119 / 0.108 | 32.34 / 0.097 / 0.094 |
| v2 | 28.05 / 0.239 / 0.174 | 30.37 / 0.152 / 0.129 | 31.70 / 0.113 / 0.106 | 32.61 / 0.092 / 0.091 |
| v2 + Texture at 0.5 | 28.04 / 0.228 / 0.158 | 30.33 / 0.142 / 0.112 | 31.65 / 0.104 / 0.089 | 32.54 / 0.083 / 0.075 |
| v2 + Texture at 1.0 | 27.97 / 0.210 / 0.137 | 30.23 / 0.125 / 0.091 | 31.50 / 0.088 / 0.069 | 32.35 / 0.068 / 0.057 |

v2 takes about 55 ms per 512x512 tile in Edge on WebGPU on one NVIDIA GPU. Weights
are in the [releases](https://github.com/Miosp/dejpeg/releases).

## What's here

| Path | What |
|---|---|
| `apps/web/` | Astro + Svelte 5 web app (the tool itself) |
| `packages/inference-core/` | Framework-free TypeScript inference library: worker, codecs, adaptive tiling |
| `packages/fbcnn-py/` | Modern PyTorch port of [FBCNN](https://github.com/jiaxi-jiang/FBCNN) |
| `packages/dejpeg-train/` | Research training pipeline: WebDataset corpora, synthetic degradation, dedup manifest |
| `tools/convert-fbcnn/` | FBCNN PyTorch → ONNX FP16 conversion tool |
| `models/dejpeg/` | DeJPEGNet v1: training, inference, ONNX export (weights via GitHub Releases). The v2 training code is not in this repository |

## Quickstart

Web app (bun):

```bash
bun install
bun run dev          # http://localhost:4321
```

Python packages (uv):

```bash
uv sync                       # fbcnn-py + tools (workspace)
pip install -e models/dejpeg  # shipped model package
```

Model inference:

```python
from dejpeg import load_model, restore_image

model = load_model()                  # v1 release weights (auto-downloaded to ~/.cache/dejpeg)
restore_image("input.jpg", "out.png", model)
```

Training from your own images:

```bash
python models/dejpeg/scripts/train.py --data ~/datasets/clean-images --out runs/v1
```

The full research pipeline (`packages/dejpeg-train/`) builds WebDataset shard
corpora from DIV2K/Flickr2K-class sources and synthesizes JPEG degradations on
the fly; see its README.

## Development

```bash
bun test               # inference-core unit tests
uv run pytest packages/fbcnn-py
pytest models/dejpeg   # model architecture/inference tests
```

## License

Apache-2.0. FBCNN-derived code (`packages/fbcnn-py/`) originates from an
Apache-2.0 upstream. Benchmark images (Classic5, LIVE1) are not distributed.
