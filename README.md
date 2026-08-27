# dejpeg

JPEG artifact removal that runs entirely in your browser. Photos never leave
the device: inference runs client-side via ONNX Runtime Web (WebGPU
primary, WASM fallback), backed by a compact custom-trained model.

**Try it: <https://dejpeg.ludwina.top>**

## What's here

| Path | What |
|---|---|
| `apps/web/` | Astro + Svelte 5 web app (the tool itself) |
| `packages/inference-core/` | Framework-free TypeScript inference library: worker, codecs, adaptive tiling |
| `packages/fbcnn-py/` | Modern PyTorch port of [FBCNN](https://github.com/jiaxi-jiang/FBCNN) |
| `packages/dejpeg-train/` | Research training pipeline: WebDataset corpora, synthetic degradation, dedup manifest |
| `tools/convert-fbcnn/` | FBCNN PyTorch → ONNX FP16 conversion tool |
| `models/dejpeg/` | DeJPEGNet: the shipped model (training, inference, ONNX export; weights via GitHub Releases) |

## Quickstart

Web app (bun):

```bash
bun install
bun run dev          # http://localhost:4321/tool
```

Python packages (uv):

```bash
uv sync                       # fbcnn-py + tools (workspace)
pip install -e models/dejpeg  # shipped model package
```

Model inference:

```python
from dejpeg import load_model, restore_image

model = load_model()                  # release weights (auto-downloaded to ~/.cache/dejpeg)
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
