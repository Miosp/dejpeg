# convert-fbcnn

One-shot conversion tool: takes original FBCNN `.pth` weights and emits a
web-ready FP16 ONNX artifact plus a TS stub for paste into `inference-core`.

## Prerequisites

From repo root:

```bash
uv sync --all-packages
```

You also need the original FBCNN weights from
<https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0>:

```bash
mkdir -p /tmp/fbcnn-weights
cd /tmp/fbcnn-weights
curl -LO https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth
curl -LO https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray.pth
curl -LO https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray_double.pth
```

## Convert one variant

```bash
cd tools/convert-fbcnn
uv run python convert.py \
    --variant fbcnn-color-real \
    --weights /tmp/fbcnn-weights/fbcnn_color.pth \
    --out ../../apps/web/public/models/
```

This will:

1. Build the modern ported network
2. Load original weights strict
3. Export to ONNX (opset 17, dynamic H/W axes)
4. Validate via shape inference + onnx.checker
5. Quantize weights to FP16 (halves file size; native on WebGPU)
6. Smoke test: torch vs ONNX Runtime CPU within 1e-2 max abs
7. Print a TS stub: paste it into a new file under `packages/inference-core/src/models/`

## Convert all variants

```bash
cd tools/convert-fbcnn
for variant in fbcnn-color-real fbcnn-gray fbcnn-gray-double; do
    case $variant in
        fbcnn-color-real)   weights=/tmp/fbcnn-weights/fbcnn_color.pth ;;
        fbcnn-gray)         weights=/tmp/fbcnn-weights/fbcnn_gray.pth ;;
        fbcnn-gray-double)  weights=/tmp/fbcnn-weights/fbcnn_gray_double.pth ;;
    esac
    uv run python convert.py --variant $variant --weights $weights --out ../../apps/web/public/models/
done
```

## Validation

- `onnx.checker.check_model` after shape inference: structural validity
- `test_weights_compat.py` in fbcnn-py: strict key-for-key weight compatibility
- `test_psr_psnr.py` in fbcnn-py: PSNR parity on Classic5 (paper tolerance ±0.2 dB)
- `smoke_test.py` here: torch vs ONNX Runtime output within FP16 tolerance

## When to re-run

- Adding a new variant: define it in `packages/fbcnn-py/src/fbcnn/config.py`, add to `VARIANTS`, run this tool
- Updating the FBCNN port: re-run all variants and re-verify smoke
- Updating PyTorch or ONNX opset: re-run all variants; investigate any smoke failures

Manual cadence by design: conversion is rare and benefits from human review of the smoke output.
