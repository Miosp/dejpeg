# fbcnn-py

Modern PyTorch port of [FBCNN](https://github.com/jiaxi-jiang/FBCNN),
*Towards Flexible Blind JPEG Artifacts Removal* (ICCV 2021).

Originally authored by Jiaxi Jiang, Kai Zhang, Radu Timofte. Apache-2.0.
This port modernizes the implementation while keeping the architecture
and weight compatibility strict.

## What this package provides

- `fbcnn.FBCNN`: modernized network class
- `fbcnn.blocks.{ResBlock, QFAttention, conv, sequential, downsample_strideconv, upsample_convtranspose}`: building blocks
- `fbcnn.config.{Variant, COLOR_REAL, GRAY, GRAY_DOUBLE, VARIANTS, VARIANTS_BY_ID, build_model}`: variant configurations
- `fbcnn.weights.{load_original_state_dict, load_pretrained}`: strict weight loader (the contract)
- `fbcnn.inference.run`: high-level single-image helper

## Modernization summary

- Architecture preserved exactly: same layers, same channels, original weights load strict
- `functional.relu` instead of in-place `relu_` (in-place breaks ONNX export)
- `@dataclass` Variant configs instead of JSON files
- Type hints throughout
- No data-dependent control flow in `forward` (compatible with `torch.compile` if ever needed)
- Only the downsample/upsample modes used by shipped variants are ported (`strideconv`, `convtranspose`); others raise `NotImplementedError`

## Tests

From repo root:

```bash
# Always-run tests (no weights needed)
uv run pytest packages/fbcnn-py/tests/test_shapes.py

# Weight-dependent tests (skip cleanly without env vars)
FBCNN_WEIGHTS_DIR=/path/to/fbcnn-weights uv run pytest packages/fbcnn-py/tests/test_weights_compat.py

# PSNR parity (needs both weights and Classic5 test images)
git clone https://github.com/jiaxi-jiang/FBCNN /tmp/fbcnn-upstream
FBCNN_WEIGHTS_DIR=/path/to/fbcnn-weights \
FBCNN_TESTSETS_DIR=/tmp/fbcnn-upstream/testsets \
uv run pytest packages/fbcnn-py/tests/test_psr_psnr.py
```

## License

Apache-2.0, matching upstream.
