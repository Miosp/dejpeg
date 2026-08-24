"""Convert ported FBCNN to web-ready ONNX.

Usage:
    uv run python tools/convert-fbcnn/convert.py \
        --variant fbcnn-color-real \
        --weights /path/to/fbcnn_color.pth \
        --out apps/web/public/models/

Produces <out>/fbcnn-<variant>.onnx (FP16 quantized) and prints a TS stub
matching the model definition format expected by inference-core.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
from onnxconverter_common import float16

from fbcnn import VARIANTS_BY_ID, build_model, load_original_state_dict
from fbcnn.config import Variant
from smoke_test import smoke_test


def build_and_load(variant: Variant, weights_pth: Path) -> torch.nn.Module:
    net = build_model(variant)
    load_original_state_dict(net, weights_pth)
    net.eval()
    return net


def export_onnx(net: torch.nn.Module, variant: Variant, out_path: Path) -> None:
    """Export with dynamic batch/h/w so any tile size works."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_image = torch.randn(1, variant.in_nc, 256, 256)
    dummy_qf = torch.tensor([[0.6]])  # 1 - qf/100, here qf=40
    torch.onnx.export(
        net,
        (dummy_image, dummy_qf),
        str(out_path),
        input_names=["input", "qf_input"],
        output_names=["output", "qf_predicted"],
        dynamic_axes={
            "input":          {0: "batch", 2: "h", 3: "w"},
            "output":         {0: "batch", 2: "h", 3: "w"},
            "qf_predicted":   {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def validate_onnx(onnx_path: Path) -> None:
    """Run shape inference + check_model. Raise on any issue."""
    model = onnx.load(str(onnx_path))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def quantize_fp16(onnx_path: Path) -> None:
    """Dynamic FP16 weight quantization. Halves file size; native on WebGPU."""
    converted = float16.convert_float_to_float16_model_path(
        str(onnx_path),
        keep_io_types=True,
    )
    onnx.save(converted, str(onnx_path))


def print_ts_stub(variant: Variant, onnx_path: Path) -> None:
    size_bytes = onnx_path.stat().st_size
    public_path = f"/models/{onnx_path.name}"
    parts = variant.id.split("-")
    ts_name = parts[0] + "".join(p.capitalize() for p in parts[1:])
    print()
    print("---- paste into packages/inference-core/src/models/<name>.ts ----")
    print(f"""import type {{ ModelDef }} from './types';

export const {ts_name}: ModelDef = {{
  id: '{variant.id}',
  name: '{variant.name}',
  description: {variant.description!r},
  task: 'jpeg-artifact-removal',
  url: '{public_path}',
  sizeBytes: {size_bytes},
  channels: {variant.in_nc},
  alignment: 8,
  tileSizeDefault: 256,
  inputs: {{
    input:    'image',
    qf_input: {{ param: 'qf' }},
  }},
  outputs: [
    {{ name: 'output' }},
    {{ name: 'qf_predicted' }},
  }},
  params: {{
    qf: {{
      kind: 'range', min: 10, max: 100, step: 1, default: 40,
      label: 'Quality Factor',
      help: 'JPEG quality the model predicts/removes. Lower = stronger artifacts predicted.',
    }},
  }},
}};
""")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert FBCNN port to web-ready ONNX")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS_BY_ID.keys()))
    parser.add_argument("--weights", required=True, type=Path, help="Path to original .pth file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for .onnx")
    parser.add_argument("--no-fp16", action="store_true", help="Skip FP16 quantization")
    args = parser.parse_args()

    variant = VARIANTS_BY_ID[args.variant]
    net = build_and_load(variant, args.weights)

    out_path = args.out / f"{variant.id}.onnx"
    export_onnx(net, variant, out_path)
    print(f"[1/4] exported ONNX: {out_path} ({out_path.stat().st_size:,} bytes)")

    validate_onnx(out_path)
    print(f"[2/4] validated: shape inference + onnx.checker passed")

    if not args.no_fp16:
        quantize_fp16(out_path)
        print(f"[3/4] quantized to FP16: {out_path} ({out_path.stat().st_size:,} bytes)")
    else:
        print(f"[3/4] FP16 skipped (--no-fp16)")

    smoke_test(out_path, net, variant.in_nc)
    print(f"[4/4] smoke test passed: torch vs ORT within tolerance")

    print_ts_stub(variant, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
