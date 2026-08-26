import type { ModelDef } from "./types.js";

// DeJPEGNet v1 (models/dejpeg), c40 checkpoint: trained for JPEG quality 40.
// Exported via scripts/export_onnx.py --dynamic (FP16 weights, FP32 I/O, opset 18).
export const dejpegC40: ModelDef = {
  id: "dejpeg-c40",
  name: "DeJPEGNet v1 · Color (QF 40)",
  description:
    "Compact residual U-Net (2.6M params) trained for JPEG quality 40. Tile-invariant, no quality input needed.",
  task: "jpeg-artifact-removal",
  url: "/models/dejpeg-c40.onnx?v=v1.0.0",
  sizeBytes: 5_372_349,
  channels: 3,
  // Four stride-2 convs -> tile size must be a multiple of 2^4 = 16.
  // Tile origins should additionally land on multiples of 32 for exact LSCA
  // window alignment; the default 512/64 overlap keeps the main grid there.
  alignment: 16,
  tileSizeDefault: 512,
  inputs: {
    input: "image",
  },
  outputs: [{ name: "output" }],
  params: {},
};
