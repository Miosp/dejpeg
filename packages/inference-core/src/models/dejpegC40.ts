import type { ModelDef } from "./types.js";

// DeJPEGNet v1 (models/dejpeg). "c40" = c0=40 base channel width, not a
// quality factor -- training samples QF 1-100 (see degrade.py sample_qf).
// Exported via scripts/export_onnx.py --dynamic (FP16 weights, FP32 I/O, opset 18).
export const dejpegC40: ModelDef = {
  id: "dejpeg-c40",
  name: "DeJPEGNet v1",
  description:
    "Compact residual U-Net (2.6M params) trained across JPEG qualities 1-100 with realistic degradation (chroma subsampling, multi-pass, resizes). Tile-invariant, no quality input needed.",
  task: "jpeg-artifact-removal",
  url: "/models/dejpeg-c40.onnx?v=v1.0.1",
  sizeBytes: 5_383_745,
  channels: 3,
  // Four stride-2 convs -> tile size must be a multiple of 2^4 = 16. Any
  // multiple works: the LSCA attention pools export as floor-division with
  // static pads (ceil_mode would put a ceil() into the ONNX shape math,
  // which onnxruntime-web cannot evaluate), so partial edge windows at any
  // grid depth are handled exactly.
  alignment: 16,
  tileSizeDefault: 512,
  inputs: {
    input: "image",
  },
  outputs: [{ name: "output" }],
  params: {},
};
