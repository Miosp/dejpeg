import type { ModelDef } from "./types.js";

// STUB: sizeBytes and url updated by Plan 1 conversion output.
// Replace this file with the printed stub from
//   uv run python tools/convert-fbcnn/convert.py --variant fbcnn-color-real ...
export const fbcnnColorReal: ModelDef = {
  id: "fbcnn-color-real",
  name: "FBCNN · Color (Real-World)",
  description:
    "Removes JPEG artifacts from real photographs with unknown compression history.",
  task: "jpeg-artifact-removal",
  // TODO: replace PLACEHOLDER with actual R2 public bucket ID
  url: "/models/fbcnn-color-real.onnx?v=fp16",
  sizeBytes: 8_000_000,
  channels: 3,
  // FBCNN has 3 downsample layers -> tile size must be a multiple of 2^3 = 8.
  alignment: 8,
  tileSizeDefault: 512,
  inputs: {
    input: "image",
    qf_input: { param: "qf" },
  },
  outputs: [{ name: "output" }, { name: "qf_predicted" }],
  params: {
    qf: {
      kind: "range",
      min: 10,
      max: 100,
      step: 1,
      default: 40,
      label: "Quality Factor",
      help: "JPEG quality the model predicts/removes. Lower = stronger artifacts predicted.",
    },
  },
};
