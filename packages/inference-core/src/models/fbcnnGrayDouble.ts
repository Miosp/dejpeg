import type { ModelDef } from "./types.js";

// STUB: sizeBytes and url updated by Plan 1 conversion output.
export const fbcnnGrayDouble: ModelDef = {
  id: "fbcnn-gray-double",
  name: "FBCNN · Grayscale (Double JPEG)",
  description:
    "Removes artifacts from grayscale images that were JPEG-compressed twice (non-aligned).",
  task: "jpeg-artifact-removal",
  // TODO: replace PLACEHOLDER with actual R2 public bucket ID
  url: "/models/fbcnn-gray-double.onnx",
  sizeBytes: 8_000_000,
  channels: 1,
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
      help: "JPEG quality the model predicts/removes.",
    },
  },
};
