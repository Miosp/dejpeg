import type { ModelDef } from "./types.js";

// DeJPEGNet v2 (HybNet, 7.47M params). Weights are published in the v2.0.0
// GitHub release. FP16 weights, FP32 I/O, opset 17, dynamic N/H/W.
const base = {
  task: "jpeg-artifact-removal",
  channels: 3,
  // Four downsamples -> tile size must be a multiple of 2^4 = 16.
  alignment: 16,
  tileSizeDefault: 512,
  outputs: [{ name: "output" }],
} as const;

export const dejpegV2: ModelDef = {
  ...base,
  id: "dejpeg-v2",
  name: "DeJPEGNet v2",
  description:
    "Removes JPEG artifacts at any quality with no quality input. Stays close to the original.",
  url: "/models/dejpeg-v2.onnx?v=H1",
  sizeBytes: 15_376_612,
  inputs: {
    input: "image",
  },
  params: {},
};

export const dejpegV2Texture: ModelDef = {
  ...base,
  id: "dejpeg-v2-texture",
  name: "DeJPEGNet v2 + Texture",
  description:
    "DeJPEGNet v2 with a texture control. Adds natural-looking grain, which is synthetic and not recovered detail.",
  url: "/models/dejpeg-v2-texture.onnx?v=H1_rhF3",
  sizeBytes: 15_479_460,
  inputs: {
    input: "image",
    // The graph takes one gain of shape [1, 1], whatever the batch size.
    g: { param: "texture", shared: true },
  },
  params: {
    texture: {
      kind: "range",
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.5,
      label: "Texture",
      help: "Strength of the added grain. 0 is the plain restoration; higher looks crisper but measures slightly further from the original.",
    },
  },
};
