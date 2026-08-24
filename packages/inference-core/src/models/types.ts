// Model definition format — one TS file per model. See spec 05-models.md.

export type ModelTask = "jpeg-artifact-removal";

export type ModelParam =
  | {
      kind: "range";
      min: number;
      max: number;
      step?: number;
      default: number;
      label: string;
      help: string;
    }
  | {
      kind: "select";
      options: readonly string[];
      default: string;
      label: string;
      help: string;
    }
  | {
      kind: "toggle";
      default: boolean;
      label: string;
      help: string;
    };

/**
 * Binding for a model input. Either it's the image tensor ("image") or it
 * pulls a value from the user params by name.
 */
export type InputBinding = "image" | { param: string };

export interface ModelDef {
  id: string;
  name: string;
  description: string;
  task: ModelTask;
  /** Path or external URL to the ONNX file. */
  url: string;
  /** File size in bytes, for byte-progress UI. */
  sizeBytes: number;
  /** 1 for grayscale, 3 for color. */
  channels: 1 | 3;
  /** Tile size must be a multiple of this. Default 1. */
  alignment: number;
  /** Starting tile size for adaptive settle. Default 256. */
  tileSizeDefault?: number;
  /** Map of input name -> binding. */
  inputs: Record<string, InputBinding>;
  /** Output names (informational). */
  outputs: readonly { name: string }[];
  /** User-tunable params. Rendered by ParamPanel in the UI. */
  params: Record<string, ModelParam>;
}
