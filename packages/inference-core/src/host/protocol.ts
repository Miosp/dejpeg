// Worker ↔ main thread wire protocol. PRIVATE to inference-core.
// Consumers never import this; they use the InferenceClient API.

import type { ModelError } from "../errors.js";
import type { ModelId } from "../models/index.js";
import type { EncodeFormat } from "../codec/types.js";

export type ClientState =
  | "idle"
  | "booting"
  | "model-loading"
  | "ready"
  | "processing"
  | "error";

export type Backend = "webgpu" | "wasm";

export type ProgressEvent =
  | { kind: "boot"; step: "shell" | "wasm" | "ort"; loaded: number; total: number }
  | { kind: "model"; modelId: string; stage: "fetch" | "deserialize"; loaded: number; total: number }
  | {
      kind: "image";
      itemId: string;
      stage: "decode" | "plan" | "tile" | "blend" | "encode" | "finalize";
      done?: number;
      total?: number;
      ms?: number;
    };

export interface SerializedError {
  _tag: string;
  [key: string]: unknown;
}

export type HostOutbound =
  | {
      kind: "state";
      state: ClientState;
      backend?: Backend;
      modelId?: string;
      tileSize?: number;
      capabilities?: ClientCapabilities;
    }
  | { kind: "progress-batch"; events: ProgressEvent[] }
  | {
      kind: "result";
      itemId: string;
      bitmap: ImageBitmap;
      blob: Blob;
      width: number;
      height: number;
      tileSizeUsed: number;
      elapsedMs: number;
      qfPredicted?: number;
    }
  | { kind: "encode-result"; itemId: string; blob: Blob }
  | { kind: "error"; itemId: string | null; error: SerializedError };

export interface ClientCapabilities {
  nativeDecoders: readonly string[];
  nativeEncoders: readonly string[];
  wasmCodecsAvailable: readonly string[];
}

export type ClientInbound =
  | { kind: "load-model"; modelId: ModelId }
  | {
      kind: "process";
      itemId: string;
      file: File;
      params?: Record<string, unknown>;
      tileSizeOverride?: number;
      tileBatch?: number;
    }
  | { kind: "encode"; itemId: string; bitmap: ImageBitmap; format: EncodeFormat; quality?: number }
  | { kind: "cancel"; itemId: string }
  | { kind: "dispose" };

export function serializeError(e: ModelError | Error): SerializedError {
  const tag = (e as { _tag?: string })._tag ?? "Unknown";
  const out: SerializedError = { _tag: tag };
  for (const [k, v] of Object.entries(e)) {
    if (k === "_tag" || k === "message" || k === "stack") continue;
    if (typeof v === "function" || typeof v === "symbol") continue;
    try {
      if (JSON.stringify(v) === undefined) continue;
      out[k] = v;
    } catch {
      // skip non-serializable
    }
  }
  if (e.message) out.message = e.message;
  return out;
}
