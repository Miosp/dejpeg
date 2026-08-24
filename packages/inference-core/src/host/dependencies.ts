import { ModelCache } from "./ModelCache.js";
import { OffscreenCanvasPool } from "./OffscreenCanvasPool.js";
import { OnnxEngine } from "../engine/onnx.js";
import { MODELS } from "../models/index.js";
import type { InferenceEngine } from "../engine/types.js";
import type { ModelDef } from "../models/types.js";
import type { HostOutbound } from "./protocol.js";

/**
 * Subset of ModelCache the Host consumes. Declared as an interface so tests
 * can substitute a byte source without touching the network.
 */
export interface ModelSource {
  fetch(
    url: string,
    onProgress?: (loaded: number, total: number) => void,
  ): Promise<Uint8Array>;
}

export interface HostDependencies {
  engine: InferenceEngine;
  modelCache: ModelSource;
  canvasPool: OffscreenCanvasPool;
  postMessage: (msg: HostOutbound, transfer?: Transferable[]) => void;
  signal?: AbortSignal;
}

export interface HostOpts {
  deps: HostDependencies;
  availableModels: readonly ModelDef[];
}

export async function makeDefaultDependencies(): Promise<HostDependencies> {
  const cache = await caches.open("inference-models");
  return {
    engine: new OnnxEngine(),
    modelCache: new ModelCache(cache),
    canvasPool: new OffscreenCanvasPool(),
    postMessage: (msg, transfer) =>
      (postMessage as (m: HostOutbound, t?: Transferable[]) => void)(msg, transfer),
  };
}

export { MODELS };
