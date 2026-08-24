export type ClientState =
  | "idle"
  | "booting"
  | "model-loading"
  | "ready"
  | "processing"
  | "error";

export interface ClientSnapshot {
  state: ClientState;
  backend: "webgpu" | "wasm" | null;
  modelId: string | null;
  tileSize: number | null;
}

export interface ClientCapabilities {
  nativeDecoders: readonly string[];
  nativeEncoders: readonly string[];
  wasmCodecsAvailable: readonly string[];
}
