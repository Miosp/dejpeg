// Engine abstraction. v1 ships OnnxEngine; future LitertEngine implements the same surface.

export type Backend = "webgpu" | "wasm";

export interface Tensor {
  /** NCHW or scalar flat data. */
  data: Float32Array;
  /** Shape of the data; e.g. [1, 3, 256, 256] for an image tile. */
  shape: readonly number[];
  dtype?: "float32";
}

export type ModelParams = Record<string, number | string | boolean>;

/**
 * Abstract inference engine. Owns the model session and runs forward passes.
 * Implementations must be disposable; sessions are not cheap.
 */
export interface InferenceEngine {
  /** Backend in use after init; surfaces to UI via progress events. */
  readonly backend: Backend;

  /**
   * Load the model from already-fetched bytes and create the inference
   * session. The Host owns fetching (via ModelCache); engines must not
   * fetch their own URL. Throws BackendUnavailable if neither backend
   * can initialize.
   */
  init(opts: { bytes: Uint8Array; signal?: AbortSignal }): Promise<void>;

  /**
   * Run a forward pass. `feeds` maps the model's input names to Tensors.
   * Implementation handles parameter tensor construction (see ModelDef.inputs).
   */
  run(
    feeds: Record<string, Tensor>,
    signal?: AbortSignal,
  ): Promise<Record<string, Tensor>>;

  /** Release all backend resources. Safe to call multiple times. */
  dispose(): void;
}
