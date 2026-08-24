import type { Backend, InferenceEngine, Tensor } from "./types.js";

export interface MockEngineConfig {
  backend?: Backend;
  /** If set, every call to `run` rejects with this error instead of returning. */
  failWith?: (feeds: Record<string, Tensor>) => Error | undefined;
  /**
   * If set, `run` rejects with the configured error when the input tile size
   * (deduced from feeds.input.shape) matches one of these sizes. Used by the
   * adaptive sizing test to simulate OOM at large sizes.
   */
  failAtTileSizes?: number[];
  /**
   * Output producer. Defaults to passing the input through unchanged
   * (useful for blending tests where we want predictable outputs).
   */
  produce?: (feeds: Record<string, Tensor>) => Record<string, Tensor>;
}

export class MockEngine implements InferenceEngine {
  readonly backend: Backend;
  readonly calls: Array<{ feeds: Record<string, Tensor> }> = [];
  private readonly cfg: MockEngineConfig;
  private disposed = false;

  constructor(cfg: MockEngineConfig = {}) {
    this.backend = cfg.backend ?? "webgpu";
    this.cfg = cfg;
  }

  async init(_opts: { bytes: Uint8Array; signal?: AbortSignal }): Promise<void> {
    // No-op for the mock.
  }

  async run(feeds: Record<string, Tensor>, _signal?: AbortSignal): Promise<Record<string, Tensor>> {
    if (this.disposed) throw new Error("MockEngine: disposed");
    this.calls.push({ feeds });

    // 1. Explicit failure producer
    if (this.cfg.failWith) {
      const err = this.cfg.failWith(feeds);
      if (err) throw err;
    }

    // 2. Allocation failure at specific tile sizes
    if (this.cfg.failAtTileSizes) {
      const input = feeds.input;
      if (input && input.shape.length === 4) {
        const h = input.shape[2]!;
        if (this.cfg.failAtTileSizes.includes(h)) {
          throw new (class extends Error {
            _tag = "TileAllocationFailure";
          })(`simulated OOM at tile ${h}`);
        }
      }
    }

    // 3. Configured output or identity passthrough
    if (this.cfg.produce) return this.cfg.produce(feeds);
    const input = feeds.input;
    if (!input) throw new Error("MockEngine: missing 'input' feed");
    return { output: input };
  }

  dispose(): void {
    this.disposed = true;
  }
}
