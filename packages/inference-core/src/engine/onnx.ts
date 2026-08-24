// Concrete InferenceEngine backed by onnxruntime-web.
// This is the only file in the package that imports onnxruntime-web.
// It is not exercised by unit tests — they use MockEngine. Plan 3
// exercises OnnxEngine end-to-end via the deployed worker.

import type { Backend, InferenceEngine, Tensor } from "./types.js";
import { BackendUnavailable } from "../errors.js";

// Minimal structural typing over the ORT surface we touch. The real package
// is declared as an optional peer dependency, so it may be absent at
// type-check time; the runtime import is dynamic and gated behind init().
interface ORTModule {
  InferenceSession: {
    create: (
      data: ArrayBuffer | Uint8Array,
      opts: ORTSessionOptions,
    ) => Promise<ORTSession>;
  };
  Tensor: new (
    type: string,
    data: Float32Array | number[],
    dims: number[],
  ) => ORTTensor;
  env: { wasm: { proxy: boolean; numThreads: number }; backends: unknown };
}

interface ORTSessionOptions {
  executionProviders: ReadonlyArray<"webgpu" | "wasm">;
  freeDimensionOverrides?: Record<string, number>;
}

interface ORTTensor {
  data: Float32Array;
  dims: number[];
}

interface ORTSession {
  run: (
    feeds: Record<string, ORTTensor>,
  ) => Promise<Record<string, ORTTensor>>;
  release: () => void;
}

interface NavigatorGPUAware extends Navigator {
  gpu?: unknown;
}

export interface OnnxEngineOptions {
  preferBackend?: ReadonlyArray<Backend> | undefined;
}

export class OnnxEngine implements InferenceEngine {
  private readonly preferBackend: ReadonlyArray<Backend>;
  private _backend: Backend;
  private session: ORTSession | null = null;
  private ort: ORTModule | null = null;

  get backend(): Backend {
    return this._backend;
  }

  constructor(opts: OnnxEngineOptions = {}) {
    this.preferBackend = opts.preferBackend ?? ["webgpu", "wasm"];
    this._backend = this.preferBackend[0] ?? "webgpu";
  }

  async init(opts: { bytes: Uint8Array; signal?: AbortSignal }): Promise<void> {
    if (opts.signal?.aborted) throw new Error("aborted");
    // Dynamic import keeps ORT out of the main bundle and the test bundle.
    // The `as string` cast sidesteps TS module resolution; the package is an
    // optional peer dependency and may not be installed at type-check time.
    this.ort = (await import("onnxruntime-web" as string)) as unknown as ORTModule;

    // Configure WASM threads before creating any session. Without this ORT
    // defaults to single-threaded, which is 4-8x slower on multi-core CPUs.
    if (this.ort.env?.wasm) {
      const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency)
        ? Math.min(navigator.hardwareConcurrency, 8)
        : 4;
      this.ort.env.wasm.numThreads = cores;
      this.ort.env.wasm.proxy = false;
    }

    const buf = opts.bytes;
    let lastErr: unknown = null;
    for (const backend of this.preferBackend) {
      try {
        if (
          backend === "webgpu" &&
          typeof navigator !== "undefined" &&
          (navigator as NavigatorGPUAware).gpu === undefined
        ) {
          throw new Error("navigator.gpu unavailable — browser does not support WebGPU");
        }
        this.session = await this.ort.InferenceSession.create(buf, {
          executionProviders: [backend],
          freeDimensionOverrides: { batch: 1 },
        });
        this._backend = backend;
        if (backend === "wasm") {
          console.warn(
            `[inference-core] Running on WASM backend. ` +
            `WebGPU is recommended for 10-50x faster inference. ` +
            `Ensure navigator.gpu is available (Chrome/Edge 113+, or enable in Firefox/Safari).`,
          );
        }
        return;
      } catch (e) {
        console.warn(`[inference-core] Backend "${backend}" failed:`, e instanceof Error ? e.message : e);
        lastErr = e;
      }
    }
    throw new BackendUnavailable({ backend: this._backend, cause: lastErr });
  }

  async run(
    feeds: Record<string, Tensor>,
    signal?: AbortSignal,
  ): Promise<Record<string, Tensor>> {
    if (!this.ort || !this.session) {
      throw new Error("OnnxEngine: not initialized");
    }
    // ORT-Web ignores AbortSignal on some code paths as of v1.17; callers
    // should also poll signal.aborted between tiles.
    if (signal?.aborted) throw new Error("aborted");

    const ortFeeds: Record<string, ORTTensor> = {};
    for (const [name, t] of Object.entries(feeds)) {
      ortFeeds[name] = new this.ort.Tensor("float32", t.data, [...t.shape]);
    }

    const out = await this.session.run(ortFeeds);

    const result: Record<string, Tensor> = {};
    for (const [name, t] of Object.entries(out)) {
      result[name] = { data: t.data, shape: t.dims };
    }
    return result;
  }

  dispose(): void {
    if (this.session) {
      try {
        this.session.release();
      } catch {
        // Release failures are best-effort.
      }
      this.session = null;
    }
    this.ort = null;
  }
}
