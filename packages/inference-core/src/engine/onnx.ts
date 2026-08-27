// Concrete InferenceEngine backed by onnxruntime-web.
// This is the only file in the package that imports onnxruntime-web.
// It is not exercised by unit tests — they use MockEngine. Plan 3
// exercises OnnxEngine end-to-end via the deployed worker.

import type { Backend, InferenceEngine, Tensor } from "./types.js";
import { BackendUnavailable } from "../errors.js";

const ORT_WASM_CDN =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

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

const DEBUG_LOG =
  typeof location !== "undefined" && new URLSearchParams(location.search).has("debug");

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

    // "error" hides the noisy per-session EP-assignment warnings; they are
    // expected (shape ops are deliberately CPU-assigned) and carry no signal.
    if (this.ort.env?.logLevel) {
      (this.ort.env as { logLevel: string }).logLevel = "error";
    }

    // Configure WASM threads before creating any session. Without this ORT
    // defaults to single-threaded, which is 4-8x slower on multi-core CPUs.
    if (this.ort.env?.wasm) {
      const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency)
        ? Math.min(navigator.hardwareConcurrency, 8)
        : 4;
      this.ort.env.wasm.numThreads = cores;
      this.ort.env.wasm.proxy = false;
      // The ORT WASM binary (~26 MiB) exceeds the 25 MiB per-file asset limit
      // on Cloudflare Workers, so it is dropped from the bundle and fetched
      // from the CDN instead. jsdelivr serves CORS headers, which satisfies
      // the page's COEP require-corp policy. Keep in sync with the version
      // resolved in apps/web's lockfile.
      this.ort.env.wasm.wasmPaths = ORT_WASM_CDN;
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
        const t0 = performance.now();
        this.session = await this.ort.InferenceSession.create(buf, {
          executionProviders: [backend],
        });
        if (DEBUG_LOG) {
          console.info(
            `[engine] session created on "${backend}" in ${(performance.now() - t0).toFixed(0)}ms`,
          );
        }
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

    const t0 = performance.now();
    const out = await this.session.run(ortFeeds);
    if (DEBUG_LOG) {
      console.info(
        `[engine] run ${this._backend} [${Object.values(ortFeeds).map((t) => t.dims.join("x")).join(",")}] ${(performance.now() - t0).toFixed(1)}ms`,
      );
    }

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
