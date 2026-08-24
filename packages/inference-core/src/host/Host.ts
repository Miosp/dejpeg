import { Effect } from "effect";
import { decode } from "../codec/decode.js";
import { encode } from "../codec/encode.js";
import { processImage, type ProcessImageOpts } from "../pipeline/pipeline.js";
import { EngineEnv } from "../pipeline/runtime.js";
import { settleTileSize } from "../pipeline/sizing.js";
import { UnknownError, type ModelError } from "../errors.js";
import type { ModelDef } from "../models/types.js";
import {
  type ClientInbound,
  type HostOutbound,
  type ProgressEvent,
  type Backend,
  type ClientState,
  type ClientCapabilities,
  serializeError,
} from "./protocol.js";
import { ProgressBatcher } from "./ProgressBatcher.js";
import type { HostDependencies, HostOpts } from "./dependencies.js";

interface TileSizeCache {
  get(key: string): number | undefined;
  set(key: string, size: number): void;
}

function getTileSizeCache(): TileSizeCache {
  const g = globalThis as unknown as {
    __inferenceTileSizeCache?: Map<string, number>;
  };
  if (!g.__inferenceTileSizeCache) {
    g.__inferenceTileSizeCache = new Map();
  }
  const map = g.__inferenceTileSizeCache;
  return {
    get: (k) => map.get(k),
    set: (k, v) => {
      map.set(k, v);
    },
  };
}

export class Host {
  private readonly deps: HostDependencies;
  private readonly availableModels: readonly ModelDef[];
  private activeModel: ModelDef | null = null;
  private readonly batcher: ProgressBatcher;
  private activeItem: AbortController | null = null;
  private capabilities: ClientCapabilities | null = null;

  constructor(opts: HostOpts) {
    this.deps = opts.deps;
    this.availableModels = opts.availableModels;
    this.batcher = new ProgressBatcher({
      flush: (events) => this.post({ kind: "progress-batch", events }),
    });
  }

  async handle(msg: ClientInbound): Promise<void> {
    try {
      switch (msg.kind) {
        case "load-model":
          await this.loadModel(msg.modelId);
          break;
        case "process":
          await this.process(msg);
          break;
        case "encode":
          await this.encode(msg);
          break;
        case "cancel":
          this.activeItem?.abort();
          break;
        case "dispose":
          this.deps.engine.dispose();
          break;
      }
    } catch (e) {
      const error =
        e instanceof Error ? e : new UnknownError({ message: String(e) });
      const itemId = "itemId" in msg ? msg.itemId : null;
      this.send({
        kind: "error",
        itemId: itemId ?? null,
        error: serializeError(error as ModelError | Error),
      });
      this.setState("error");
    }
  }

  private async loadModel(modelId: string): Promise<void> {
    const def = this.availableModels.find((m) => m.id === modelId);
    if (!def) {
      throw new UnknownError({ message: `Unknown model id: ${modelId}` });
    }

    this.setState("model-loading");

    const bytes = await this.deps.modelCache.fetch(def.url, (loaded, total) => {
      this.emit({ kind: "model", modelId, stage: "fetch", loaded, total });
    });

    console.log(`[host] Model loaded: ${def.url} → ${(bytes.byteLength / 1e6).toFixed(1)} MB (${bytes.byteLength > 200_000_000 ? "FP32" : "FP16"})`);

    this.emit({
      kind: "model",
      modelId,
      stage: "deserialize",
      loaded: 0,
      total: bytes.byteLength,
    });

    await this.deps.engine.init({ bytes });

    this.activeModel = def;
    if (!this.capabilities) this.capabilities = await this.probeCapabilities();

    const tileSize = await this.cachedTileSize(def, this.deps.engine.backend);
    this.setState("ready", {
      backend: this.deps.engine.backend,
      modelId,
      tileSize,
      capabilities: this.capabilities,
    });
  }

  private async process(
    msg: Extract<ClientInbound, { kind: "process" }>,
  ): Promise<void> {
    if (!this.activeModel) {
      throw new UnknownError({
        message: "process called before load-model completed",
      });
    }
    const def = this.activeModel;
    this.setState("processing");
    this.activeItem = new AbortController();

    const t0 = performance.now();
    this.emit({ kind: "image", itemId: msg.itemId, stage: "decode" });
    const decoded = await decode(msg.file, def.channels);

    const pipelineOpts: ProcessImageOpts = {};
    if (msg.tileSizeOverride !== undefined) {
      pipelineOpts.tileSizeOverride = msg.tileSizeOverride;
    }
    const params = (msg.params ?? {}) as Record<string, number | string | boolean>;

    const program = processImage(
      {
        itemId: msg.itemId,
        image: {
          data: decoded.rgb,
          channels: def.channels,
          width: decoded.width,
          height: decoded.height,
        },
      },
      def,
      params,
      (ev) => this.emit(ev),
      pipelineOpts,
    );

    const result = await Effect.runPromise(
      Effect.provideService(program, EngineEnv, { engine: this.deps.engine }),
    );

    this.emit({ kind: "image", itemId: msg.itemId, stage: "encode" });
    const encoded = await encode(
      {
        rgb: result.data,
        alpha: decoded.alpha,
        width: result.width,
        height: result.height,
      },
      { format: "png" },
    );

    const elapsedMs = performance.now() - t0;
    this.emit({ kind: "image", itemId: msg.itemId, stage: "finalize" });

    this.send(
      {
        kind: "result",
        itemId: msg.itemId,
        bitmap: encoded.bitmap,
        blob: encoded.blob,
        width: result.width,
        height: result.height,
        tileSizeUsed: 0,
        elapsedMs,
        qfPredicted: result.qfPredicted,
      },
      [encoded.bitmap],
    );

    this.setState("ready");
  }

  private async encode(
    msg: Extract<ClientInbound, { kind: "encode" }>,
  ): Promise<void> {
    const canvas = this.deps.canvasPool.acquire(
      msg.bitmap.width,
      msg.bitmap.height,
    );
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(msg.bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, msg.bitmap.width, msg.bitmap.height);
    this.deps.canvasPool.release(canvas);

    const { data, width, height } = imageData;
    const pixelCount = width * height;
    const rgb = new Float32Array(3 * pixelCount);
    const alpha = data.some((v, i) => i % 4 === 3 && v < 255)
      ? new Uint8ClampedArray(pixelCount)
      : null;
    for (let i = 0; i < pixelCount; i++) {
      rgb[i] = data[i * 4]! / 255;
      rgb[pixelCount + i] = data[i * 4 + 1]! / 255;
      rgb[2 * pixelCount + i] = data[i * 4 + 2]! / 255;
      if (alpha && data[i * 4 + 3]! < 255) alpha[i] = data[i * 4 + 3]!;
    }

    const encodeOpts: { format: typeof msg.format; quality?: number } = {
      format: msg.format,
    };
    if (msg.quality !== undefined) encodeOpts.quality = msg.quality;

    const { blob } = await encode(
      { rgb, alpha, width, height },
      encodeOpts,
    );

    this.send({ kind: "encode-result", itemId: msg.itemId, blob });
  }

  private async cachedTileSize(
    def: ModelDef,
    backend: Backend,
  ): Promise<number> {
    const cacheKey = `tileSize:${def.id}:${backend}`;
    const cache = getTileSizeCache();
    const cached = cache.get(cacheKey);
    if (cached !== undefined) return cached;

    const size = await Effect.runPromise(settleTileSize(this.deps.engine, def));
    cache.set(cacheKey, size);
    return size;
  }

  private async probeCapabilities(): Promise<ClientCapabilities> {
    return {
      nativeDecoders: ["png", "jpeg", "webp", "gif", "bmp"],
      nativeEncoders: ["png", "jpeg", "webp"],
      wasmCodecsAvailable: ["heic", "tiff", "avif"],
    };
  }

  private setState(
    state: ClientState,
    extra?: Partial<Extract<HostOutbound, { kind: "state" }>>,
  ): void {
    const msg: Extract<HostOutbound, { kind: "state" }> = {
      kind: "state",
      state,
      ...extra,
    };
    this.send(msg);
  }

  private emit(ev: ProgressEvent): void {
    this.batcher.emit(ev);
  }

  private post(msg: HostOutbound, transfer?: Transferable[]): void {
    this.deps.postMessage(msg, transfer ?? []);
  }

  private send(msg: HostOutbound, transfer?: Transferable[]): void {
    this.batcher.flushNow();
    this.deps.postMessage(msg, transfer ?? []);
  }
}
