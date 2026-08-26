import {
  NoModelLoaded,
  ClientBusy,
  WorkerCrash,
  Cancelled,
  type ModelError,
} from "../errors.js";
import type { ModelDef } from "../models/types.js";
import type { ModelParams } from "../engine/types.js";
import type { EncodeFormat } from "../codec/types.js";
import type {
  ClientInbound,
  HostOutbound,
  Backend,
  ProgressEvent,
} from "../host/protocol.js";
import type { HostTransport, } from "./transport.js";
import { WorkerTransport } from "./transport.js";
import { Subscribable } from "./subscribe.js";
import { rehydrateError } from "./rehydrateError.js";
import type { ClientState, ClientSnapshot, ClientCapabilities } from "./ClientState.js";

export interface ProcessOpts {
  file: File;
  params?: ModelParams;
  tileSizeOverride?: number;
  tileBatch?: number;
  signal?: AbortSignal;
  onProgress?: (event: ProgressEvent) => void;
  itemId?: string;
}

export interface ProcessResult {
  bitmap: ImageBitmap;
  itemId: string;
  blob: Blob;
  width: number;
  height: number;
  tileSizeUsed: number;
  elapsedMs: number;
  qfPredicted?: number;
}

export interface EncodeOpts {
  format: EncodeFormat;
  quality?: number;
  itemId?: string;
}

export interface InferenceClient {
  readonly state: ClientState;
  readonly backend: Backend | null;
  readonly modelId: string | null;
  readonly tileSize: number | null;
  readonly availableModels: readonly ModelDef[];
  readonly capabilities: ClientCapabilities | null;
  subscribe(listener: (snap: ClientSnapshot) => void): () => void;
  loadModel(modelId: string, signal?: AbortSignal): Promise<void>;
  process(opts: ProcessOpts): Promise<ProcessResult>;
  encode(bitmap: ImageBitmap, opts: EncodeOpts): Promise<Blob>;
  dispose(): void;
}

export interface CreateClientOpts {
  transport?: HostTransport;
  workerURL?: URL;
  availableModels?: readonly ModelDef[];
}

type OpKind = "process" | "encode" | "load-model";

interface PendingProcess {
  resolve: (r: ProcessResult) => void;
  reject: (e: ModelError) => void;
  itemId: string;
  onProgress?: (event: ProgressEvent) => void;
}
interface PendingEncode {
  resolve: (b: Blob) => void;
  reject: (e: ModelError) => void;
  itemId: string;
}
interface PendingLoadModel {
  resolve: () => void;
  reject: (e: ModelError) => void;
}

class InferenceClientImpl extends Subscribable<ClientSnapshot> implements InferenceClient {
  private readonly transport: HostTransport;
  private _state: ClientState = "idle";
  private _backend: Backend | null = null;
  private _modelId: string | null = null;
  private _tileSize: number | null = null;
  private _capabilities: ClientCapabilities | null = null;
  private readonly _availableModels: readonly ModelDef[];
  private activeOp: OpKind | null = null;
  private pendingProcess: PendingProcess | null = null;
  private pendingEncode: PendingEncode | null = null;
  private pendingLoadModel: PendingLoadModel | null = null;

  constructor(opts: CreateClientOpts) {
    super();
    if (opts.transport) {
      this.transport = opts.transport;
    } else if (opts.workerURL) {
      this.transport = new WorkerTransport(
        new Worker(opts.workerURL, { type: "module" }),
      );
    } else {
      throw new Error("createInferenceClient requires either transport or workerURL");
    }
    this._availableModels = opts.availableModels ?? [];
    this.transport.onmessage = (e) => this.handleOutbound(e.data);
    this.transport.onerror = () => {
      this.failAllPending(new WorkerCrash({ workerId: "main" }));
      this.setState("error");
    };
  }

  get state(): ClientState {
    return this._state;
  }
  get backend(): Backend | null {
    return this._backend;
  }
  get modelId(): string | null {
    return this._modelId;
  }
  get tileSize(): number | null {
    return this._tileSize;
  }
  get availableModels(): readonly ModelDef[] {
    return this._availableModels;
  }
  get capabilities(): ClientCapabilities | null {
    return this._capabilities;
  }

  loadModel(modelId: string, signal?: AbortSignal): Promise<void> {
    this.requireIdle("load-model");
    this.activeOp = "load-model";
    this.send({ kind: "load-model", modelId });
    return new Promise<void>((resolve, reject) => {
      this.pendingLoadModel = { resolve, reject };
      if (signal) {
        signal.addEventListener(
          "abort",
          () => {
            this.send({ kind: "cancel", itemId: modelId });
            reject(new Cancelled({ id: modelId }));
            this.pendingLoadModel = null;
            this.activeOp = null;
          },
          { once: true },
        );
      }
    });
  }

  process(opts: ProcessOpts): Promise<ProcessResult> {
    if (this._state !== "ready" && this._state !== "processing") {
      throw new NoModelLoaded({ operation: "process" });
    }
    this.requireIdle("process");
    this.activeOp = "process";
    const itemId = opts.itemId ?? generateItemId();
    const msg: ClientInbound = {
      kind: "process",
      itemId,
      file: opts.file,
    };
    if (opts.params) {
      // Deep-clone to strip Svelte 5 $state reactivity Proxies, which cannot
      // survive structured clone across postMessage.
      msg.params = JSON.parse(JSON.stringify(opts.params));
    }
    if (opts.tileSizeOverride !== undefined) msg.tileSizeOverride = opts.tileSizeOverride;
    if (opts.tileBatch !== undefined) msg.tileBatch = opts.tileBatch;
    this.send(msg);
    return new Promise<ProcessResult>((resolve, reject) => {
      this.pendingProcess = {
        resolve,
        reject,
        itemId,
        ...(opts.onProgress ? { onProgress: opts.onProgress } : {}),
      };
      if (opts.signal) {
        opts.signal.addEventListener(
          "abort",
          () => {
            this.send({ kind: "cancel", itemId });
            reject(new Cancelled({ id: itemId }));
            this.pendingProcess = null;
            this.activeOp = null;
          },
          { once: true },
        );
      }
    });
  }

  encode(bitmap: ImageBitmap, opts: EncodeOpts): Promise<Blob> {
    this.requireIdle("encode");
    this.activeOp = "encode";
    const itemId = opts.itemId ?? generateItemId();
    const msg: ClientInbound = {
      kind: "encode",
      itemId,
      bitmap,
      format: opts.format,
    };
    if (opts.quality !== undefined) msg.quality = opts.quality;
    this.send(msg);
    return new Promise<Blob>((resolve, reject) => {
      this.pendingEncode = { resolve, reject, itemId };
    });
  }

  dispose(): void {
    this.send({ kind: "dispose" });
    this.transport.terminate();
  }

  private handleOutbound(msg: HostOutbound): void {
    switch (msg.kind) {
      case "state":
        this.applyState(msg);
        break;
      case "progress-batch":
        this.dispatchProgress(msg.events);
        break;
      case "result":
        if (this.pendingProcess?.itemId === msg.itemId) {
          const p = this.pendingProcess;
          this.pendingProcess = null;
          this.activeOp = null;
          p.resolve({
            bitmap: msg.bitmap,
            itemId: msg.itemId,
            blob: msg.blob,
            width: msg.width,
            height: msg.height,
            tileSizeUsed: msg.tileSizeUsed,
            elapsedMs: msg.elapsedMs,
            qfPredicted: msg.qfPredicted,
          });
        }
        break;
      case "encode-result":
        if (this.pendingEncode?.itemId === msg.itemId) {
          const p = this.pendingEncode;
          this.pendingEncode = null;
          this.activeOp = null;
          p.resolve(msg.blob);
        }
        break;
      case "error": {
        const err = rehydrateError(msg.error);
        this.failPendingForItemId(msg.itemId, err);
        break;
      }
    }
  }

  private applyState(msg: Extract<HostOutbound, { kind: "state" }>): void {
    if (msg.backend) this._backend = msg.backend;
    if (msg.modelId !== undefined) this._modelId = msg.modelId;
    if (msg.tileSize !== undefined) this._tileSize = msg.tileSize;
    if (msg.capabilities) this._capabilities = msg.capabilities;
    this.setState(msg.state);
    if (msg.state === "ready" && this.pendingLoadModel) {
      const p = this.pendingLoadModel;
      this.pendingLoadModel = null;
      this.activeOp = null;
      p.resolve();
    }
  }

  private dispatchProgress(events: readonly ProgressEvent[]): void {
    if (!this.pendingProcess?.onProgress) return;
    for (const e of events) {
      if (e.kind === "image" && e.itemId === this.pendingProcess.itemId) {
        this.pendingProcess.onProgress(e);
      }
    }
  }

  private setState(state: ClientState): void {
    if (this._state === state) return;
    this._state = state;
    this.emit(this.snapshot());
  }

  private snapshot(): ClientSnapshot {
    return {
      state: this._state,
      backend: this._backend,
      modelId: this._modelId,
      tileSize: this._tileSize,
    };
  }

  private requireIdle(op: OpKind): void {
    if (this.activeOp !== null) throw new ClientBusy({ operation: op });
  }

  private failPendingForItemId(itemId: string | null, err: ModelError): void {
    if (itemId && this.pendingProcess?.itemId === itemId) {
      const p = this.pendingProcess;
      this.pendingProcess = null;
      this.activeOp = null;
      p.reject(err);
    } else if (itemId && this.pendingEncode?.itemId === itemId) {
      const p = this.pendingEncode;
      this.pendingEncode = null;
      this.activeOp = null;
      p.reject(err);
    } else if (this.pendingLoadModel && !itemId) {
      const p = this.pendingLoadModel;
      this.pendingLoadModel = null;
      this.activeOp = null;
      p.reject(err);
    }
    this.setState("error");
  }

  private failAllPending(err: ModelError): void {
    const process = this.pendingProcess;
    const encode = this.pendingEncode;
    const load = this.pendingLoadModel;
    this.pendingProcess = null;
    this.pendingEncode = null;
    this.pendingLoadModel = null;
    this.activeOp = null;
    process?.reject(err);
    encode?.reject(err);
    load?.reject(err);
  }

  private send(msg: ClientInbound): void {
    this.transport.postMessage(msg);
  }
}

function generateItemId(): string {
  return `item-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createInferenceClient(opts: CreateClientOpts): InferenceClient {
  return new InferenceClientImpl(opts);
}
