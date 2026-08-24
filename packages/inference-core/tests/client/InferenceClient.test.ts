import { createInferenceClient } from "../../src/client/InferenceClient.js";
import type { HostTransport } from "../../src/client/transport.js";
import type { HostOutbound, ClientInbound } from "../../src/host/protocol.js";
import { NoModelLoaded, ClientBusy, ImageDecodeError } from "../../src/errors.js";
import { describe, it, expect } from "bun:test";

function makeMockTransport(): HostTransport & {
  inbound: ClientInbound[];
  emit: (m: HostOutbound) => void;
  emitError: (e: ErrorEvent) => void;
} {
  const inbound: ClientInbound[] = [];
  let onmessage: ((e: MessageEvent<HostOutbound>) => void) | null = null;
  let onerror: ((e: ErrorEvent) => void) | null = null;
  return {
    inbound,
    postMessage: (msg) => inbound.push(msg),
    get onmessage() {
      return onmessage;
    },
    set onmessage(v) {
      onmessage = v;
    },
    get onerror() {
      return onerror;
    },
    set onerror(v) {
      onerror = v;
    },
    terminate: () => {},
    emit: (m) => onmessage?.({ data: m } as MessageEvent<HostOutbound>),
    emitError: (e) => onerror?.(e),
  };
}

describe("InferenceClient", () => {
  it("starts in idle state with no backend", () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    expect(client.state).toBe("idle");
    expect(client.backend).toBeNull();
    expect(client.modelId).toBeNull();
    expect(client.tileSize).toBeNull();
    expect(client.capabilities).toBeNull();
  });

  it("requires transport or workerURL", () => {
    expect(() => createInferenceClient({})).toThrow();
  });

  it("loadModel sends a load-model message and resolves on ready", async () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    const promise = client.loadModel("fbcnn-color-real");
    expect(transport.inbound).toEqual([{ kind: "load-model", modelId: "fbcnn-color-real" }]);
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "fbcnn-color-real",
      tileSize: 256,
    });
    await promise;
    expect(client.state).toBe("ready");
    expect(client.backend).toBe("wasm");
    expect(client.modelId).toBe("fbcnn-color-real");
    expect(client.tileSize).toBe(256);
  });

  it("process throws NoModelLoaded synchronously when not ready", () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    const file = new File([new Uint8Array([0])], "x.png", { type: "image/png" });
    expect(() => client.process({ file })).toThrow(NoModelLoaded);
  });

  it("process throws ClientBusy synchronously when called concurrently", async () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    const first = client.process({
      file: new File([], "a.png"),
      itemId: "a",
    });
    expect(() =>
      client.process({ file: new File([], "b.png"), itemId: "b" }),
    ).toThrow(ClientBusy);
    transport.emit({
      kind: "error",
      itemId: "a",
      error: { _tag: "Cancelled", id: "a" },
    });
    await first.catch(() => {});
  });

  it("subscribe fires on state changes", () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    const snapshots: string[] = [];
    client.subscribe((s) => snapshots.push(s.state));
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    expect(snapshots).toEqual(["ready"]);
  });

  it("subscribe does not fire when state is unchanged", () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    const snapshots: string[] = [];
    client.subscribe((s) => snapshots.push(s.state));
    transport.emit({
      kind: "state",
      state: "idle",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    expect(snapshots).toEqual([]);
  });

  it("rehydrates errors from the wire", async () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    const file = new File([new Uint8Array([0])], "bad.png", { type: "image/png" });
    const promise = client.process({ file, itemId: "bad" });
    transport.emit({
      kind: "error",
      itemId: "bad",
      error: { _tag: "ImageDecodeError", filename: "bad.png" },
    });
    await expect(promise).rejects.toBeInstanceOf(ImageDecodeError);
    expect(client.state).toBe("error");
  });

  it("resolves process on result message", async () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    const file = new File([new Uint8Array([0])], "in.png", { type: "image/png" });
    const promise = client.process({ file, itemId: "r1" });
    const blob = new Blob([new Uint8Array([1, 2, 3])]);
    const bitmap = {} as ImageBitmap;
    transport.emit({
      kind: "result",
      itemId: "r1",
      bitmap,
      blob,
      width: 10,
      height: 10,
      tileSizeUsed: 256,
      elapsedMs: 5,
    });
    const result = await promise;
    expect(result.itemId).toBe("r1");
    expect(result.blob).toBe(blob);
    expect(result.bitmap).toBe(bitmap);
    expect(result.width).toBe(10);
    expect(result.height).toBe(10);
    expect(result.tileSizeUsed).toBe(256);
  });

  it("encode sends encode message and resolves on encode-result", async () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    const bitmap = {} as ImageBitmap;
    const promise = client.encode(bitmap, { format: "webp", itemId: "e1" });
    expect(transport.inbound).toContainEqual({
      kind: "encode",
      itemId: "e1",
      bitmap,
      format: "webp",
    });
    const blob = new Blob([new Uint8Array([9])]);
    transport.emit({ kind: "encode-result", itemId: "e1", blob });
    expect(await promise).toBe(blob);
  });

  it("transport onerror fails all pending and sets error state", async () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    const file = new File([new Uint8Array([0])], "y.png", { type: "image/png" });
    const promise = client.process({ file, itemId: "y" });
    transport.emitError(new ErrorEvent("error", { message: "boom" }));
    await expect(promise).rejects.toMatchObject({ _tag: "WorkerCrash" });
    expect(client.state).toBe("error");
  });

  it("cancel via AbortSignal rejects process with Cancelled", async () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    transport.emit({
      kind: "state",
      state: "ready",
      backend: "wasm",
      modelId: "x",
      tileSize: 256,
    });
    const ac = new AbortController();
    const file = new File([new Uint8Array([0])], "z.png", { type: "image/png" });
    const promise = client.process({ file, itemId: "z", signal: ac.signal });
    ac.abort();
    transport.emit({ kind: "cancel", itemId: "z" } as unknown as HostOutbound);
    await expect(promise).rejects.toMatchObject({ _tag: "Cancelled" });
    expect(transport.inbound.some((m) => m.kind === "cancel" && m.itemId === "z")).toBe(true);
  });

  it("dispose terminates transport and sends dispose message", () => {
    const transport = makeMockTransport();
    const client = createInferenceClient({ transport });
    let terminated = false;
    transport.terminate = () => {
      terminated = true;
    };
    client.dispose();
    expect(terminated).toBe(true);
    expect(transport.inbound).toContainEqual({ kind: "dispose" });
  });
});
