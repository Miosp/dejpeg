import { describe, it, expect } from "bun:test";
import { createInferenceClient } from "../src/client/InferenceClient.js";
import { Host } from "../src/host/Host.js";
import type { HostTransport } from "../src/client/transport.js";
import type { ModelSource } from "../src/host/dependencies.js";
import type { HostOutbound, ClientInbound } from "../src/host/protocol.js";
import { MockEngine } from "../src/engine/mock.js";
import { OffscreenCanvasPool } from "../src/host/OffscreenCanvasPool.js";
import { MODELS } from "../src/models/index.js";
import { NoModelLoaded, ClientBusy, Cancelled } from "../src/errors.js";

const STUB_BYTES = new Uint8Array(0);

/**
 * In-process integration: client + host wired through a mock transport.
 * No real Worker, no real onnxruntime-web. Exercises the full message flow.
 */
function makeWiredPair() {
  let clientOnMessage: ((e: MessageEvent<HostOutbound>) => void) | null = null;
  let hostReceived: ((msg: ClientInbound) => void) | null = null;

  const clientTransport: HostTransport = {
    postMessage: (msg) => hostReceived?.(msg),
    get onmessage() {
      return clientOnMessage;
    },
    set onmessage(v) {
      clientOnMessage = v;
    },
    get onerror() {
      return null;
    },
    set onerror(_) {},
    terminate: () => {},
  };

  const hostPostMessage = (msg: HostOutbound) => {
    clientOnMessage?.({ data: msg } as MessageEvent<HostOutbound>);
  };

  const modelCache: ModelSource = {
    fetch: async () => STUB_BYTES,
  };

  const host = new Host({
    deps: {
      engine: new MockEngine(),
      modelCache,
      canvasPool: new OffscreenCanvasPool(),
      postMessage: hostPostMessage,
    },
    availableModels: MODELS,
  });

  hostReceived = (msg) => {
    host.handle(msg);
  };

  const client = createInferenceClient({
    transport: clientTransport,
    availableModels: MODELS,
  });

  return { client, host };
}

async function makePngFile(size = 64): Promise<File> {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.createImageData(size, size);
  for (let i = 0; i < imageData.data.length; i += 4) {
    imageData.data[i] = 128;
    imageData.data[i + 1] = 128;
    imageData.data[i + 2] = 128;
    imageData.data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  return new File([blob], "test.png", { type: "image/png" });
}

describe("integration: client ↔ host", () => {
  it("full flow: load-model, process, encode", async () => {
    const { client } = makeWiredPair();

    await client.loadModel("fbcnn-color-real");
    expect(client.state).toBe("ready");
    expect(client.modelId).toBe("fbcnn-color-real");

    const file = await makePngFile(64);
    const result = await client.process({ file });
    expect(result.width).toBe(64);
    expect(result.height).toBe(64);
    expect(result.blob.type).toBe("image/png");

    const jpegBlob = await client.encode(result.bitmap, {
      format: "jpeg",
      quality: 85,
    });
    expect(jpegBlob.type).toBe("image/jpeg");

    expect(client.state).toBe("ready");
  });

  it("process throws NoModelLoaded before any model is loaded", async () => {
    const { client } = makeWiredPair();
    const file = await makePngFile(32);
    expect(() => client.process({ file })).toThrow(NoModelLoaded);
  });

  it("concurrent process calls throw ClientBusy on the second", async () => {
    const { client } = makeWiredPair();
    await client.loadModel("fbcnn-color-real");
    const file = await makePngFile(32);
    const first = client.process({ file });
    expect(() => client.process({ file })).toThrow(ClientBusy);
    await first;
  });

  it("subscribe fires across state transitions", async () => {
    const { client } = makeWiredPair();
    const snapshots: string[] = [];
    client.subscribe((s) => snapshots.push(s.state));
    await client.loadModel("fbcnn-color-real");
    expect(snapshots).toContain("model-loading");
    expect(snapshots).toContain("ready");
  });

  it("abort signal during process rejects with Cancelled", async () => {
    const { client } = makeWiredPair();
    await client.loadModel("fbcnn-color-real");
    const controller = new AbortController();
    const file = await makePngFile(32);
    const promise = client.process({ file, signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toBeInstanceOf(Cancelled);
  });

  it("capabilities are populated after load-model", async () => {
    const { client } = makeWiredPair();
    expect(client.capabilities).toBeNull();
    await client.loadModel("fbcnn-color-real");
    expect(client.capabilities).not.toBeNull();
    expect(client.capabilities!.nativeDecoders).toContain("png");
  });
});
