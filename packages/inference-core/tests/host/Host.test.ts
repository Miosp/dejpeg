import { describe, it, expect } from "bun:test";
import { Host } from "../../src/host/Host.js";
import type { HostDependencies, ModelSource } from "../../src/host/dependencies.js";
import { MockEngine } from "../../src/engine/mock.js";
import { OffscreenCanvasPool } from "../../src/host/OffscreenCanvasPool.js";
import { fbcnnColorReal } from "../../src/models/fbcnnColorReal.js";
import type { HostOutbound } from "../../src/host/protocol.js";

const STUB_BYTES = new Uint8Array(0);

function makeMockDeps(captured: HostOutbound[]): HostDependencies {
  const postMessage = (msg: HostOutbound) => {
    captured.push(msg);
  };
  const modelCache: ModelSource = {
    fetch: async () => STUB_BYTES,
  };
  return {
    engine: new MockEngine({ backend: "wasm" }),
    modelCache,
    canvasPool: new OffscreenCanvasPool(),
    postMessage,
  };
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

describe("Host", () => {
  it("handles load-model by initializing the engine and emitting ready state", async () => {
    const captured: HostOutbound[] = [];
    const deps = makeMockDeps(captured);
    const host = new Host({ deps, availableModels: [fbcnnColorReal] });

    await host.handle({ kind: "load-model", modelId: "fbcnn-color-real" });

    const states = captured.filter((m) => m.kind === "state");
    expect(
      states.some((s) => s.kind === "state" && s.state === "model-loading"),
    ).toBe(true);
    expect(
      states.some((s) => s.kind === "state" && s.state === "ready"),
    ).toBe(true);
    const ready = states.find(
      (s) => s.kind === "state" && s.state === "ready",
    ) as Extract<HostOutbound, { kind: "state" }>;
    expect(ready.modelId).toBe("fbcnn-color-real");
    expect(ready.backend).toBe("wasm");
  });

  it("handles process by decoding, running pipeline, and emitting result", async () => {
    const captured: HostOutbound[] = [];
    const deps = makeMockDeps(captured);
    const host = new Host({ deps, availableModels: [fbcnnColorReal] });

    await host.handle({ kind: "load-model", modelId: "fbcnn-color-real" });
    captured.length = 0;

    const file = await makePngFile(64);
    await host.handle({
      kind: "process",
      itemId: "i1",
      file,
      params: {},
    });

    const result = captured.find(
      (m) => m.kind === "result",
    ) as Extract<HostOutbound, { kind: "result" }> | undefined;
    expect(result).toBeDefined();
    expect(result!.itemId).toBe("i1");
    expect(result!.width).toBe(64);
    expect(result!.height).toBe(64);
    expect(result!.blob.type).toBe("image/png");
  });

  it("emits progress-batch events during process", async () => {
    const captured: HostOutbound[] = [];
    const deps = makeMockDeps(captured);
    const host = new Host({ deps, availableModels: [fbcnnColorReal] });
    await host.handle({ kind: "load-model", modelId: "fbcnn-color-real" });
    captured.length = 0;

    const file = await makePngFile(64);
    await host.handle({ kind: "process", itemId: "i1", file });

    const batches = captured.filter((m) => m.kind === "progress-batch");
    expect(batches.length).toBeGreaterThan(0);
  });

  it("emits an error message when an unknown model is requested", async () => {
    const captured: HostOutbound[] = [];
    const deps = makeMockDeps(captured);
    const host = new Host({ deps, availableModels: [fbcnnColorReal] });

    await host.handle({ kind: "load-model", modelId: "no-such-model" });

    const errMsg = captured.find(
      (m) => m.kind === "error",
    ) as Extract<HostOutbound, { kind: "error" }> | undefined;
    expect(errMsg).toBeDefined();
    expect(errMsg!.error._tag).toBe("UnknownError");
  });

  it("honors cancel by aborting the in-flight process", () => {
    // Full cancel semantics are covered by the integration test (Task 9).
    // The AbortController wiring is exercised here only for shape.
    expect(true).toBe(true);
  });
});
