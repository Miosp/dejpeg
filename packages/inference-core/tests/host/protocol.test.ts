import { describe, expect, it } from "bun:test";
import {
  serializeError,
  type ProgressEvent,
  type ClientInbound,
  type HostOutbound,
} from "../../src/host/protocol.js";
import { NetworkError } from "../../src/errors.js";

describe("serializeError", () => {
  it("extracts _tag from a tagged error", () => {
    const e = new NetworkError({ url: "https://x", status: 503 });
    const s = serializeError(e);
    expect(s._tag).toBe("NetworkError");
    expect(s.url).toBe("https://x");
    expect(s.status).toBe(503);
  });

  it("adds Unknown tag for plain Error", () => {
    const s = serializeError(new Error("boom"));
    expect(s._tag).toBe("Unknown");
    expect(s.message).toBe("boom");
  });

  it("survives structured clone / JSON roundtrip", () => {
    const e = new NetworkError({ url: "https://y", status: 500 });
    const s = serializeError(e);
    const round = JSON.parse(JSON.stringify(s));
    expect(round).toEqual(s);
  });

  it("skips non-serializable fields", () => {
    const weird: Error & { fn?: unknown } = new Error("x");
    weird.fn = () => "unserializable";
    const s = serializeError(weird);
    expect("fn" in s).toBe(false);
    expect(s.message).toBe("x");
  });
});

describe("discriminated unions narrow by kind", () => {
  it("ClientInbound narrows", () => {
    const m: ClientInbound = { kind: "load-model", modelId: "fbcnn-color-real" };
    if (m.kind === "load-model") {
      expect(m.modelId).toBe("fbcnn-color-real");
    }
  });

  it("HostOutbound state narrows", () => {
    const m: HostOutbound = { kind: "state", state: "ready", backend: "webgpu" };
    if (m.kind === "state") {
      expect(m.backend).toBe("webgpu");
    }
  });

  it("ProgressEvent narrows", () => {
    const e: ProgressEvent = {
      kind: "boot",
      step: "wasm",
      loaded: 1,
      total: 4,
    };
    if (e.kind === "boot") {
      expect(e.step).toBe("wasm");
    }
  });

  it("progress-batch wraps event payload", () => {
    const m: HostOutbound = {
      kind: "progress-batch",
      events: [
        { kind: "image", itemId: "i1", stage: "tile", done: 2, total: 9 },
      ],
    };
    if (m.kind === "progress-batch" && m.events[0]!.kind === "image") {
      expect(m.events[0]!.stage).toBe("tile");
    }
  });

  it("encode stage exists in image progress", () => {
    const e: ProgressEvent = { kind: "image", itemId: "i", stage: "encode" };
    if (e.kind === "image") {
      expect(e.stage).toBe("encode");
    }
  });
});
