import { test, expect, describe, it } from "bun:test";
import { Effect } from "effect";
import {
  BackendUnavailable,
  Cancelled,
  ClientBusy,
  ImageDecodeError,
  NetworkError,
  NoModelLoaded,
  TileAllocationFailure,
  TileFloorExceeded,
  UnknownError,
} from "../src/errors.js";
import type { ModelError } from "../src/errors.js";

test("tagged errors carry their payload", () => {
  const e = new NetworkError({
    url: "https://example.com/model.onnx",
    status: 404,
  });
  expect(e._tag).toBe("NetworkError");
  expect(e.url).toBe("https://example.com/model.onnx");
  expect(e.status).toBe(404);
  expect(e instanceof Error).toBe(true);
});

test("tagged errors are distinguishable via _tag in a catch block", () => {
  const fail = Effect.fail(new TileFloorExceeded({ modelId: "x", floor: 32 }));
  const exit = Effect.runSyncExit(fail);
  expect(exit._tag).toBe("Failure");
  if (exit._tag === "Failure") {
    expect(exit.cause._tag).toBe("Fail");
  }
});

test("each error category has a unique _tag", () => {
  const tags = new Set<string>();
  for (const Err of [
    NetworkError,
    BackendUnavailable,
    TileAllocationFailure,
    TileFloorExceeded,
    ImageDecodeError,
    Cancelled,
  ]) {
    const instance = new (Err as any)({
      url: "x",
      backend: "webgpu",
      attemptedSize: 256,
      modelId: "x",
      floor: 32,
      filename: "x",
      id: "x",
    });
    expect(tags.has(instance._tag)).toBe(false);
    tags.add(instance._tag);
  }
});

describe("new error classes", () => {
  it("NoModelLoaded carries the operation", () => {
    const e = new NoModelLoaded({ operation: "process" });
    expect(e._tag).toBe("NoModelLoaded");
    expect(e.operation).toBe("process");
  });

  it("UnknownError carries message and optional cause", () => {
    const e = new UnknownError({ message: "boom", cause: new Error("root") });
    expect(e._tag).toBe("UnknownError");
    expect(e.message).toBe("boom");
    expect((e.cause as Error).message).toBe("root");
  });

  it("ClientBusy carries the attempted operation", () => {
    const e = new ClientBusy({ operation: "load-model" });
    expect(e._tag).toBe("ClientBusy");
    expect(e.operation).toBe("load-model");
  });

  it("all three are part of the ModelError union", () => {
    // Type-level check — compiles only if the union includes them.
    const a: ModelError = new NoModelLoaded({ operation: "encode" });
    const b: ModelError = new UnknownError({ message: "x" });
    const c: ModelError = new ClientBusy({ operation: "process" });
    expect([a, b, c]).toHaveLength(3);
  });
});
