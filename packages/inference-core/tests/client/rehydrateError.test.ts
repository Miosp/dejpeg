import { rehydrateError } from "../../src/client/rehydrateError.js";
import {
  ImageDecodeError,
  CanvasCapExceeded,
  UnknownError,
  NetworkError,
  Cancelled,
} from "../../src/errors.js";
import { describe, it, expect } from "bun:test";

describe("rehydrateError", () => {
  it("rehydrates ImageDecodeError with its fields", () => {
    const serialized = { _tag: "ImageDecodeError", filename: "x.jpg" };
    const rehydrated = rehydrateError(serialized);
    expect(rehydrated).toBeInstanceOf(ImageDecodeError);
    expect((rehydrated as ImageDecodeError).filename).toBe("x.jpg");
  });

  it("rehydrates CanvasCapExceeded", () => {
    const serialized = {
      _tag: "CanvasCapExceeded",
      width: 10000,
      height: 10000,
      browserCap: 268000000,
    };
    const rehydrated = rehydrateError(serialized);
    expect(rehydrated).toBeInstanceOf(CanvasCapExceeded);
  });

  it("rehydrates NetworkError with url and status", () => {
    const rehydrated = rehydrateError({
      _tag: "NetworkError",
      url: "https://x.onnx",
      status: 503,
    });
    expect(rehydrated).toBeInstanceOf(NetworkError);
    expect((rehydrated as NetworkError).url).toBe("https://x.onnx");
    expect((rehydrated as NetworkError).status).toBe(503);
  });

  it("rehydrates Cancelled with id", () => {
    const rehydrated = rehydrateError({ _tag: "Cancelled", id: "i1" });
    expect(rehydrated).toBeInstanceOf(Cancelled);
    expect((rehydrated as Cancelled).id).toBe("i1");
  });

  it("falls back to UnknownError for unrecognized _tag", () => {
    const serialized = { _tag: "MysteryError", message: "weird" };
    const rehydrated = rehydrateError(serialized);
    expect(rehydrated).toBeInstanceOf(UnknownError);
  });

  it("preserves message field on UnknownError fallback", () => {
    const serialized = { _tag: "Mystery", message: "weird" };
    const rehydrated = rehydrateError(serialized) as UnknownError;
    expect(rehydrated.message).toBe("weird");
  });
});
