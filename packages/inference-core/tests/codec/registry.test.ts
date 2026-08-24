import { WASM_CODECS, getWasmLoader, requiresWasm } from "../../src/codec/registry.js";
import { describe, it, expect } from "bun:test";

describe("codec registry", () => {
  it("lists known WASM formats", () => {
    const formats = WASM_CODECS.map((c) => c.format);
    expect(formats).toContain("heic");
    expect(formats).toContain("tiff");
    expect(formats).toContain("avif");
  });

  it("getWasmLoader returns a function for known formats", () => {
    const loader = getWasmLoader("heic");
    expect(typeof loader).toBe("function");
  });

  it("getWasmLoader returns undefined for native-handled formats", () => {
    expect(getWasmLoader("png")).toBeUndefined();
    expect(getWasmLoader("jpeg")).toBeUndefined();
  });

  it("requiresWasm distinguishes WASM-only from native", () => {
    expect(requiresWasm("heic")).toBe(true);
    expect(requiresWasm("png")).toBe(false);
  });
});
