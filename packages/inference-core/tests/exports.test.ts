import { describe, it, expect } from "bun:test";
import * as ic from "../src/index.js";

describe("default barrel exports", () => {
  it("exports createInferenceClient", () => {
    expect(typeof ic.createInferenceClient).toBe("function");
  });

  it("exports error classes for instanceof checks", () => {
    expect(typeof ic.ImageDecodeError).toBe("function");
    expect(typeof ic.NoModelLoaded).toBe("function");
    expect(typeof ic.ClientBusy).toBe("function");
  });

  it("exports MODELS registry", () => {
    expect(Array.isArray(ic.MODELS)).toBe(true);
    expect(ic.MODELS.length).toBeGreaterThan(0);
  });

  it("does NOT export internal modules", () => {
    const ns = ic as Record<string, unknown>;
    expect(ns.processImage).toBeUndefined();
    expect(ns.computePlan).toBeUndefined();
    expect(ns.combineTiles).toBeUndefined();
    expect(ns.serializeError).toBeUndefined();
    expect(ns.Host).toBeUndefined();
  });
});
