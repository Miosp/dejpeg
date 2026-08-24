import { OffscreenCanvasPool } from "../../src/host/OffscreenCanvasPool.js";
import { describe, it, expect } from "bun:test";

describe("OffscreenCanvasPool", () => {
  it("returns a canvas with the requested dimensions", () => {
    const pool = new OffscreenCanvasPool();
    const canvas = pool.acquire(64, 64);
    expect(canvas.width).toBe(64);
    expect(canvas.height).toBe(64);
  });

  it("reuses a released canvas of the same dimensions", () => {
    const pool = new OffscreenCanvasPool();
    const a = pool.acquire(64, 64);
    pool.release(a);
    const b = pool.acquire(64, 64);
    expect(b).toBe(a); // same instance
  });

  it("allocates a new canvas for different dimensions", () => {
    const pool = new OffscreenCanvasPool();
    const a = pool.acquire(64, 64);
    const b = pool.acquire(128, 128);
    expect(b).not.toBe(a);
    expect(b.width).toBe(128);
  });
});
