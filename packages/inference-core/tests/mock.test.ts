import { test, expect } from "bun:test";
import { MockEngine } from "../src/engine/mock.js";
import type { Tensor } from "../src/engine/types.js";

function tile(h = 64, w = 64, fill = 0): Tensor {
  return {
    data: new Float32Array(1 * 3 * h * w).fill(fill),
    shape: [1, 3, h, w],
  };
}

test("MockEngine records calls", async () => {
  const e = new MockEngine();
  await e.run({ input: tile() });
  await e.run({ input: tile() });
  expect(e.calls.length).toBe(2);
});

test("MockEngine passthrough returns the input", async () => {
  const e = new MockEngine();
  const input = tile(64, 64, 0.5);
  const out = await e.run({ input });
  expect(out.output!).toBe(input);
});

test("MockEngine failWith is invoked and throws", async () => {
  const e = new MockEngine({
    failWith: () => new Error("boom"),
  });
  await expect(e.run({ input: tile() })).rejects.toThrow("boom");
});

test("MockEngine failAtTileSizes triggers at matching size", async () => {
  const e = new MockEngine({ failAtTileSizes: [256, 128] });
  await expect(e.run({ input: tile(256, 256) })).rejects.toThrow(/OOM at tile 256/);
  await expect(e.run({ input: tile(128, 128) })).rejects.toThrow(/OOM at tile 128/);
  // Smaller sizes pass through
  const out = await e.run({ input: tile(64, 64) });
  expect(out.output!.shape).toEqual([1, 3, 64, 64]);
});

test("MockEngine produce returns configured output", async () => {
  const custom: Tensor = { data: new Float32Array([1]), shape: [1] };
  const e = new MockEngine({ produce: () => ({ output: custom }) });
  const out = await e.run({ input: tile() });
  expect(out.output!).toBe(custom);
});
