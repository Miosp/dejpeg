import { test, expect } from "bun:test";
import { computePlan } from "../src/tiling/slicer.js";
import {
  combineTiles,
  finalizeBlend,
  tileWeightCanvas,
} from "../src/tiling/blender.js";

test("single-tile plan with zero overlap: weights all 1", () => {
  const plan = computePlan(64, 64, 64, 0);
  const w = tileWeightCanvas(plan.tiles[0]!, plan);
  expect(w.length).toBe(64 * 64);
  for (const v of w) expect(v).toBe(1);
});

test("tile at image edge on all sides: weights all 1", () => {
  const plan = computePlan(256, 256, 256, 32);
  const w = tileWeightCanvas(plan.tiles[0]!, plan);
  for (const v of w) expect(v).toBe(1);
});

test("interior tile: overlap margins ramp, center is 1", () => {
  // Image 1024 wide, tile 256, overlap 32 → step 224 → tiles at x=0,224,448,...
  const plan = computePlan(1024, 256, 256, 32);
  const interior = plan.tiles.find((t) => t.x0 === 448 && t.y0 === 0);
  expect(interior).toBeDefined();
  if (!interior) return;
  const w = tileWeightCanvas(interior, plan);

  // (0,0) in tile coords: x in left overlap (not at image edge), y at top edge.
  // x-weight tiny, y-weight 1 → product tiny.
  expect(w[0]).toBeLessThan(0.2);
  // Center of tile: interior on both axes.
  expect(w[128 * 256 + 128]!).toBe(1);
  // Left edge midpoint: deep in left overlap.
  expect(w[128 * 256 + 0]!).toBeLessThan(0.2);
});

test("cosine weight is monotonic across overlap margin", () => {
  const plan = computePlan(1024, 1024, 256, 64);
  const interior = plan.tiles.find((t) => t.x0 === 192 && t.y0 === 192);
  expect(interior).toBeDefined();
  if (!interior) return;
  const w = tileWeightCanvas(interior, plan);

  // Walk along middle row from left edge inward; weights should be increasing.
  const y = 128;
  let prev = -1;
  for (let x = 0; x < 96; x++) {
    const v = w[y * 256 + x]!;
    expect(v).toBeGreaterThanOrEqual(prev);
    prev = v;
  }
  // Innermost overlap pixel (x = overlap - 1 = 63) hits weight 1.
  expect(w[y * 256 + 63]!).toBeCloseTo(1, 5);
});

test("combineTiles of two constant tiles yields the constant everywhere", () => {
  // 96×64 image, tile 64, overlap 32, step 32 → tiles at x=0 and x=32.
  const plan = computePlan(96, 64, 64, 32);
  expect(plan.tiles).toHaveLength(2);

  const channels = 1;
  const tileOutputs = new Map<number, Float32Array>();
  for (const t of plan.tiles) {
    tileOutputs.set(t.index, new Float32Array(channels * 64 * 64).fill(0.5));
  }
  const acc = combineTiles(plan, channels, tileOutputs);
  const final = finalizeBlend(acc, channels, plan.width * plan.height);
  for (const v of final) expect(Math.abs(v - 0.5)).toBeLessThan(1e-5);
});

test("combineTiles of two contrasting tiles blends smoothly in overlap", () => {
  // Left tile all 0.0, right tile all 1.0. Overlap should produce a ramp
  // between them; both halves outside overlap should be exactly the source.
  const plan = computePlan(96, 64, 64, 32);
  const channels = 1;
  const tileOutputs = new Map<number, Float32Array>();
  tileOutputs.set(plan.tiles[0]!.index, new Float32Array(64 * 64).fill(0.0));
  tileOutputs.set(plan.tiles[1]!.index, new Float32Array(64 * 64).fill(1.0));
  const acc = combineTiles(plan, channels, tileOutputs);
  const final = finalizeBlend(acc, channels, plan.width * plan.height);

  // Far-left (image x=0): only left tile covers, value 0.
  expect(final[64 * 0 + 0]!).toBeCloseTo(0, 6);
  // Far-right (image x=95): only right tile covers, value 1.
  expect(final[64 * 0 + 95]!).toBeCloseTo(1, 6);
  // Midpoint of overlap (image x≈47/48): blend should be between 0 and 1.
  const mid = final[64 * 0 + 48]!;
  expect(mid).toBeGreaterThan(0);
  expect(mid).toBeLessThan(1);
});

test("weights accumulate to a positive value at every covered pixel", () => {
  const plan = computePlan(300, 300, 128, 16);
  const tileOutputs = new Map<number, Float32Array>();
  for (const t of plan.tiles) {
    tileOutputs.set(t.index, new Float32Array(128 * 128).fill(0.7));
  }
  const acc = combineTiles(plan, 1, tileOutputs);
  for (let i = 0; i < plan.width * plan.height; i++) {
    expect(acc.weights[i]).toBeGreaterThan(0);
  }
});

test("3-channel CHW layout: each channel blended independently", () => {
  const plan = computePlan(96, 64, 64, 32);
  const channels = 3;
  const tileOutputs = new Map<number, Float32Array>();
  for (const t of plan.tiles) {
    const data = new Float32Array(channels * 64 * 64);
    // Channel 0 = 0.1, channel 1 = 0.5, channel 2 = 0.9.
    for (let c = 0; c < channels; c++) {
      const val = c === 0 ? 0.1 : c === 1 ? 0.5 : 0.9;
      data.fill(val, c * 64 * 64, (c + 1) * 64 * 64);
    }
    tileOutputs.set(t.index, data);
  }
  const acc = combineTiles(plan, channels, tileOutputs);
  const final = finalizeBlend(acc, channels, plan.width * plan.height);
  const n = plan.width * plan.height;
  for (let i = 0; i < n; i++) {
    expect(Math.abs(final[i]! - 0.1)).toBeLessThan(1e-5);
    expect(Math.abs(final[n + i]! - 0.5)).toBeLessThan(1e-5);
    expect(Math.abs(final[2 * n + i]! - 0.9)).toBeLessThan(1e-5);
  }
});

test("combineTiles throws when a tile output is missing", () => {
  const plan = computePlan(96, 64, 64, 32);
  expect(() => combineTiles(plan, 1, new Map())).toThrow(/missing tile output/);
});

test("non-256 tile: right overlap margin ramps on interior tile (tileSize threading regression)", () => {
  // tileSize=64, overlap=16. Pick a tile that is interior on the x-axis (not
  // at the image right edge) so the right feather margin must fire. Tile sits
  // at y0=0 (atTopEdge → wy = 1 everywhere), so the canvas equals wx directly.
  const plan = computePlan(256, 64, 64, 16);
  const tile = plan.tiles.find((t) => t.x0 === 144 && t.y0 === 0);
  expect(tile).toBeDefined();
  if (!tile) return;
  expect(tile.x0 + 64).toBeLessThan(plan.width); // not at image right edge

  const w = tileWeightCanvas(tile, plan);
  const T = 64;
  const O = 16;
  const row = 32 * T; // middle row; wy = 1 → w[row + pos] = wx(pos)

  // Right margin covers pos ∈ [T - O, T) = [48, 64). Distance from outer edge
  // is d = T - pos ∈ [O, 1]. Weight = 0.5 * (1 - cos(π * d / O)).
  for (let pos = T - O; pos < T; pos++) {
    const d = T - pos;
    const expected = 0.5 * (1 - Math.cos((Math.PI * d) / O));
    expect(w[row + pos]!).toBeCloseTo(expected, 6);
  }

  // Explicit anchors. If the brief's `tileSize = 256` default were restored,
  // the right-margin branch never fires for this 64-wide tile, so every
  // position in [48, 64) returns weight 1 and these fail hard.
  expect(w[row + (T - O)]!).toBeCloseTo(1, 6); // pos=48, d=16=O → weight 1
  expect(w[row + (T - 1)]!).toBeLessThan(0.2); // pos=63, d=1 → ~0.0096
});
