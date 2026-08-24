import { test, expect } from "bun:test";
import { computePlan } from "../src/tiling/slicer.js";

test("image exactly one tile wide -> single tile at origin", () => {
  const plan = computePlan(256, 256, 256, 0);
  expect(plan.tiles).toHaveLength(1);
  expect(plan.tiles[0]).toMatchObject({ x0: 0, y0: 0, w: 256, h: 256 });
});

test("image smaller than tile -> single tile at origin", () => {
  const plan = computePlan(100, 100, 256, 0);
  expect(plan.tiles).toHaveLength(1);
  expect(plan.tiles[0]).toMatchObject({ x0: 0, y0: 0 });
});

test("2x2 grid with no overlap", () => {
  const plan = computePlan(512, 512, 256, 0);
  expect(plan.tiles).toHaveLength(4);
  const origins = new Set(plan.tiles.map((t) => `${t.x0},${t.y0}`));
  expect(origins).toEqual(new Set(["0,0", "256,0", "0,256", "256,256"]));
});

test("overlap increases tile count", () => {
  // 768 = 3*256 divides evenly: 9 tiles no overlap.
  // overlap=64 -> step=192 -> 4 origins + last-snap -> 16 tiles.
  const noOverlap = computePlan(768, 768, 256, 0);
  const withOverlap = computePlan(768, 768, 256, 64);
  expect(noOverlap.tiles).toHaveLength(9);
  expect(withOverlap.tiles).toHaveLength(16);
});

test("last-tile-snap covers the right edge", () => {
  // width=300, tile=256, overlap=0: positions 0 only, but 256+ doesn't fit,
  // and 0+256=256 < 300 so we snap one to 300-256=44.
  const plan = computePlan(300, 300, 256, 0);
  const xs = new Set(plan.tiles.map((t) => t.x0));
  expect(xs).toContain(0);
  expect(xs).toContain(44);
});

test("every tile is full-size (no partials)", () => {
  const plan = computePlan(1023, 768, 256, 32);
  for (const t of plan.tiles) {
    expect(t.w).toBe(256);
    expect(t.h).toBe(256);
  }
});

test("overlap >= tileSize throws", () => {
  expect(() => computePlan(512, 512, 256, 256)).toThrow();
  expect(() => computePlan(512, 512, 256, 300)).toThrow();
});

test("off-by-one: image 1024, tile 256, overlap 0 -> exactly 16 tiles", () => {
  const plan = computePlan(1024, 1024, 256, 0);
  expect(plan.tiles).toHaveLength(16);
});
