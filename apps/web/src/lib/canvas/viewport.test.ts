import { test, expect } from "bun:test";
import {
  type Viewport,
  fitToScreen,
  zoomToCursor,
  pan,
  clampScale,
  MIN_SCALE,
  MAX_SCALE,
} from "./viewport";

test("fitToScreen centers image with correct scale", () => {
  const vp = fitToScreen(800, 600, 1600, 1200);
  expect(vp.scale).toBeCloseTo(0.5, 5);
  expect(vp.offsetX).toBe(0);
  expect(vp.offsetY).toBe(0);
});

// Brief specified offsetX=100, but that is not centered: scaled image is
// 400*0.5=200 wide, centered in 800-wide canvas => (800-200)/2 = 300.
test("fitToScreen handles tall image", () => {
  const vp = fitToScreen(800, 600, 400, 1200);
  expect(vp.scale).toBeCloseTo(0.5, 5);
  expect(vp.offsetX).toBe(300);
  expect(vp.offsetY).toBe(0);
});

test("clampScale enforces range", () => {
  expect(clampScale(0.001)).toBe(MIN_SCALE);
  expect(clampScale(100)).toBe(MAX_SCALE);
  expect(clampScale(1.5)).toBe(1.5);
});

test("zoomToCursor keeps cursor point fixed", () => {
  const vp: Viewport = { scale: 1, offsetX: 0, offsetY: 0 };
  const next = zoomToCursor(vp, 400, 300, 2);
  expect(next.scale).toBe(2);
  expect(next.offsetX).toBe(-400);
  expect(next.offsetY).toBe(-300);
});

test("pan moves offset by delta", () => {
  const vp: Viewport = { scale: 1, offsetX: 100, offsetY: 200 };
  const next = pan(vp, 50, -30);
  expect(next.offsetX).toBe(150);
  expect(next.offsetY).toBe(170);
  expect(next.scale).toBe(1);
});
