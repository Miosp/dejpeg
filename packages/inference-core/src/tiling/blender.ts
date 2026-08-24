import type { Tile, TilePlan } from "./slicer.js";

/**
 * Build a per-tile weight canvas. Cosine-feather from ~0 at the overlap margin
 * edge to 1 in the interior. Edges of the image (no neighbor on that side) get
 * weight 1 across the full tile — no feathering into nothing.
 *
 * For a tile of size T with overlap O, pixels inside [O, T-O) get weight 1.
 * Pixels in the overlap margin ramp as 0.5*(1 - cos(pi * d / O)) where d is the
 * distance from the outer edge.
 *
 * If a tile sits at the image's outer edge on a given axis, its overlap margin
 * on that side is forced to weight 1 — otherwise we'd darken the image borders.
 */
export function tileWeightCanvas(tile: Tile, plan: TilePlan): Float32Array {
  const { tileSize, overlap } = plan;

  const atLeftEdge = tile.x0 === 0;
  const atRightEdge = tile.x0 + tileSize >= plan.width;
  const atTopEdge = tile.y0 === 0;
  const atBottomEdge = tile.y0 + tileSize >= plan.height;

  const weights = new Float32Array(tileSize * tileSize);
  for (let y = 0; y < tileSize; y++) {
    const wy = axisWeight(y, overlap, tileSize, atTopEdge, atBottomEdge);
    for (let x = 0; x < tileSize; x++) {
      const wx = axisWeight(x, overlap, tileSize, atLeftEdge, atRightEdge);
      weights[y * tileSize + x] = wx * wy;
    }
  }
  return weights;
}

/**
 * Per-axis feather weight for a single coordinate.
 *
 * - `pos` in `[0, overlap)`: left margin. Distance from outer edge is `pos + 1`.
 * - `pos` in `[tileSize - overlap, tileSize)`: right margin. Distance is
 *   `tileSize - pos`.
 * - Otherwise: interior, weight 1.
 *
 * `atLowEdge`/`atHighEdge` force the corresponding margin to 1 (image border).
 */
function axisWeight(
  pos: number,
  overlap: number,
  tileSize: number,
  atLowEdge: boolean,
  atHighEdge: boolean,
): number {
  if (overlap === 0) return 1;
  if (pos < overlap) {
    if (atLowEdge) return 1;
    const d = pos + 1;
    return 0.5 * (1 - Math.cos((Math.PI * d) / overlap));
  }
  if (pos >= tileSize - overlap) {
    if (atHighEdge) return 1;
    const d = tileSize - pos;
    return 0.5 * (1 - Math.cos((Math.PI * d) / overlap));
  }
  return 1;
}

/**
 * Combine tile outputs into a full-image accumulator.
 *
 * `tileOutputs` maps tile.index → predicted pixel data, Float32Array of size
 * `channels * tileSize * tileSize` in CHW layout.
 *
 * Returns the accumulated weighted sum and the accumulated weights. Call
 * `finalizeBlend` to divide and get final pixels.
 */
export function combineTiles(
  plan: TilePlan,
  channels: number,
  tileOutputs: Map<number, Float32Array>,
): { weighted: Float32Array; weights: Float32Array } {
  const { width, height, tileSize } = plan;
  const pixelCount = width * height;
  const weighted = new Float32Array(channels * pixelCount);
  const weights = new Float32Array(pixelCount);

  for (const tile of plan.tiles) {
    const out = tileOutputs.get(tile.index);
    if (!out) throw new Error(`missing tile output for index ${tile.index}`);
    const w = tileWeightCanvas(tile, plan);
    for (let dy = 0; dy < tileSize; dy++) {
      const y = tile.y0 + dy;
      if (y >= height) break;
      for (let dx = 0; dx < tileSize; dx++) {
        const x = tile.x0 + dx;
        if (x >= width) break;
        const wPx = w[dy * tileSize + dx]!;
        const wIdx = y * width + x;
        weights[wIdx]! += wPx;
        for (let c = 0; c < channels; c++) {
          const src = out[c * tileSize * tileSize + dy * tileSize + dx]!;
          const dst = c * pixelCount + wIdx;
          weighted[dst]! += wPx * src;
        }
      }
    }
  }

  return { weighted, weights };
}

/** Divide accumulated weighted values by accumulated weights → final pixels. */
export function finalizeBlend(
  accumulated: { weighted: Float32Array; weights: Float32Array },
  channels: number,
  pixelCount: number,
): Float32Array {
  const out = new Float32Array(channels * pixelCount);
  for (let c = 0; c < channels; c++) {
    const srcOff = c * pixelCount;
    for (let i = 0; i < pixelCount; i++) {
      const w = accumulated.weights[i]!;
      out[srcOff + i] = w === 0 ? 0 : accumulated.weighted[srcOff + i]! / w;
    }
  }
  return out;
}
