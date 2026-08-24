export interface Tile {
  /** Index in row-major order across the grid. */
  index: number;
  /** Top-left pixel coordinates within the source image. */
  x0: number;
  y0: number;
  /** Tile pixel dimensions (always tileSize x tileSize for now). */
  w: number;
  h: number;
}

export interface TilePlan {
  width: number;
  height: number;
  tileSize: number;
  overlap: number;
  tiles: readonly Tile[];
}

/**
 * Compute a tile grid for an image.
 *
 * Last-tile-snap: if a tile would extend past the image, shift it inward so
 * it is a full tile (with more overlap on its interior side). No partials.
 */
export function computePlan(
  width: number,
  height: number,
  tileSize: number,
  overlap: number,
): TilePlan {
  if (tileSize <= 0)
    throw new Error(`tileSize must be positive, got ${tileSize}`);
  if (overlap < 0)
    throw new Error(`overlap must be non-negative, got ${overlap}`);
  if (overlap >= tileSize)
    throw new Error(`overlap ${overlap} >= tileSize ${tileSize}`);

  const step = tileSize - overlap;

  const xs = computeAxisOrigins(width, tileSize, step);
  const ys = computeAxisOrigins(height, tileSize, step);

  const tiles: Tile[] = [];
  let i = 0;
  for (const y0 of ys) {
    for (const x0 of xs) {
      tiles.push({ index: i++, x0, y0, w: tileSize, h: tileSize });
    }
  }

  return { width, height, tileSize, overlap, tiles };
}

function computeAxisOrigins(
  length: number,
  tileSize: number,
  step: number,
): number[] {
  if (length <= tileSize) return [0];
  const origins: number[] = [];
  let pos = 0;
  while (pos + tileSize <= length) {
    origins.push(pos);
    pos += step;
  }
  const last = origins[origins.length - 1]!;
  if (last + tileSize < length) {
    origins.push(length - tileSize);
  }
  return origins;
}
