type Dimensions = string; // `${width}x${height}`

/**
 * Pools OffscreenCanvas instances by dimensions. Reused across encode calls
 * of the same shape. Per spec perf invariant #5/#13.
 *
 * Idle-eviction is intentionally omitted in v1 — pool stays small in
 * practice because image dimensions cluster around common sizes. Add a
 * timeout if memory pressure becomes measurable.
 */
export class OffscreenCanvasPool {
  private pool = new Map<Dimensions, OffscreenCanvas[]>();

  acquire(width: number, height: number): OffscreenCanvas {
    const key = `${width}x${height}`;
    const stack = this.pool.get(key);
    if (stack && stack.length > 0) {
      return stack.pop()!;
    }
    return new OffscreenCanvas(width, height);
  }

  release(canvas: OffscreenCanvas): void {
    const key = `${canvas.width}x${canvas.height}`;
    let stack = this.pool.get(key);
    if (!stack) {
      stack = [];
      this.pool.set(key, stack);
    }
    // Clear before returning to pool so the next user starts clean.
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    stack.push(canvas);
  }
}
