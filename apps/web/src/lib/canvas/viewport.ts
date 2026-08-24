export interface Viewport {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export const MIN_SCALE = 0.01;
export const MAX_SCALE = 50;

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
}

/**
 * Compute viewport that fits an image centered within a canvas.
 * canvasW/canvasH are CSS pixels (not device pixels).
 */
export function fitToScreen(
  canvasW: number,
  canvasH: number,
  imgW: number,
  imgH: number,
): Viewport {
  if (imgW === 0 || imgH === 0) return { scale: 1, offsetX: 0, offsetY: 0 };
  const scaleX = canvasW / imgW;
  const scaleY = canvasH / imgH;
  const scale = clampScale(Math.min(scaleX, scaleY));
  const offsetX = (canvasW - imgW * scale) / 2;
  const offsetY = (canvasH - imgH * scale) / 2;
  return { scale, offsetX, offsetY };
}

/**
 * Zoom while keeping the point under the cursor fixed.
 * cursorX/cursorY are in CSS pixels relative to the canvas.
 */
export function zoomToCursor(
  vp: Viewport,
  cursorX: number,
  cursorY: number,
  newScale: number,
): Viewport {
  const clamped = clampScale(newScale);
  const ratio = clamped / vp.scale;
  return {
    scale: clamped,
    offsetX: cursorX - (cursorX - vp.offsetX) * ratio,
    offsetY: cursorY - (cursorY - vp.offsetY) * ratio,
  };
}

/**
 * Pan by a delta in CSS pixels.
 */
export function pan(vp: Viewport, dx: number, dy: number): Viewport {
  return {
    scale: vp.scale,
    offsetX: vp.offsetX + dx,
    offsetY: vp.offsetY + dy,
  };
}
