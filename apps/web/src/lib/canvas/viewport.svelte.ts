import {
  type Viewport,
  fitToScreen,
  zoomToCursor,
  pan as panVp,
  clampScale,
} from "./viewport";

export class ViewportState {
  vp = $state<Viewport>({ scale: 1, offsetX: 0, offsetY: 0 });
  #targetScale = 1;
  #rafId: number | null = null;
  #canvasW = 0;
  #canvasH = 0;

  setCanvasSize(w: number, h: number) {
    this.#canvasW = w;
    this.#canvasH = h;
  }

  fit(imgW: number, imgH: number) {
    this.cancelSmooth();
    const next = fitToScreen(this.#canvasW, this.#canvasH, imgW, imgH);
    this.#targetScale = next.scale;
    this.vp = next;
  }

  zoom(cursorX: number, cursorY: number, factor: number) {
    this.cancelSmooth();
    const newScale = clampScale(this.vp.scale * factor);
    this.vp = zoomToCursor(this.vp, cursorX, cursorY, newScale);
    this.#targetScale = newScale;
  }

  smoothZoom(cursorX: number, cursorY: number, targetFactor: number) {
    this.#targetScale = clampScale(this.vp.scale * targetFactor);
    if (this.#rafId !== null) return;

    const startScale = this.vp.scale;
    const startTime = performance.now();
    const duration = 150;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / duration);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      const currentScale = startScale + (this.#targetScale - startScale) * eased;
      this.vp = zoomToCursor(this.vp, cursorX, cursorY, currentScale);

      if (t < 1) {
        this.#rafId = requestAnimationFrame(tick);
      } else {
        this.#rafId = null;
      }
    };
    this.#rafId = requestAnimationFrame(tick);
  }

  pan(dx: number, dy: number) {
    this.vp = panVp(this.vp, dx, dy);
  }

  reset(imgW: number, imgH: number) {
    this.fit(imgW, imgH);
  }

  cancelSmooth() {
    if (this.#rafId !== null) {
      cancelAnimationFrame(this.#rafId);
      this.#rafId = null;
    }
  }
}

// Shared singleton — imported by both CanvasStage (rendering) and EditorApp (keyboard shortcuts)
export const viewport = new ViewportState();
