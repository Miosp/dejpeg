<script lang="ts">
  import { queue } from "../lib/state/queue.svelte";
  import { ui } from "../lib/state/ui.svelte";
  import { viewport } from "../lib/canvas/viewport.svelte";
  import ProgressTopBar from "./ProgressTopBar.svelte";
  import ErrorOverlay from "./ErrorOverlay.svelte";

  let canvas = $state<HTMLCanvasElement | null>(null);
  let canvasB = $state<HTMLCanvasElement | null>(null);
  let container = $state<HTMLDivElement | null>(null);

  let active = $derived(queue.active);
  let original = $derived(active?.original);
  let restored = $derived(active?.restored?.bitmap);
  let hasResult = $derived(!!restored);

  let sliderPos = $state(0.5);
  let sliderDragging = $state(false);
  let splitStacked = $state(false);

  let showOriginal = $derived(
    !hasResult || (ui.compareMode === "toggle" && ui.togglingOriginal)
  );

  let panning = $state(false);
  let peekingOriginal = $state(false);
  let dragStarted = false;

  interface ActivePointer {
    downX: number;
    downY: number;
    lastX: number;
    lastY: number;
  }
  const pointers = new Map<number, ActivePointer>();
  let pinchDist = 0;
  let suppressTap = false;
  let lastTap: { x: number; y: number; t: number } | null = null;

  function toLocalX(clientX: number): number {
    const rect = container?.getBoundingClientRect();
    if (!rect) return 0;
    return clientX - rect.left;
  }

  function pointerDistance(): number {
    const pts = [...pointers.values()];
    if (pts.length < 2) return 0;
    return Math.hypot(pts[0]!.lastX - pts[1]!.lastX, pts[0]!.lastY - pts[1]!.lastY);
  }

  function pinchMidpoint(): { x: number; y: number } {
    const pts = [...pointers.values()];
    if (pts.length < 2) return { x: 0, y: 0 };
    const midClientX = (pts[0]!.lastX + pts[1]!.lastX) / 2;
    const midClientY = (pts[0]!.lastY + pts[1]!.lastY) / 2;
    // Anchor in the frame of the half-canvas under the pinch (split mode)
    const el = document.elementFromPoint(midClientX, midClientY);
    if (el instanceof HTMLCanvasElement) {
      const r = el.getBoundingClientRect();
      return { x: midClientX - r.left, y: midClientY - r.top };
    }
    const rect = container?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: midClientX - rect.left, y: midClientY - rect.top };
  }

  function sizeAndFit() {
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const isSplit = ui.compareMode === "split" && hasResult;
    const img = original ?? restored;

    if (isSplit) {
      splitStacked = rect.height > rect.width;
      const halfW = splitStacked ? rect.width : rect.width / 2;
      const halfH = splitStacked ? rect.height / 2 : rect.height;
      for (const c of [canvas, canvasB]) {
        if (!c) continue;
        c.width = halfW * dpr;
        c.height = halfH * dpr;
        c.style.width = splitStacked ? "100%" : "50%";
        c.style.height = splitStacked ? "50%" : "100%";
      }
      viewport.setCanvasSize(halfW, halfH);
    } else {
      if (canvas) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
      }
      viewport.setCanvasSize(rect.width, rect.height);
    }

    if (img) viewport.fit(img.width, img.height);
  }

  function resetCtx(ctx: CanvasRenderingContext2D) {
    ctx.filter = "none";
    ctx.globalAlpha = 1;
  }

  function drawImage(
    c: HTMLCanvasElement,
    img: ImageBitmap | undefined,
  ) {
    if (!c || !img) return;
    const ctx = c.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    const { scale, offsetX, offsetY } = viewport.vp;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    resetCtx(ctx);
    ctx.setTransform(scale * dpr, 0, 0, scale * dpr, offsetX * dpr, offsetY * dpr);
    ctx.drawImage(img, 0, 0);
  }

  function drawSlider() {
    if (!canvas || !original) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    const { scale, offsetX, offsetY } = viewport.vp;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resetCtx(ctx);
    ctx.setTransform(scale * dpr, 0, 0, scale * dpr, offsetX * dpr, offsetY * dpr);

    // Before processing: just show the original, no divider
    if (!hasResult || !restored) {
      ctx.drawImage(original, 0, 0);
      return;
    }

    // After processing: restored as base, original clipped on left
    ctx.drawImage(restored, 0, 0);

    const splitX = sliderPos * original.width;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitX, original.height);
    ctx.clip();
    resetCtx(ctx);
    ctx.drawImage(original, 0, 0);
    ctx.restore();

    // Divider line
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    resetCtx(ctx);
    const screenSplitX = (splitX * scale + offsetX) * dpr;
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 4 * dpr;
    ctx.beginPath();
    ctx.moveTo(screenSplitX, 0);
    ctx.lineTo(screenSplitX, canvas.height);
    ctx.stroke();
  }

  function draw() {
    if (ui.compareMode === "slider") {
      drawSlider();
    } else if (ui.compareMode === "split") {
      if (hasResult && restored) {
        drawImage(canvas, showOriginal ? restored : original);
        drawImage(canvasB, showOriginal ? original : restored);
      } else {
        drawImage(canvas, original);
        drawImage(canvasB, original);
      }
    } else {
      // toggle
      const img = showOriginal ? original : (restored ?? original);
      drawImage(canvas, img);
    }
  }

  $effect(() => {
    void original;
    void restored;
    void ui.compareMode;
    void hasResult;
    sizeAndFit();
  });

  $effect(() => {
    if (!container) return;
    const ro = new ResizeObserver(() => sizeAndFit());
    ro.observe(container);
    return () => ro.disconnect();
  });

  $effect(() => {
    void viewport.vp.scale;
    void viewport.vp.offsetX;
    void viewport.vp.offsetY;
    void original;
    void restored;
    void sliderPos;
    void ui.compareMode;
    void showOriginal;
    draw();
  });

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    const rect = container?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    viewport.smoothZoom(x, y, factor);
  }

  function onPointerDown(e: PointerEvent) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    // Let overlay UI (ErrorOverlay buttons etc.) receive its own clicks
    if (!(e.target instanceof HTMLCanvasElement)) return;
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {
      // No active pointer (synthetic events) — safe to continue without capture
    }
    pointers.set(e.pointerId, { downX: e.clientX, downY: e.clientY, lastX: e.clientX, lastY: e.clientY });

    if (pointers.size === 2) {
      pinchDist = pointerDistance();
      panning = false;
      sliderDragging = false;
      peekingOriginal = false;
      ui.togglingOriginal = false;
      suppressTap = true;
      return;
    }
    if (pointers.size > 2) return;

    if (ui.compareMode === "slider" && hasResult && original) {
      const x = toLocalX(e.clientX);
      const screenSplitX = sliderPos * original.width * viewport.vp.scale + viewport.vp.offsetX;
      const grabRadius = e.pointerType === "touch" ? 24 : 10;
      if (Math.abs(x - screenSplitX) < grabRadius) {
        sliderDragging = true;
        dragStarted = false;
        return;
      }
    }
    if (hasResult) {
      peekingOriginal = true;
      ui.togglingOriginal = true;
    } else {
      panning = true;
    }
    dragStarted = false;
  }

  function onPointerMove(e: PointerEvent) {
    const p = pointers.get(e.pointerId);
    if (!p) return;

    const dx = e.clientX - p.lastX;
    const dy = e.clientY - p.lastY;
    p.lastX = e.clientX;
    p.lastY = e.clientY;

    if (pointers.size >= 2) {
      const dist = pointerDistance();
      if (pinchDist > 0 && dist > 0) {
        const mid = pinchMidpoint();
        viewport.cancelSmooth();
        viewport.zoom(mid.x, mid.y, dist / pinchDist);
      }
      pinchDist = dist;
      return;
    }

    if (sliderDragging) {
      if (original) {
        const x = toLocalX(e.clientX);
        const imgX = (x - viewport.vp.offsetX) / (viewport.vp.scale * original.width);
        sliderPos = Math.min(1, Math.max(0, imgX));
      }
      return;
    }

    if (!peekingOriginal && !panning) return;

    if (peekingOriginal && !dragStarted) {
      const moved = Math.hypot(e.clientX - p.downX, e.clientY - p.downY);
      if (moved > (e.pointerType === "touch" ? 10 : 4)) {
        dragStarted = true;
        peekingOriginal = false;
        ui.togglingOriginal = false;
        panning = true;
      }
    }

    if (panning) viewport.pan(dx, dy);
  }

  function endPointer(e: PointerEvent, canceled: boolean) {
    pointers.delete(e.pointerId);
    pinchDist = 0;

    if (pointers.size === 1) {
      panning = true;
      peekingOriginal = false;
      ui.togglingOriginal = false;
      dragStarted = true;
      sliderDragging = false;
      return;
    }

    const wasDrag = dragStarted || sliderDragging;
    panning = false;
    sliderDragging = false;
    peekingOriginal = false;
    ui.togglingOriginal = false;

    if (!canceled && !wasDrag && !suppressTap && e.pointerType !== "mouse") {
      const now = performance.now();
      const isDoubleTap =
        lastTap !== null &&
        now - lastTap.t < 300 &&
        Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 48;
      if (isDoubleTap) {
        const img = original ?? restored;
        if (img) viewport.fit(img.width, img.height);
        lastTap = null;
      } else {
        lastTap = { x: e.clientX, y: e.clientY, t: now };
      }
    }
    suppressTap = false;
    dragStarted = false;
  }

  function onPointerUp(e: PointerEvent) {
    endPointer(e, false);
  }

  function onPointerCancel(e: PointerEvent) {
    endPointer(e, true);
  }

  function onDoubleClick() {
    const img = original ?? restored;
    if (img) viewport.fit(img.width, img.height);
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.code === "Space" && ui.compareMode === "toggle") {
      e.preventDefault();
      ui.togglingOriginal = true;
    }
  }
  function onKeyUp(e: KeyboardEvent) {
    if (e.code === "Space") {
      ui.togglingOriginal = false;
    }
  }

  let dragOver = $state(false);

  function onDragOver(e: DragEvent) {
    e.preventDefault();
    dragOver = true;
  }

  function onDragLeave() {
    dragOver = false;
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    dragOver = false;
    const files = Array.from(e.dataTransfer?.files ?? []).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (files.length > 0) queue.enqueue(files);
  }
</script>

<svelte:window onkeydown={onKeyDown} onkeyup={onKeyUp} />

<div
  class="stage"
  role="region"
  aria-label="Image canvas — drop images here to add them to the queue"
  bind:this={container}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  ondrop={onDrop}
  onwheel={onWheel}
  onpointerdown={onPointerDown}
  onpointermove={onPointerMove}
  onpointerup={onPointerUp}
  onpointercancel={onPointerCancel}
  class:drag-over={dragOver}
>
  <ProgressTopBar />
  <ErrorOverlay />
  {#if ui.compareMode === "split" && hasResult}
    <div class="split-container" class:stacked={splitStacked}>
      <canvas bind:this={canvas} ondblclick={onDoubleClick}></canvas>
      <canvas bind:this={canvasB} ondblclick={onDoubleClick}></canvas>
    </div>
  {:else}
    <canvas
      bind:this={canvas}
      ondblclick={onDoubleClick}
      style="cursor:{sliderDragging ? 'ew-resize' : panning ? 'grabbing' : 'grab'}"
    ></canvas>
  {/if}
</div>

<style>
  .stage { position: absolute; inset: 0; overflow: hidden; touch-action: none; }
  .stage.drag-over::after {
    content: "";
    position: absolute;
    inset: 8px;
    border: 3px dashed var(--accent);
    border-radius: var(--radius);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
    pointer-events: none;
    z-index: 30;
  }
  canvas { position: absolute; inset: 0; }
  .split-container { position: absolute; inset: 0; display: flex; }
  .split-container.stacked { flex-direction: column; }
  .split-container canvas { position: relative; }
</style>
