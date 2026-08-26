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

  let showOriginal = $derived(
    !hasResult || (ui.compareMode === "toggle" && ui.togglingOriginal)
  );

  let panning = $state(false);
  let peekingOriginal = $state(false);
  let dragStarted = false;
  let lastX = 0;
  let lastY = 0;

  function sizeAndFit() {
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const isSplit = ui.compareMode === "split" && hasResult;
    const img = original ?? restored;

    if (isSplit) {
      const halfW = rect.width / 2;
      for (const c of [canvas, canvasB]) {
        if (!c) continue;
        c.width = halfW * dpr;
        c.height = rect.height * dpr;
        c.style.width = "50%";
        c.style.height = `${rect.height}px`;
      }
      viewport.setCanvasSize(halfW, rect.height);
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
    const target = e.currentTarget as HTMLCanvasElement;
    if (!target) return;
    const rect = target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    viewport.smoothZoom(x, y, factor);
  }

  function onMouseDown(e: MouseEvent) {
    if (e.button !== 0) return;
    if (ui.compareMode === "slider" && hasResult && original) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const screenSplitX = sliderPos * original.width * viewport.vp.scale + viewport.vp.offsetX;
      if (Math.abs(x - screenSplitX) < 10) {
        sliderDragging = true;
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
    lastX = e.clientX;
    lastY = e.clientY;
  }

  function onMouseMove(e: MouseEvent) {
    if (sliderDragging && canvas && original) {
      const rect = canvas.getBoundingClientRect();
      const imgX = (e.clientX - rect.left - viewport.vp.offsetX) / (viewport.vp.scale * original.width);
      sliderPos = Math.min(1, Math.max(0, imgX));
      return;
    }
    if (!peekingOriginal && !panning) return;

    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;

    if (peekingOriginal && !dragStarted) {
      if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
        dragStarted = true;
        peekingOriginal = false;
        ui.togglingOriginal = false;
        panning = true;
      }
    }

    if (panning) {
      viewport.pan(dx, dy);
      lastX = e.clientX;
      lastY = e.clientY;
    }
  }

  function onMouseUp() {
    panning = false;
    sliderDragging = false;
    peekingOriginal = false;
    ui.togglingOriginal = false;
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

<svelte:window onmouseup={onMouseUp} onkeydown={onKeyDown} onkeyup={onKeyUp} />

<div class="stage" role="region" aria-label="Image canvas — drop images here to add them to the queue" bind:this={container} ondragover={onDragOver} ondragleave={onDragLeave} ondrop={onDrop} class:drag-over={dragOver}>
  <ProgressTopBar />
  <ErrorOverlay />
  {#if ui.compareMode === "split" && hasResult}
    <div class="split-container">
      <canvas bind:this={canvas} onwheel={onWheel} onmousedown={onMouseDown} onmousemove={onMouseMove} ondblclick={onDoubleClick}></canvas>
      <canvas bind:this={canvasB} onwheel={onWheel} onmousedown={onMouseDown} onmousemove={onMouseMove} ondblclick={onDoubleClick}></canvas>
    </div>
  {:else}
    <canvas
      bind:this={canvas}
      onwheel={onWheel}
      onmousedown={onMouseDown}
      onmousemove={onMouseMove}
      ondblclick={onDoubleClick}
      style="cursor:{sliderDragging ? 'ew-resize' : panning ? 'grabbing' : 'grab'}"
    ></canvas>
  {/if}
</div>

<style>
  .stage { position: absolute; inset: 0; overflow: hidden; }
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
  .split-container canvas { position: relative; flex: 1; width: 50%; }
</style>
