<script lang="ts">
  import { tick } from "svelte";
  import { queue } from "../lib/state/queue.svelte";
  import { theme } from "../lib/state/theme.svelte";
  import { ui } from "../lib/state/ui.svelte";
  import { inference, snapshot, download, fetchWithProgress } from "../lib/state/inference.svelte";
  import { engine, persistedModelId } from "../lib/state/engine.svelte";
  import { toasts } from "../lib/state/ui.svelte";
  import {
    MODELS,
    Cancelled,
    ClientBusy,
    describeError,
    errorSummary,
  } from "inference-core";
  import CanvasStage from "../components/CanvasStage.svelte";
  import FloatingControls from "../components/FloatingControls.svelte";
  import SettingsPanel from "../components/SettingsPanel.svelte";
  import QueueBar from "../components/QueueBar.svelte";
  import Dropzone from "../components/Dropzone.svelte";
  import Toasts from "../components/Toasts.svelte";
  import { setupKeyboardShortcuts } from "../lib/keyboard";
  import { viewport } from "../lib/canvas/viewport.svelte";

  let activeController: AbortController | null = $state(null);

  // Reactive so a model picked during a load is loaded right after it.
  let modelLoading = $state(false);
  let failedModelId: string | null = null;
  // Settles when the current load is over; processing waits for it.
  let modelLoad: Promise<void> = Promise.resolve();

  // Load model when persistedModelId changes. Pre-fetches the model file with
  // progress tracking, then lets the worker load from browser cache.
  $effect(() => {
    const modelId = persistedModelId.value;
    if (!modelId) {
      persistedModelId.value = MODELS[0]!.id;
      return;
    }
    if (modelLoading || snapshot.state === "processing") return;
    if (snapshot.modelId === modelId && snapshot.state === "ready") return;
    if (failedModelId === modelId) return;
    const modelDef = MODELS.find((m) => m.id === modelId);
    if (!modelDef) {
      persistedModelId.value = MODELS[0]!.id;
      return;
    }
    modelLoading = true;
    modelLoad = fetchWithProgress(modelDef.url, `Downloading ${modelDef.name}`)
      .then(() => inference.loadModel(modelId))
      .then(() => { failedModelId = null; })
      .catch((e) => {
        // The client can't switch models while busy. This effect runs again once it is free.
        if (e instanceof ClientBusy) return;
        failedModelId = modelId;
        console.error("model load failed", e);
        const message = describeError(e);
        toasts.push(`Model failed to load: ${message}`, "error");
        engine.errorMessage = message;
      })
      .finally(() => { modelLoading = false; });
  });

  // Keyboard shortcuts: 0=fit, 1=actual, +/-=zoom, space=toggle original,
  // arrows=navigate, enter=process, del=remove, B/A=settings
  $effect(() => {
    return setupKeyboardShortcuts({
      onFit: () => {
        const img = queue.active?.original ?? queue.active?.restored?.bitmap;
        if (img && viewport) viewport.fit(img.width, img.height);
      },
      onActualSize: () => {
        if (viewport) viewport.vp.scale = 1;
      },
      onZoomIn: () => {
        if (!viewport) return;
        const rect = document.querySelector(".stage")?.getBoundingClientRect();
        if (rect) viewport.smoothZoom(rect.width / 2, rect.height / 2, 1.5);
      },
      onZoomOut: () => {
        if (!viewport) return;
        const rect = document.querySelector(".stage")?.getBoundingClientRect();
        if (rect) viewport.smoothZoom(rect.width / 2, rect.height / 2, 1 / 1.5);
      },
      onToggleOriginal: (down) => { ui.togglingOriginal = down; },
      onPrevItem: () => {
        const idx = queue.items.findIndex((i) => i.id === queue.activeId);
        if (idx > 0) queue.select(queue.items[idx - 1].id);
      },
      onNextItem: () => {
        const idx = queue.items.findIndex((i) => i.id === queue.activeId);
        if (idx >= 0 && idx < queue.items.length - 1) queue.select(queue.items[idx + 1].id);
      },
      onProcessSelected: () => {
        if (queue.selectedIds.size > 1) void processSelected();
        else if (queue.activeId) processOne(queue.activeId);
      },
      onRemoveItem: () => {
        if (queue.activeId) queue.remove(queue.activeId);
      },
      onToggleSettingsBasic: () => {
        ui.settingsTab = "basic";
        ui.settingsCollapsed = !ui.settingsCollapsed;
      },
      onToggleSettingsAdvanced: () => {
        ui.settingsTab = "advanced";
        ui.settingsCollapsed = !ui.settingsCollapsed;
      },
    });
  });

  // Decode the original bitmap for display when a pending item is selected,
  // Decode original bitmap for display whenever the active item lacks one.
  $effect(() => {
    const item = queue.active;
    if (!item || item.original) return;
    createImageBitmap(item.file)
      .then((bitmap) => {
        if (queue.activeId === item.id) {
          queue.update(item.id, { original: bitmap });
        }
      })
      .catch((e) => console.error("decode original for display failed", e));
  });

  // Process a single item
  async function processItem(itemId: string) {
    const item = queue.items.find((i) => i.id === itemId);
    if (!item) return;

    // A model picked a moment ago may have a load starting or under way.
    await tick();
    await modelLoad;

    const modelDef = MODELS.find((m) => m.id === snapshot.modelId);
    if (!modelDef) {
      toasts.push("Select a model first.", "error");
      return;
    }

    queue.markActive(itemId);
    const controller = new AbortController();
    activeController = controller;

    let original: ImageBitmap | undefined;
    try {
      original = await createImageBitmap(item.file);
    } catch (e) {
      console.error("decode original failed", e);
    }

    try {
      toasts.clear();
      const result = await inference.process({
        file: item.file,
        params: engine.getParams(modelDef.id),
        signal: controller.signal,
        tileSizeOverride: engine.tileOverride ?? undefined,
        tileBatch: engine.batchOverride ?? undefined,
        onProgress: (ev) => {
          if (ev.kind === "image" && ev.stage === "tile" && ev.done !== undefined && ev.total !== undefined) {
            queue.update(itemId, { phase: "inferring", tileProgress: { done: ev.done, total: ev.total } });
          } else if (ev.kind === "image") {
            queue.update(itemId, {
              phase: ev.stage === "decode" ? "decoding" : ev.stage === "blend" || ev.stage === "finalize" ? "blending" : "inferring",
            });
          }
        },
      });
      queue.markDone(itemId, { bitmap: result.bitmap, blob: result.blob, width: result.width, height: result.height });
      if (result.qfPredicted !== undefined) queue.update(itemId, { qfPredicted: result.qfPredicted });
      queue.update(itemId, { elapsedMs: result.elapsedMs });
      if (original) queue.update(itemId, { original });
    } catch (e) {
      if (e instanceof Cancelled) {
        queue.markCancelled(itemId);
        toasts.push("Cancelled.", "info");
        return;
      }
      console.error("processing failed", e);
      const userMessage = describeError(e);
      toasts.push(userMessage, "error");
      engine.errorMessage = userMessage;
      queue.markFailed(itemId, errorSummary(e), userMessage);
    } finally {
      if (activeController === controller) {
        activeController = null;
      }
    }
  }

  // Exposed for QueueBar to call via props
  function processOne(id: string) { void processItem(id); }
  let cancelRequested = false;

  async function processSelected() {
    cancelRequested = false;
    for (const item of queue.getSelected()) {
      if (cancelRequested) break;
      if (item.phase === "queued" || item.phase === "done" || item.phase === "failed") await processItem(item.id);
    }
  }
  async function processAll() {
    cancelRequested = false;
    for (const item of queue.getPending()) {
      if (cancelRequested) break;
      await processItem(item.id);
    }
  }
  function cancelActive() {
    cancelRequested = true;
    activeController?.abort();
  }

  function onDrop(files: File[]) {
    queue.enqueue(files);
  }
</script>

<div class="editor">
  <div class="editor-canvas">
    {#if queue.items.length === 0}
      <Dropzone handleFiles={onDrop} />
    {:else}
      <CanvasStage />
    {/if}
    <FloatingControls />
    <SettingsPanel />
  </div>
  <QueueBar onProcess={processOne} onProcessSelected={processSelected} onProcessAll={processAll} onCancel={cancelActive} />
</div>

<Toasts />

<style>
  .editor {
    display: flex;
    flex-direction: column;
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
  }
  .editor-canvas {
    flex: 1;
    position: relative;
    overflow: hidden;
    background: var(--bg-canvas);
  }
</style>
