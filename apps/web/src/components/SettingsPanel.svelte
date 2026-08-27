<script lang="ts">
  import { onMount } from "svelte";
  import { ui } from "../lib/state/ui.svelte";
  import { engine, persistedModelId } from "../lib/state/engine.svelte";
  import { snapshot, download, inference } from "../lib/state/inference.svelte";
  import { queue } from "../lib/state/queue.svelte";
  import { MODELS, type ModelParam, type EncodeFormat } from "inference-core";

  let mounted = $state(false);

  onMount(() => {
    if (window.matchMedia("(pointer: coarse)").matches || window.innerWidth <= 768) {
      ui.settingsCollapsed = true;
    }
    mounted = true;
  });

  let def = $derived(MODELS.find((m) => m.id === snapshot.modelId));
  let params = $derived(def ? engine.getParams(def.id) : {});
  let qfAuto = $state(true);

  // Advanced state
  let advancedOpen = $state(false);

  function selectModel(id: string) {
    persistedModelId.value = id;
  }

  // Export state
  let exportFormat = $state<EncodeFormat>("png");
  let exportQuality = $state(92);
  let exporting = $state(false);
  const formats: { value: EncodeFormat; label: string }[] = [
    { value: "png", label: "PNG (lossless)" },
    { value: "jpeg", label: "JPEG" },
    { value: "webp", label: "WebP" },
    { value: "avif", label: "AVIF" },
  ];

  let doneCount = $derived(queue.items.filter((i) => i.phase === "done").length);
  let selectedDone = $derived(queue.getSelected().filter((i) => i.phase === "done"));
  let activeQfPredicted = $derived(queue.active?.qfPredicted);

  async function doExport() {
    const items = selectedDone.length > 0 ? selectedDone : queue.items.filter((i) => i.phase === "done" && i.id === queue.activeId);
    if (items.length === 0) return;
    exporting = true;
    try {
      for (const item of items) {
        if (!item.restored) continue;
        const blob = await inference.encode(item.restored.bitmap, { format: exportFormat, quality: exportQuality });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${item.file.name.replace(/\.[^.]+$/, "")}-dejpeg.${exportFormat}`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (e) {
      console.error("export failed", e);
    } finally {
      exporting = false;
    }
  }

  // Progress state
  let downloading = $derived(download.active);
  let modelLoading = $derived(snapshot.state === "loading");
  let processing = $derived(
    queue.items.find((i) => i.phase === "decoding" || i.phase === "inferring" || i.phase === "blending")
  );

  let batchTotal = $derived.by(() => {
    return queue.items.filter((i) =>
      i.phase === "queued" || i.phase === "decoding" || i.phase === "inferring" ||
      i.phase === "blending" || i.phase === "done" || i.phase === "failed"
    ).length;
  });
  let batchDone = $derived(queue.items.filter((i) => i.phase === "done").length);

  let statusLabel = $derived.by(() => {
    if (downloading) return download.label;
    if (modelLoading) return "Initializing runtime…";
    if (processing) {
      const tile = processing.tileProgress ? ` — ${processing.tileProgress.done}/${processing.tileProgress.total} tiles` : " — decoding…";
      if (batchTotal > 1) return `Image ${batchDone + 1}/${batchTotal}${tile}`;
      return `Processing${tile}`;
    }
    if (snapshot.state === "ready") return "Ready";
    if (snapshot.state === "error") return "Error";
    return "Idle";
  });

  let progressPct = $derived.by(() => {
    if (downloading && download.total > 0) return Math.round((download.loaded / download.total) * 100);
    if (processing && batchTotal > 0) {
      let frac = batchDone;
      if (processing.tileProgress) frac += processing.tileProgress.done / processing.tileProgress.total;
      return Math.round((frac / batchTotal) * 100);
    }
    if (downloading || modelLoading || processing) return -1;
    return 100;
  });

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
</script>

{#if mounted}
  {#if ui.settingsCollapsed}
  <button
    class="settings-fab"
    onclick={() => (ui.settingsCollapsed = false)}
    title="Settings"
    aria-label="Open settings"
  >
    ⚙
  </button>
{:else}
<div class="settings-panel">
  <div class="header">
    <div class="tabs">
      <button class="tab" class:active={ui.settingsTab !== "export"} onclick={() => (ui.settingsTab = "processing")}>Processing</button>
      <button class="tab" class:active={ui.settingsTab === "export"} onclick={() => (ui.settingsTab = "export")}>Export</button>
    </div>
    <button
      class="close-btn"
      onclick={() => (ui.settingsCollapsed = true)}
      title="Collapse settings"
      aria-label="Collapse settings"
    >
      ✕
    </button>
  </div>

  {#if ui.settingsTab === "export"}
    <div class="body">
      <label class="field">
        <span class="field-label">Format</span>
        <select bind:value={exportFormat}>
          {#each formats as f}
            <option value={f.value}>{f.label}</option>
          {/each}
        </select>
      </label>
      {#if exportFormat !== "png"}
        <label class="field">
          <span class="field-label">Quality</span>
          <div class="range-row">
            <input type="range" min="10" max="100" bind:value={exportQuality} />
            <span class="value">{exportQuality}</span>
          </div>
        </label>
      {/if}
      <button class="export-btn" disabled={doneCount === 0 || exporting} onclick={doExport}>
        {#if exporting}
          Exporting…
        {:else if selectedDone.length > 1}
          Export {selectedDone.length} selected
        {:else}
          Export{doneCount > 1 ? ` (${doneCount} available)` : ""}
        {/if}
      </button>
      {#if doneCount === 0}
        <p class="hint">Process an image first, then export the result.</p>
      {/if}
    </div>
  {:else}
    <div class="body">
      <label class="field">
        <span class="field-label">Model</span>
        <select value={snapshot.modelId ?? ""} onchange={(e) => selectModel((e.currentTarget as HTMLSelectElement).value)}>
          {#if !snapshot.modelId}
            <option value="" disabled>Select a model…</option>
          {/if}
          {#each MODELS as m (m.id)}
            <option value={m.id}>{m.name}</option>
          {/each}
        </select>
      </label>

      {#if def && Object.keys(def.params).length > 0}
        {#each Object.entries(def.params) as [name, p] (name)}
          {@const mp = p as ModelParam}
          {#if name === "qf" && mp.kind === "range"}
            <div class="field">
              <span class="field-label">Quality Factor</span>
              <div class="radio-row">
                <label class="radio">
                  <input type="radio" name="qf-mode" value="auto" checked={qfAuto} onchange={() => { qfAuto = true; if (def) engine.setParam(def.id, name, mp.default); }} />
                  <span>Auto-detect</span>
                </label>
                <label class="radio">
                  <input type="radio" name="qf-mode" value="manual" checked={!qfAuto} onchange={() => { qfAuto = false; }} />
                  <span>Manual</span>
                </label>
              </div>
              {#if !qfAuto}
                <div class="range-row">
                  <input type="range" min={mp.min} max={mp.max} step={mp.step ?? 1} value={params[name] ?? mp.default}
                    oninput={(e) => def && engine.setParam(def.id, name, Number((e.currentTarget as HTMLInputElement).value))} />
                  <span class="value">{params[name] ?? mp.default}</span>
                </div>
              {/if}
            </div>
          {:else if mp.kind === "range"}
            <label class="field">
              <span class="field-label">{mp.label}</span>
              <div class="range-row">
                <input type="range" min={mp.min} max={mp.max} step={mp.step ?? 1} value={params[name] ?? mp.default}
                  oninput={(e) => def && engine.setParam(def.id, name, Number((e.currentTarget as HTMLInputElement).value))} />
                <span class="value">{params[name] ?? mp.default}</span>
              </div>
            </label>
          {:else if mp.kind === "select"}
            <label class="field">
              <span class="field-label">{mp.label}</span>
              <select value={params[name] ?? mp.default} onchange={(e) => def && engine.setParam(def.id, name, (e.currentTarget as HTMLSelectElement).value)}>
                {#each mp.options as opt (opt)}<option value={opt}>{opt}</option>{/each}
              </select>
            </label>
          {:else if mp.kind === "toggle"}
            <label class="field">
              <span class="field-label">{mp.label}</span>
              <input type="checkbox" checked={Boolean(params[name] ?? mp.default)}
                onchange={(e) => def && engine.setParam(def.id, name, (e.currentTarget as HTMLInputElement).checked)} />
            </label>
          {/if}
        {/each}
      {/if}

      {#if activeQfPredicted !== undefined}
        <div class="qf-predicted">
          <span class="field-label">Predicted QF</span>
          <span class="qf-value">{activeQfPredicted}</span>
        </div>
      {/if}

      <button class="accordion-toggle" onclick={() => (advancedOpen = !advancedOpen)}>
        <span class="chevron">{advancedOpen ? "▾" : "▸"}</span>
        Advanced
      </button>
      {#if advancedOpen}
        <div class="advanced-body">
          <div class="field">
            <span class="field-label">Tile Size</span>
            <div class="range-row">
              <input type="range" min="64" max="1024" step="8"
                value={engine.tileOverride ?? snapshot.tileSize ?? 512}
                oninput={(e) => { engine.tileOverride = Number((e.currentTarget as HTMLInputElement).value); }} />
              <span class="value">{engine.tileOverride ?? snapshot.tileSize ?? 512}</span>
            </div>
            {#if engine.tileOverride !== null}
              <button class="reset-btn" onclick={() => { engine.tileOverride = null; }}>Reset to default</button>
            {/if}
          </div>
          <div class="field">
            <span class="field-label">Batch Size</span>
            <div class="range-row">
              <input type="range" min="1" max="8" step="1"
                value={engine.batchOverride ?? 1}
                oninput={(e) => { engine.batchOverride = Number((e.currentTarget as HTMLInputElement).value); }} />
              <span class="value">{engine.batchOverride ?? 1}</span>
            </div>
            {#if engine.batchOverride !== null && engine.batchOverride !== 1}
              <button class="reset-btn" onclick={() => { engine.batchOverride = null; }}>Reset to default</button>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <div class="progress-section">
    <div class="status-row">
      <span class="status-label">{statusLabel}</span>
      {#if downloading && download.total > 0}
        <span class="status-detail">{formatBytes(download.loaded)} / {formatBytes(download.total)}</span>
      {/if}
    </div>
    <div class="progress-track">
      {#if progressPct >= 0}
        <div class="progress-fill" style="width:{progressPct}%"></div>
      {:else}
        <div class="progress-indeterminate"></div>
      {/if}
    </div>
  </div>
</div>
{/if}
{/if}

<style>
  .settings-panel {
    position: absolute; bottom: calc(12px + env(safe-area-inset-bottom)); right: 12px; width: min(280px, calc(100vw - 24px)); z-index: 20;
    background: var(--bg-panel); backdrop-filter: blur(12px); box-shadow: var(--shadow);
    border-radius: var(--radius); border: 1px solid var(--border); overflow: hidden;
    display: flex; flex-direction: column;
  }
  .settings-fab {
    position: absolute; bottom: calc(12px + env(safe-area-inset-bottom)); right: 12px; z-index: 20;
    width: 44px; height: 44px;
    border-radius: var(--radius);
    background: var(--bg-panel); backdrop-filter: blur(12px); box-shadow: var(--shadow);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; color: var(--text-primary);
    transition: background 0.15s;
  }
  .settings-fab:hover { background: var(--bg-panel-hover); }
  .close-btn {
    width: 24px; height: 24px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; color: var(--text-secondary);
    border-radius: calc(var(--radius) - 2px);
  }
  .close-btn:hover { color: var(--text-primary); background: var(--bg-panel-hover); }
  .header { display: flex; align-items: center; justify-content: space-between; padding: 4px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .tabs { display: flex; gap: 2px; }
  .tab { padding: 6px 12px; border-radius: calc(var(--radius) - 2px); font-size: 0.8125rem; color: var(--text-secondary); transition: background 0.15s, color 0.15s; }
  .tab.active { background: var(--accent); color: var(--text-on-accent); }
  .body { padding: 12px; display: flex; flex-direction: column; gap: 12px; max-height: 50vh; overflow-y: auto; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.04em; }
  select { width: 100%; padding: 6px 8px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--bg-panel-hover); color: var(--text-primary); font-size: 0.8125rem; }
  select:focus { outline: 2px solid var(--accent); }
  .range-row { display: flex; align-items: center; gap: 8px; }
  .range-row input[type="range"] { flex: 1; }
  .radio-row { display: flex; gap: 12px; }
  .radio { display: flex; align-items: center; gap: 4px; font-size: 0.8125rem; color: var(--text-primary); cursor: pointer; }
  .value { font-size: 0.8125rem; min-width: 28px; text-align: right; color: var(--text-primary); }
  .hint { font-size: 0.75rem; color: var(--text-secondary); margin: 0; }
  .export-btn { padding: 8px 14px; border-radius: var(--radius); background: var(--accent); color: var(--text-on-accent); font-size: 0.8125rem; font-weight: 500; transition: background 0.15s, opacity 0.15s; }
  .export-btn:hover:not(:disabled) { background: var(--accent-hover); }
  .export-btn:disabled { opacity: 0.4; cursor: default; }
  .qf-predicted { display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: var(--bg-panel-hover); border-radius: var(--radius); }
  .qf-value { font-size: 0.875rem; font-weight: 600; color: var(--accent); }
  .accordion-toggle { display: flex; align-items: center; gap: 6px; padding: 6px 0; font-size: 0.8125rem; color: var(--text-secondary); font-weight: 500; }
  .accordion-toggle:hover { color: var(--text-primary); }
  .chevron { font-size: 0.625rem; }
  .advanced-body { display: flex; flex-direction: column; gap: 12px; padding-left: 8px; border-left: 2px solid var(--border); }
  .reset-btn { align-self: flex-start; font-size: 0.6875rem; color: var(--accent); padding: 2px 6px; }
  .reset-btn:hover { text-decoration: underline; }
  .progress-section { padding: 8px 12px; border-top: 1px solid var(--border); flex-shrink: 0; display: flex; flex-direction: column; gap: 4px; }
  .status-row { display: flex; justify-content: space-between; align-items: center; }
  .status-label { font-size: 0.75rem; color: var(--text-primary); font-weight: 500; }
  .status-detail { font-size: 0.6875rem; color: var(--text-secondary); }
  .progress-track { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; position: relative; }
  .progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.2s ease; }
  .progress-indeterminate { height: 100%; width: 40%; background: var(--accent); border-radius: 2px; position: absolute; animation: indeterminate 1.2s ease-in-out infinite; }
  @keyframes indeterminate { 0% { left: -40%; } 100% { left: 100%; } }
</style>
