<script lang="ts">
  import { theme } from "../lib/state/theme.svelte";
  import { ui, type CompareMode } from "../lib/state/ui.svelte";
  import { queue } from "../lib/state/queue.svelte";

  const modes: { value: CompareMode; label: string; icon: string }[] = [
    { value: "slider", label: "Slider", icon: "⇆" },
    { value: "split", label: "Split", icon: "⊟" },
    { value: "toggle", label: "Toggle", icon: "◐" },
  ];

  let infoOpen = $state(false);
  let active = $derived(queue.active);

  function formatMs(ms: number): string {
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(1)} s`;
  }

  function formatDim(w: number, h: number): string {
    return `${w} × ${h}`;
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
</script>

<div class="floating-controls">
  <div class="info-wrapper">
    <button
      class="icon-btn"
      class:active={infoOpen}
      onclick={(e) => { e.stopPropagation(); infoOpen = !infoOpen; }}
      title="Details"
      aria-label="Details"
    >
      ⓘ
    </button>
    {#if infoOpen}
      <div class="info-popout" onclick={(e) => e.stopPropagation()}>
        {#if active}
          <div class="info-row">
            <span class="info-label">File</span>
            <span class="info-value">{active.file.name}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Size</span>
            <span class="info-value">{formatBytes(active.file.size)}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Type</span>
            <span class="info-value">{active.file.type || "—"}</span>
          </div>
          {#if active.restored}
            <div class="info-row">
              <span class="info-label">Dimensions</span>
              <span class="info-value">{formatDim(active.restored.width, active.restored.height)}</span>
            </div>
          {/if}
          {#if active.qfPredicted !== undefined}
            <div class="info-row">
              <span class="info-label">Predicted QF</span>
              <span class="info-value">{active.qfPredicted}</span>
            </div>
          {/if}
        {/if}
        <div class="info-row">
          <span class="info-label">Processing time</span>
          <span class="info-value">{active?.elapsedMs !== undefined ? formatMs(active.elapsedMs) : "—"}</span>
        </div>
      </div>
    {/if}
  </div>

  <button
    class="icon-btn"
    onclick={() => theme.toggle()}
    title="Toggle theme"
    aria-label="Toggle theme"
  >
    {theme.current === "dark" ? "☀" : "☾"}
  </button>

  <div class="view-switch">
    {#each modes as m}
      <button
        class="view-btn"
        class:active={ui.compareMode === m.value}
        onclick={() => (ui.compareMode = m.value)}
        title={m.label}
      >
        {m.icon}
      </button>
    {/each}
  </div>
</div>

<svelte:window onclick={() => (infoOpen = false)} />

<style>
  .floating-controls {
    position: absolute;
    top: 12px;
    right: 12px;
    z-index: 20;
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .info-wrapper {
    position: relative;
  }
  .icon-btn {
    width: 36px;
    height: 36px;
    border-radius: var(--radius);
    background: var(--bg-panel);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    color: var(--text-primary);
    transition: background 0.15s;
  }
  .icon-btn:hover, .icon-btn.active {
    background: var(--bg-panel-hover);
  }
  .info-popout {
    position: absolute;
    top: 44px;
    right: 0;
    min-width: 200px;
    background: var(--bg-panel);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    z-index: 30;
  }
  .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  .info-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .info-value {
    font-size: 0.8125rem;
    color: var(--text-primary);
    text-align: right;
  }
  .view-switch {
    display: flex;
    background: var(--bg-panel);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .view-btn {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    color: var(--text-secondary);
    transition: background 0.15s, color 0.15s;
  }
  .view-btn:hover {
    background: var(--bg-panel-hover);
  }
  .view-btn.active {
    background: var(--accent);
    color: var(--text-on-accent);
  }
</style>
