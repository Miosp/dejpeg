<script lang="ts">
  import { queue, type QueueItem } from "../lib/state/queue.svelte";

  let {
    onProcess,
    onProcessSelected,
    onProcessAll,
    onCancel,
  }: {
    onProcess: (id: string) => void;
    onProcessSelected: () => void;
    onProcessAll: () => void;
    onCancel: () => void;
  } = $props();

  let active = $derived(queue.active);
  let selectedCount = $derived(queue.selectedIds.size);
  let pendingCount = $derived(queue.getPending().length);
  let isProcessing = $derived(
    !!queue.items.find((i) =>
      i.phase === "decoding" || i.phase === "inferring" || i.phase === "blending"
    )
  );

  function fileInput(): HTMLInputElement {
    return document.querySelector("#queue-file-input") as HTMLInputElement;
  }

  function onFileChange(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    if (input.files) {
      queue.enqueue(Array.from(input.files));
      input.value = "";
    }
  }

  function onItemClick(e: MouseEvent, item: QueueItem) {
    if (e.ctrlKey || e.metaKey) {
      queue.toggleSelect(item.id);
    } else {
      queue.select(item.id);
    }
  }

  function badgeColor(phase: QueueItem["phase"]): string {
    switch (phase) {
      case "queued":
        return "var(--badge-pending)";
      case "decoding":
      case "inferring":
      case "blending":
        return "var(--badge-processing)";
      case "done":
        return "var(--badge-done)";
      case "failed":
        return "var(--badge-error)";
      case "cancelled":
        return "var(--badge-pending)";
    }
  }

  function badgeText(item: QueueItem): string {
    switch (item.phase) {
      case "queued":
        return "⏳";
      case "decoding":
        return "◐";
      case "inferring":
        if (item.tileProgress)
          return `${Math.round((item.tileProgress.done / item.tileProgress.total) * 100)}%`;
        return "◐";
      case "blending":
        return "◐";
      case "done":
        return "✓";
      case "failed":
        return "!";
      case "cancelled":
        return "⊘";
    }
  }

  let contextMenu = $state<{ x: number; y: number; itemId: string } | null>(null);

  function onContextMenu(e: MouseEvent, item: QueueItem) {
    e.preventDefault();
    contextMenu = { x: e.clientX, y: e.clientY, itemId: item.id };
  }

  function closeContextMenu() {
    contextMenu = null;
  }

  function ctxRemove() {
    if (contextMenu) queue.remove(contextMenu.itemId);
    closeContextMenu();
  }

  function ctxReprocess() {
    if (contextMenu) {
      queue.update(contextMenu.itemId, { phase: "queued", errorMessage: undefined, errorTag: undefined });
    }
    closeContextMenu();
  }

  function ctxDownload() {
    if (!contextMenu) return;
    const item = queue.items.find((i) => i.id === contextMenu.itemId);
    if (item?.restored) {
      const url = URL.createObjectURL(item.restored.blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${item.file.name.replace(/\.[^.]+$/, "")}-dejpeg.png`;
      a.click();
      URL.revokeObjectURL(url);
    }
    closeContextMenu();
  }

  let downloading = $state(false);

  async function downloadBatch() {
    const items = queue.getSelected().filter((i) => i.restored);
    if (items.length === 0) return;
    downloading = true;
    try {
      for (const item of items) {
        if (!item.restored) continue;
        const url = URL.createObjectURL(item.restored.blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${item.file.name.replace(/\.[^.]+$/, "")}-dejpeg.png`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } finally {
      downloading = false;
    }
  }

  let doneSelected = $derived(
    queue.getSelected().filter((i) => i.phase === "done").length,
  );

  function ctxDuplicate() {
    if (!contextMenu) return;
    const item = queue.items.find((i) => i.id === contextMenu.itemId);
    if (item) {
      const idx = queue.items.indexOf(item);
      const newId = crypto.randomUUID();
      const newItem: QueueItem = {
        ...item,
        id: newId,
        phase: "queued",
        restored: undefined,
        original: undefined,
        tileProgress: undefined,
        errorMessage: undefined,
        errorTag: undefined,
        finishedAt: undefined,
        thumbnailUrl: URL.createObjectURL(item.file),
      };
      const items = [...queue.items];
      items.splice(idx + 1, 0, newItem);
      queue.items = items;
    }
    closeContextMenu();
  }

  // Drag reorder
  let dragId = $state<string | null>(null);

  function onDragStart(e: DragEvent, item: QueueItem) {
    dragId = item.id;
    e.dataTransfer?.setData("text/plain", item.id);
  }

  function onDragOver(e: DragEvent, item: QueueItem) {
    e.preventDefault();
    if (!dragId || dragId === item.id) return;
    const items = [...queue.items];
    const fromIdx = items.findIndex((i) => i.id === dragId);
    const toIdx = items.findIndex((i) => i.id === item.id);
    if (fromIdx === -1 || toIdx === -1) return;
    const [moved] = items.splice(fromIdx, 1);
    items.splice(toIdx, 0, moved);
    queue.items = items;
  }

  function onDragEnd() {
    dragId = null;
  }
</script>

<div class="queue-bar">
  <input
    id="queue-file-input"
    type="file"
    accept="image/*"
    multiple
    onchange={onFileChange}
    hidden
  />

  <div class="thumbs">
    {#each queue.items as item (item.id)}
      <button
        class="thumb"
        class:active={queue.activeId === item.id}
        class:selected={queue.selectedIds.has(item.id)}
        class:dragging={dragId === item.id}
        onclick={(e) => onItemClick(e, item)}
        oncontextmenu={(e) => onContextMenu(e, item)}
        draggable="true"
        ondragstart={(e) => onDragStart(e, item)}
        ondragover={(e) => onDragOver(e, item)}
        ondragend={onDragEnd}
        title={item.file.name}
      >
        {#if item.thumbnailUrl}
          <img src={item.thumbnailUrl} alt={item.file.name} />
        {/if}
        <span class="badge" style="background:{badgeColor(item.phase)}">{badgeText(item)}</span>
      </button>
    {/each}
    <button class="thumb add" onclick={() => fileInput().click()} title="Add images"> + </button>
  </div>

  <div class="actions">
    {#if selectedCount > 1}
      <button class="action-btn" disabled={pendingCount === 0} onclick={onProcessSelected}>
        Process Selected ({selectedCount})
      </button>
      {#if doneSelected > 0}
        <button class="action-btn" disabled={downloading} onclick={downloadBatch}>
          Download ({doneSelected})
        </button>
      {/if}
    {/if}
    {#if isProcessing}
      <button class="action-btn danger" onclick={onCancel}>Cancel</button>
    {:else}
      <button
        class="action-btn primary"
        disabled={!active || active.phase === "decoding" || active.phase === "inferring" || active.phase === "blending"}
        onclick={() => active && onProcess(active.id)}
      >
        {active?.phase === "done" || active?.phase === "failed" ? "Reprocess" : "Process"}
      </button>
      <button class="action-btn" disabled={pendingCount === 0} onclick={onProcessAll}>
        Process All ({pendingCount})
      </button>
    {/if}
  </div>
</div>

{#if contextMenu}
  <div
    class="ctx-overlay"
    role="button"
    tabindex={-1}
    onclick={closeContextMenu}
    onkeydown={closeContextMenu}
    oncontextmenu={(e) => {
      e.preventDefault();
      closeContextMenu();
    }}
  ></div>
  <div class="ctx-menu" style="left:{contextMenu.x}px;top:{contextMenu.y}px">
    <button
      onclick={ctxDownload}
      disabled={!queue.items.find((i) => i.id === contextMenu?.itemId)?.restored}>Download</button
    >
    <button onclick={ctxReprocess}>Re-process</button>
    <button onclick={ctxDuplicate}>Duplicate</button>
    <hr />
    <button class="danger" onclick={ctxRemove}>Remove</button>
  </div>
{/if}

<style>
  .queue-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    height: 88px;
    background: var(--bg-queue);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .thumbs {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    flex: 1;
    height: 100%;
    align-items: center;
  }
  .thumbs::-webkit-scrollbar {
    height: 4px;
  }
  .thumbs::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 2px;
  }
  .thumb {
    position: relative;
    width: 64px;
    height: 64px;
    border-radius: var(--radius);
    overflow: hidden;
    flex-shrink: 0;
    background: var(--bg-thumb);
    border: 2px solid transparent;
    transition: border-color 0.15s;
  }
  .thumb.active {
    border-color: var(--accent);
  }
  .thumb.selected {
    border-color: var(--badge-processing);
  }
  .thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .thumb.add {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: var(--text-secondary);
    border: 2px dashed var(--border-strong);
  }
  .thumb.add:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .badge {
    position: absolute;
    bottom: 2px;
    right: 2px;
    min-width: 16px;
    height: 16px;
    padding: 0 4px;
    border-radius: 8px;
    font-size: 0.625rem;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
  }
  .actions {
    display: flex;
    gap: 8px;
    flex-shrink: 0;
  }
  .action-btn {
    padding: 8px 14px;
    border-radius: var(--radius);
    background: var(--bg-panel);
    color: var(--text-primary);
    font-size: 0.8125rem;
    font-weight: 500;
    transition: background 0.15s, opacity 0.15s;
  }
  .action-btn:hover:not(:disabled) {
    background: var(--bg-panel-hover);
  }
  .action-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .action-btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
  }
  .action-btn.primary:hover:not(:disabled) {
    background: var(--accent-hover);
  }
  .action-btn.danger {
    background: var(--badge-error);
    color: var(--text-on-accent);
  }
  .action-btn.danger:hover:not(:disabled) {
    opacity: 0.85;
  }
  .ctx-overlay {
    position: fixed;
    inset: 0;
    z-index: 100;
  }
  .ctx-menu {
    position: fixed;
    z-index: 101;
    background: var(--bg-panel);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 4px;
    min-width: 140px;
  }
  .ctx-menu button {
    display: block;
    width: 100%;
    text-align: left;
    padding: 8px 12px;
    border-radius: calc(var(--radius) - 2px);
    font-size: 0.8125rem;
    color: var(--text-primary);
  }
  .ctx-menu button:hover:not(:disabled) {
    background: var(--bg-panel-hover);
  }
  .ctx-menu button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .ctx-menu .danger {
    color: var(--badge-error);
  }
  .ctx-menu hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 4px 0;
  }
  .thumb.dragging {
    opacity: 0.4;
  }
  @media (max-width: 768px) {
    .queue-bar { height: 72px; padding: 6px; gap: 6px; }
    .thumb { width: 48px; height: 48px; }
    .action-btn { padding: 6px 10px; font-size: 0.75rem; }
  }
</style>
