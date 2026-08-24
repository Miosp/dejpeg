<script lang="ts">
  import { queue } from "../lib/state/queue.svelte";

  let active = $derived(queue.active);
  let hasError = $derived(active?.phase === "failed");
</script>

{#if hasError && active}
  <div class="error-overlay">
    <div class="error-card">
      <p class="error-icon">!</p>
      <p class="error-msg">{active.errorMessage ?? "Processing failed"}</p>
      <button class="retry-btn" onclick={() => queue.update(active.id, { phase: "queued", errorMessage: undefined, errorTag: undefined })}>
        Retry
      </button>
    </div>
  </div>
{/if}

<style>
  .error-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 15;
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(4px);
  }
  .error-card {
    background: var(--bg-panel);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    border-left: 3px solid var(--badge-error);
    padding: 24px 32px;
    text-align: center;
    max-width: 400px;
  }
  .error-icon {
    font-size: 2rem;
    font-weight: 700;
    color: var(--badge-error);
    margin: 0 0 8px 0;
  }
  .error-msg { color: var(--text-primary); margin: 0 0 16px 0; }
  .retry-btn {
    padding: 8px 24px;
    border-radius: var(--radius);
    background: var(--accent);
    color: var(--text-on-accent);
    font-weight: 500;
  }
  .retry-btn:hover { background: var(--accent-hover); }
</style>
