<script lang="ts">
  import { toasts } from "../lib/state/ui.svelte";
</script>

{#if toasts.toasts.length > 0}
  <div class="toasts">
    {#each toasts.toasts as t (t.id)}
      <button class="toast toast-{t.kind}" onclick={() => toasts.dismiss(t.id)}>
        {t.message}
      </button>
    {/each}
  </div>
{/if}

<style>
  .toasts {
    position: fixed;
    bottom: 100px;
    right: 16px;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 8px;
    pointer-events: none;
  }
  .toast {
    padding: 10px 16px;
    border: none;
    border-radius: var(--radius);
    background: var(--bg-panel);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
    pointer-events: auto;
    cursor: pointer;
    max-width: 400px;
    font: inherit;
    color: inherit;
    text-align: left;
  }
  .toast-error { border-left: 3px solid var(--badge-error); }
  .toast-success { border-left: 3px solid var(--badge-done); }
  .toast-info { border-left: 3px solid var(--badge-processing); }
</style>
