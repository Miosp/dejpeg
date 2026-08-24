<script lang="ts">
  import { queue } from "../lib/state/queue.svelte";

  let processing = $derived(
    queue.items.find((i) =>
      i.phase === "decoding" || i.phase === "inferring" || i.phase === "blending"
    )
  );

  let percent = $derived(
    processing?.tileProgress
      ? Math.round((processing.tileProgress.done / processing.tileProgress.total) * 100)
      : processing ? 0 : 100
  );
</script>

{#if processing}
  <div class="progress-top-bar">
    <div class="progress-fill" style="width:{percent}%"></div>
  </div>
{/if}

<style>
  .progress-top-bar {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: transparent;
    z-index: 10;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.2s ease;
  }
</style>
