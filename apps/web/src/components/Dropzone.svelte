<script lang="ts">
  let { handleFiles }: { handleFiles: (files: File[]) => void } = $props();
  let dragging = $state(false);

  let fileInput = $state<HTMLInputElement>();

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragging = false;
    const files = Array.from(e.dataTransfer?.files ?? []);
    if (files.length > 0) handleFiles(files);
  }

  function handleChange(e: Event) {
    const target = e.currentTarget as HTMLInputElement;
    const files = Array.from(target.files ?? []);
    if (files.length > 0) handleFiles(files);
    target.value = "";
  }
</script>

<input
  bind:this={fileInput}
  type="file"
  accept="image/*"
  multiple
  onchange={handleChange}
  hidden
/>

<div
  class="dropzone"
  class:dragging
  onclick={() => fileInput?.click()}
  onkeydown={(e) => { if (e.key === "Enter" || e.key === " ") fileInput?.click(); }}
  ondragover={(e) => { e.preventDefault(); dragging = true; }}
  ondragleave={() => (dragging = false)}
  ondrop={handleDrop}
  role="button"
  tabindex={0}
>
  <p>Drop images here or click to browse</p>
</div>

<style>
  .dropzone {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px dashed var(--border-strong);
    border-radius: var(--radius);
    color: var(--text-secondary);
    font-size: 1.125rem;
    cursor: pointer;
  }
  .dragging {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
  }
</style>
