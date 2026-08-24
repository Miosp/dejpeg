# inference-core

Framework-free browser inference library for dejpeg. Provides a typed client API for running FBCNN (and future models) on images, with all heavy lifting absorbed into a Web Worker that ships with the package.

Plan 3 (the Astro/Svelte UI) consumes this package and never touches `effect`, the wire protocol, decode/encode, or tensors.

## What's inside

- **`InferenceClient`** — main-thread façade. State + subscribe + `loadModel`/`process`/`encode`/`dispose`.
- **Worker host** — bundled worker entry that owns engine + codec + pipeline.
- **Codec layer** — native-first decode/encode with lazy `@jsquash/*` fallback for HEIC/TIFF/AVIF.
- **Typed errors** — 13 tagged error classes; full `instanceof` narrowing across the worker boundary.

## Install

This package is a workspace member; consumers import via the workspace.

## Usage

### 1. Create a one-line worker file in the app

```ts
// apps/web/src/workers/inference.worker.ts
import "inference-core/worker";
```

### 2. Create the client

```ts
// apps/web/src/lib/inference.svelte.ts
import { createInferenceClient } from "inference-core";

export const inference = createInferenceClient({
  workerURL: new URL("../workers/inference.worker.ts", import.meta.url),
});
```

### 3. Use it

```ts
import { inference } from "./lib/inference.svelte";

// Load a model (one at a time in v1).
await inference.loadModel("fbcnn-color-real");

// Read state reactively (Svelte auto-subscribes via $state wrapper in the .svelte.ts file).
console.log(inference.state); // "ready"
console.log(inference.backend); // "webgpu" | "wasm"
console.log(inference.tileSize); // 256

// Process an image. Returns PNG blob + ImageBitmap preview.
const result = await inference.process({
  file: myFile,
  onProgress: (event) => updateProgressPanel(event),
});

// Re-encode the result for download.
const jpegBlob = await inference.encode(result.bitmap, { format: "jpeg", quality: 85 });
```

### Subscribe to state changes

```ts
const unsubscribe = inference.subscribe((snapshot) => {
  console.log(snapshot.state, snapshot.backend, snapshot.modelId);
});
```

### Cancellation

Pass an `AbortSignal`:

```ts
const controller = new AbortController();
cancelButton.onclick = () => controller.abort();
await inference.process({ file, signal: controller.signal });
```

### Errors

All thrown errors are typed class instances. Catch with `instanceof`:

```ts
import { ImageDecodeError, ClientBusy, NoModelLoaded, Cancelled } from "inference-core";

try {
  await inference.process({ file });
} catch (e) {
  if (e instanceof ImageDecodeError) showBadFileToast(e.filename);
  else if (e instanceof ClientBusy) showWaitToast();
  else if (e instanceof NoModelLoaded) showModelPickerFirst();
  else if (e instanceof Cancelled) /* silent */;
  else throw e;
}
```

## Sub-path exports

- `inference-core` (default) — the main-thread API.
- `inference-core/worker` — the worker entry. Import once for its side effect in your worker file.
- `inference-core/models` — static model registry for typed iteration (optional; `inference.availableModels` covers the runtime case).

## Tests

- `bun test` — all unit and integration tests.
- `bun run typecheck` — strict TypeScript.

## Dependencies

- Runtime: `effect` (internal, tree-shaken from the main-thread bundle).
- Optional peer: `@jsquash/heic`, `@jsquash/tiff`, `@jsquash/avif` (lazy-loaded on demand for non-native formats).
- Worker-only: `onnxruntime-web` (dynamic-imported).

## Spec

See `docs/superpowers/specs/2026-07-28-dejpegweb/10-inference-core-redesign.md` for the full design.
