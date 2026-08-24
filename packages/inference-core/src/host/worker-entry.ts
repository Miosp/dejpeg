// Sub-path entry: inference-core/worker
// The host-side message handler boots here and binds to the worker's
// global `onmessage`. App workers should be one-line files that import
// this for its side effect:
//
//   import "inference-core/worker";
//
import { Host } from "./Host.js";
import { makeDefaultDependencies, MODELS } from "./dependencies.js";

// Set onmessage synchronously so messages from the main thread are buffered
// while the async boot (caches.open, etc.) completes. Without this, messages
// sent before boot finishes are silently dropped.
let host: Host | null = null;
const pending: MessageEvent[] = [];

self.onmessage = (event: MessageEvent) => {
  if (host) {
    host.handle(event.data);
  } else {
    pending.push(event);
  }
};

async function boot(): Promise<void> {
  const deps = await makeDefaultDependencies();
  host = new Host({ deps, availableModels: MODELS });

  // Drain buffered messages.
  const queued = pending.splice(0);
  for (const event of queued) {
    host.handle(event.data);
  }
}

void boot();
