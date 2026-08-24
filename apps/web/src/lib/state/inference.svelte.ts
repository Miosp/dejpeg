import {
  createInferenceClient,
  MODELS,
  type InferenceClient,
  type ClientSnapshot,
} from "inference-core";

// Lazy singleton: the InferenceClient boots a Worker on construction, which
// fails outside a browser context. Defer construction until first property
// access via a Proxy so SSR can import this module safely.
let _client: InferenceClient | null = null;

function getClient(): InferenceClient {
  if (_client) return _client;
  if (typeof Worker === "undefined") {
    throw new Error(
      "InferenceClient cannot be constructed outside a browser context",
    );
  }
  _client = createInferenceClient({
    workerURL: new URL("../../workers/inference.worker.ts", import.meta.url),
    availableModels: MODELS,
  });
  // Mirror the client's snapshot into a $state object so Svelte components
  // can read reactive state. Subscribed once, at first construction.
  _client.subscribe((s) => {
    snapshot.state = s.state;
    snapshot.backend = s.backend;
    snapshot.modelId = s.modelId;
    snapshot.tileSize = s.tileSize;
  });
  return _client;
}

/** Proxy that lazily constructs the client on first property access. */
export const inference: InferenceClient = new Proxy({} as InferenceClient, {
  get(_target, prop, receiver) {
    const client = getClient();
    const value = Reflect.get(client, prop, receiver);
    return typeof value === "function" ? value.bind(client) : value;
  },
});

export const snapshot = $state<ClientSnapshot>({
  state: "idle",
  backend: null,
  modelId: null,
  tileSize: null,
});

export const download = $state<{
  active: boolean;
  label: string;
  loaded: number;
  total: number;
}>({
  active: false,
  label: "",
  loaded: 0,
  total: 0,
});

export async function fetchWithProgress(url: string, label: string): Promise<void> {
  download.active = true;
  download.label = label;
  download.loaded = 0;
  download.total = 0;

  try {
    const response = await fetch(url);
    const contentLength = response.headers.get("content-length");
    if (contentLength) download.total = Number(contentLength);

    const reader = response.body?.getReader();
    if (!reader) {
      return;
    }

    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value?.byteLength ?? 0;
      download.loaded = loaded;
    }
  } catch {
    // Fetch failed — let loadModel handle the error
  } finally {
    download.active = false;
  }
}
