import { persisted } from "./persist.svelte";
import { MODELS, type ModelId, type ModelParam } from "inference-core";

class EngineState {
  params = $state<Record<string, Record<string, number | string | boolean>>>({});
  errorMessage = $state<string | null>(null);
  tileOverride = $state<number | null>(null);
  batchOverride = $state<number | null>(null);

  /** Initialise defaults for a model's params if not already set. */
  ensureParamsFor(modelId: ModelId) {
    if (this.params[modelId]) return;
    const def = MODELS.find((m) => m.id === modelId);
    if (!def) return;
    const defaults: Record<string, number | string | boolean> = {};
    for (const [name, p] of Object.entries(def.params) as Array<[string, ModelParam]>) {
      defaults[name] = p.default;
    }
    this.params = { ...this.params, [modelId]: defaults };
  }

  /** Read the current params for a model (or empty object). */
  getParams(modelId: ModelId): Record<string, number | string | boolean> {
    return this.params[modelId] ?? {};
  }

  setParam(modelId: ModelId, name: string, value: number | string | boolean) {
    this.params = {
      ...this.params,
      [modelId]: { ...(this.params[modelId] ?? {}), [name]: value },
    };
  }
}

export const engine = new EngineState();

// Persist active model selection across reloads. Consumed by EditorApp
// (Task 8) to auto-load the last-used model on startup.
export const persistedModelId = persisted<string | null>(
  "dejpeg:activeModel",
  null,
);
