import { persisted } from "./persist.svelte";
import type { ProgressEvent } from "inference-core";

export type CompareMode = "slider" | "split" | "toggle";

class UiState {
  #compareModeBox = persisted<CompareMode>("dejpeg:compareMode", "slider");
  progressEvents = $state<ProgressEvent[]>([]);
  /** Whether the progress panel is collapsed (drawer closed). */
  panelCollapsed = $state(false);
  /** True while user holds the "show original" key in toggle compare mode. */
  togglingOriginal = $state(false);
  /** Active tab in the settings panel. */
  settingsTab = $state<"basic" | "advanced">("basic");
  /** Whether the settings panel is collapsed. */
  settingsCollapsed = $state(false);

  get compareMode(): CompareMode {
    return this.#compareModeBox.value;
  }
  set compareMode(v: CompareMode) {
    this.#compareModeBox.value = v;
  }

  get currentPhase(): ProgressEvent | null {
    return this.progressEvents.length === 0
      ? null
      : this.progressEvents[this.progressEvents.length - 1]!;
  }

  pushProgress(e: ProgressEvent) {
    const next = [...this.progressEvents, e];
    if (next.length > 50) next.shift();
    this.progressEvents = next;
  }

  clear() {
    this.progressEvents = [];
  }
}

export const ui = new UiState();

export interface Toast {
  id: string;
  message: string;
  kind: "error" | "info" | "success";
  durationMs: number;
}

class ToastState {
  toasts = $state<Toast[]>([]);

  push(message: string, kind: Toast["kind"] = "info", durationMs = 5000) {
    const id = crypto.randomUUID();
    this.toasts = [...this.toasts, { id, message, kind, durationMs }];
    setTimeout(() => this.dismiss(id), durationMs);
  }

  dismiss(id: string) {
    this.toasts = this.toasts.filter((t) => t.id !== id);
  }

  clear() {
    this.toasts = [];
  }
}

export const toasts = new ToastState();
