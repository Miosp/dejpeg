export type QueueItemPhase =
  | "queued"
  | "decoding"
  | "inferring"
  | "blending"
  | "done"
  | "failed"
  | "cancelled";

export interface QueueItem {
  id: string;
  file: File;
  /** Cumulative tile progress, e.g. "3 of 9 tiles done". */
  tileProgress?: { done: number; total: number };
  phase: QueueItemPhase;
  errorMessage?: string;
  errorTag?: string;
  startedAt?: number;
  finishedAt?: number;
  /** Set when processing completes. */
  restored?: { bitmap: ImageBitmap; blob: Blob; width: number; height: number };
  qfPredicted?: number;
  elapsedMs?: number;
  /** Original decoded bitmap, kept for compare UI. */
  original?: ImageBitmap;
  /** Object URL for thumbnail display. */
  thumbnailUrl?: string;
}

class QueueState {
  items = $state<QueueItem[]>([]);
  activeId = $state<string | null>(null);
  selectedIds = $state<Set<string>>(new Set());

  get active(): QueueItem | null {
    return this.items.find((i) => i.id === this.activeId) ?? null;
  }

  enqueue(files: File[]): QueueItem[] {
    const added: QueueItem[] = [];
    for (const file of files) {
      const id = crypto.randomUUID();
      const thumbnailUrl = URL.createObjectURL(file);
      const item: QueueItem = {
        id,
        file,
        phase: "queued",
        startedAt: Date.now(),
        thumbnailUrl,
      };
      this.items = [...this.items, item];
      added.push(item);
    }
    return added;
  }

  update(id: string, patch: Partial<QueueItem>) {
    this.items = this.items.map((i) => (i.id === id ? { ...i, ...patch } : i));
  }

  markActive(id: string) {
    this.activeId = id;
    this.update(id, { phase: "decoding" });
  }

  // activeId is intentionally NOT cleared on terminal transitions.
  // The completed/failed/cancelled item stays selected so its result (or error)
  // remains visible in QueueBar. The EditorApp process loop gates
  // on item phase (decoding/inferring/blending), not on activeId === null, so
  // subsequent queued items still process. Selection changes when a new item is
  // marked active (processNext) or via click-to-select (Plan 4).
  markDone(id: string, restored: NonNullable<QueueItem["restored"]>) {
    this.update(id, { phase: "done", finishedAt: Date.now(), restored });
  }

  markFailed(id: string, tag: string, message: string) {
    this.update(id, {
      phase: "failed",
      finishedAt: Date.now(),
      errorTag: tag,
      errorMessage: message,
    });
  }

  markCancelled(id: string) {
    this.update(id, { phase: "cancelled", finishedAt: Date.now() });
  }

  select(id: string) {
    this.activeId = id;
    this.clearSelection();
  }

  toggleSelect(id: string) {
    const next = new Set(this.selectedIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    this.selectedIds = next;
  }

  selectMany(ids: string[]) {
    this.selectedIds = new Set(ids);
  }

  clearSelection() {
    this.selectedIds = new Set();
  }

  getSelected(): QueueItem[] {
    return this.items.filter((i) => this.selectedIds.has(i.id));
  }

  getPending(): QueueItem[] {
    return this.items.filter((i) => i.phase === "queued");
  }

  remove(id: string) {
    const item = this.items.find((i) => i.id === id);
    if (item?.thumbnailUrl) URL.revokeObjectURL(item.thumbnailUrl);
    this.items = this.items.filter((i) => i.id !== id);
    if (this.activeId === id) this.activeId = null;
    this.selectedIds = new Set([...this.selectedIds].filter((sid) => sid !== id));
  }
}

export const queue = new QueueState();
