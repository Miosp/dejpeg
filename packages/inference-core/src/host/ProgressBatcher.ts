import type { ProgressEvent } from "./protocol.js";

export interface ProgressBatcherOpts {
  flush: (events: ProgressEvent[]) => void;
  frameMs?: number; // default 16
}

/**
 * Batches progress events to at most one flush per ~16ms frame.
 * Order preserved within a batch. Coalesces naturally for per-tile events.
 */
export class ProgressBatcher {
  private pending: ProgressEvent[] = [];
  private scheduled = false;
  private readonly frameMs: number;
  private readonly flush: (events: ProgressEvent[]) => void;

  constructor(opts: ProgressBatcherOpts) {
    this.flush = opts.flush;
    this.frameMs = opts.frameMs ?? 16;
  }

  emit(ev: ProgressEvent): void {
    this.pending.push(ev);
    if (!this.scheduled) {
      this.scheduled = true;
      setTimeout(() => this.tick(), this.frameMs);
    }
  }

  flushNow(): void {
    if (this.pending.length === 0) return;
    this.scheduled = false;
    const events = this.pending;
    this.pending = [];
    this.flush(events);
  }

  private tick(): void {
    this.scheduled = false;
    this.flushNow();
  }
}
