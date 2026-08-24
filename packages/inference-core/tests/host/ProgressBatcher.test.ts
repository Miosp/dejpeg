import { ProgressBatcher } from "../../src/host/ProgressBatcher.js";
import { describe, it, expect, mock } from "bun:test";

describe("ProgressBatcher", () => {
  it("batches multiple emits within one frame", async () => {
    let captured: unknown[] = [];
    const batcher = new ProgressBatcher({
      flush: (events) => {
        captured = events;
      },
      frameMs: 16,
    });

    batcher.emit({ kind: "image", itemId: "x", stage: "tile", done: 1, total: 9 });
    batcher.emit({ kind: "image", itemId: "x", stage: "tile", done: 2, total: 9 });
    batcher.emit({ kind: "image", itemId: "x", stage: "tile", done: 3, total: 9 });

    expect(captured).toEqual([]); // not flushed yet

    await new Promise((r) => setTimeout(r, 32)); // wait two frames

    expect(captured).toHaveLength(3);
    expect(captured[0]).toMatchObject({ done: 1 });
    expect(captured[2]).toMatchObject({ done: 3 });
  });

  it("flushes immediately when flushNow() is called", () => {
    let captured: unknown[] = [];
    const batcher = new ProgressBatcher({
      flush: (events) => {
        captured = events;
      },
      frameMs: 16,
    });

    batcher.emit({ kind: "image", itemId: "x", stage: "decode" });
    batcher.flushNow();

    expect(captured).toHaveLength(1);
  });

  it("never calls flush when there are no pending events", () => {
    const flush = mock(() => {});
    const batcher = new ProgressBatcher({ flush, frameMs: 16 });
    batcher.flushNow();
    expect(flush).not.toHaveBeenCalled();
  });
});
