import { test, expect, beforeEach } from "bun:test";
import { queue, type QueueItem } from "./queue.svelte";

function makeItem(id: string, phase: QueueItem["phase"] = "queued"): QueueItem {
  return {
    id,
    file: new File([], `test-${id}.jpg`),
    phase,
    startedAt: Date.now(),
  };
}

beforeEach(() => {
  queue.items = [];
  queue.activeId = null;
  queue.clearSelection();
});

test("enqueue creates pending items", () => {
  queue.enqueue([new File([], "a.jpg")]);
  expect(queue.items.length).toBe(1);
  expect(queue.items[0].phase).toBe("queued");
});

test("select sets activeId", () => {
  queue.items = [makeItem("1"), makeItem("2")];
  queue.select("1");
  expect(queue.activeId).toBe("1");
});

test("toggleSelect adds and removes from selectedIds", () => {
  queue.items = [makeItem("1"), makeItem("2")];
  queue.toggleSelect("1");
  expect(queue.selectedIds.has("1")).toBe(true);
  queue.toggleSelect("1");
  expect(queue.selectedIds.has("1")).toBe(false);
});

test("getPending returns only queued items", () => {
  queue.items = [makeItem("1", "queued"), makeItem("2", "done"), makeItem("3", "queued")];
  const pending = queue.getPending();
  expect(pending.length).toBe(2);
  expect(pending.map((i) => i.id)).toEqual(["1", "3"]);
});

test("getSelected returns selected items", () => {
  queue.items = [makeItem("1"), makeItem("2"), makeItem("3")];
  queue.toggleSelect("1");
  queue.toggleSelect("3");
  const selected = queue.getSelected();
  expect(selected.length).toBe(2);
  expect(selected.map((i) => i.id)).toEqual(["1", "3"]);
});
