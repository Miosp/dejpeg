import { ModelCache } from "../../src/host/ModelCache.js";
import { describe, it, expect, mock, beforeEach } from "bun:test";

describe("ModelCache", () => {
  let cache: Cache;
  let modelCache: ModelCache;

  beforeEach(() => {
    cache = new Cache();
    modelCache = new ModelCache(cache);
  });

  it("fetches and caches a model by URL", async () => {
    const url = "https://example.com/model.onnx";
    const bytes = new Uint8Array([1, 2, 3]);

    globalThis.fetch = mock(async (_req: Request) => {
      return new Response(bytes, { status: 200, headers: { "content-length": "3" } });
    }) as unknown as typeof fetch;

    const onProgress = mock(() => {});
    const result = await modelCache.fetch(url, onProgress);
    expect(result).toEqual(bytes);

    // Second call hits the cache, no fetch.
    globalThis.fetch = mock(async () => {
      throw new Error("should not be called");
    }) as unknown as typeof fetch;
    const result2 = await modelCache.fetch(url, () => {});
    expect(result2).toEqual(bytes);
  });

  it("emits byte progress during fetch", async () => {
    const url = "https://example.com/model.onnx";
    const bytes = new Uint8Array(1024);
    globalThis.fetch = mock(async () => {
      return new Response(bytes, { status: 200, headers: { "content-length": "1024" } });
    }) as unknown as typeof fetch;

    const events: { loaded: number; total: number }[] = [];
    await modelCache.fetch(url, (loaded, total) => events.push({ loaded, total }));
    expect(events.length).toBeGreaterThan(0);
    expect(events[events.length - 1]!.total).toBe(1024);
  });
});
