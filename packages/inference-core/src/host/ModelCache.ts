import { NetworkError } from "../errors.js";

/**
 * Streams a model from `url`, emitting byte progress, and caches the
 * response in the provided Cache API store. Repeat visits skip the network.
 */
export class ModelCache {
  constructor(private readonly cache: Cache) {}

  async fetch(
    url: string,
    onProgress?: (loaded: number, total: number) => void,
  ): Promise<Uint8Array> {
    const cached = await this.cache.match(url);
    if (cached) {
      const buffer = await cached.arrayBuffer();
      return new Uint8Array(buffer);
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new NetworkError({ url, status: response.status });
    }

    const total = Number(response.headers.get("content-length") ?? 0);
    const reader = response.body?.getReader();
    if (!reader) {
      const buffer = await response.arrayBuffer();
      await this.cache.put(url, new Response(buffer));
      return new Uint8Array(buffer);
    }

    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.(loaded, total);
    }

    const merged = mergeChunks(chunks, loaded);
    const copy = new ArrayBuffer(merged.byteLength);
    new Uint8Array(copy).set(merged);
    await this.cache.put(url, new Response(copy));
    return merged;
  }
}

function mergeChunks(chunks: Uint8Array[], totalLength: number): Uint8Array {
  const out = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}
