// Stub OffscreenCanvas + createImageBitmap + ImageData + ImageBitmap for the
// Bun test runtime, which has none of these. PNG round-trip is faked: pixels
// are encoded into the Blob with a magic header so they survive the
// File([blob], ...) wrapper. No PNG codec needed.
//
// This is a test-only polyfill; production code uses the real browser APIs.

const MAGIC_HI = 0xDEADC0FE;
const MAGIC_LO = 0xBAADF00D;
const HEADER_BYTES = 16;

class StubImageData {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
  readonly colorSpace: "srgb" | "display-p3" = "srgb";
  constructor(
    widthOrData: number | Uint8ClampedArray,
    heightOrWidth: number,
    maybeHeightOrData?: number | Uint8ClampedArray,
  ) {
    if (typeof widthOrData === "object") {
      // new ImageData(dataArray, width[, height])
      this.data = widthOrData;
      this.width = heightOrWidth;
      this.height =
        typeof maybeHeightOrData === "number"
          ? maybeHeightOrData
          : widthOrData.length / (4 * heightOrWidth);
    } else {
      // new ImageData(width, height[, data])  — polyfill-internal convention
      this.width = widthOrData;
      this.height = heightOrWidth;
      this.data =
        maybeHeightOrData instanceof Uint8ClampedArray
          ? maybeHeightOrData
          : new Uint8ClampedArray(this.width * this.height * 4);
    }
  }
}

class StubImageBitmap {
  readonly width: number;
  readonly height: number;
  readonly pixels: Uint8ClampedArray;
  constructor(width: number, height: number, pixels: Uint8ClampedArray) {
    this.width = width;
    this.height = height;
    this.pixels = pixels;
  }
  close(): void {}
}

class Stub2DContext {
  constructor(private canvas: StubOffscreenCanvas) {}
  createImageData(width: number, height: number): StubImageData {
    return new StubImageData(width, height);
  }
  putImageData(src: StubImageData): void {
    this.canvas.buf.set(src.data.subarray(0, this.canvas.buf.length));
  }
  drawImage(bitmap: StubImageBitmap): void {
    this.canvas.buf.set(bitmap.pixels.subarray(0, this.canvas.buf.length));
  }
  getImageData(_sx: number, _sy: number, width: number, height: number): StubImageData {
    return new StubImageData(width, height, this.canvas.buf);
  }
  clearRect(_x: number, _y: number, width: number, height: number): void {
    // Zero the underlying buffer for the full canvas — matches browser
    // clearRect semantics for opaque canvases closely enough for tests.
    this.canvas.buf.fill(0, 0, Math.min(width * height * 4, this.canvas.buf.length));
  }
}

// Minimal Cache API polyfill — Bun does not ship one. Backed by an
// in-memory Map keyed by URL string. Sufficient for ModelCache tests.
class StubCache {
  private store = new Map<string, Response>();

  async match(url: string | Request | URL): Promise<Response | undefined> {
    const key = typeof url === "string" ? url : url instanceof URL ? url.href : url.url;
    const hit = this.store.get(key);
    if (!hit) return undefined;
    // Return a fresh Response so callers can consume the body independently.
    return new Response(await hit.clone().arrayBuffer(), {
      status: hit.status,
      headers: hit.headers,
    });
  }
  async put(url: string | Request | URL, response: Response): Promise<void> {
    const key = typeof url === "string" ? url : url instanceof URL ? url.href : url.url;
    // Clone into a fresh Response backed by a Buffer so the body can be
    // read multiple times across future match() calls.
    const buf = await response.clone().arrayBuffer();
    this.store.set(
      key,
      new Response(buf, { status: response.status, headers: response.headers }),
    );
  }
}

class StubOffscreenCanvas {
  readonly width: number;
  readonly height: number;
  readonly buf: Uint8ClampedArray;
  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.buf = new Uint8ClampedArray(width * height * 4);
  }
  getContext(kind: "2d"): Stub2DContext | null {
    return kind === "2d" ? new Stub2DContext(this) : null;
  }
  async convertToBlob(opts?: { type?: string; quality?: number }): Promise<Blob> {
    const header = new ArrayBuffer(HEADER_BYTES);
    const dv = new DataView(header);
    dv.setUint32(0, MAGIC_HI, true);
    dv.setUint32(4, MAGIC_LO, true);
    dv.setUint32(8, this.width, true);
    dv.setUint32(12, this.height, true);
    const out = new Uint8Array(HEADER_BYTES + this.buf.length);
    out.set(new Uint8Array(header), 0);
    out.set(this.buf, HEADER_BYTES);
    const type = opts?.type ?? "image/png";
    return new Blob([out], { type });
  }
}

const g = globalThis as unknown as Record<string, unknown>;
g.OffscreenCanvas = StubOffscreenCanvas;
g.ImageData = StubImageData;
g.ImageBitmap = StubImageBitmap;
g.Cache = StubCache;
g.createImageBitmap = async (source: Blob | { width: number; height: number; buf: Uint8ClampedArray }): Promise<StubImageBitmap> => {
  // Canvas-like source: wrap pixel buffer directly (matches browser API
  // createImageBitmap(OffscreenCanvas)).
  if (!(source instanceof Blob) && "buf" in source) {
    return new StubImageBitmap(source.width, source.height, source.buf);
  }
  const bytes = new Uint8Array(await (source as Blob).arrayBuffer());
  if (bytes.length < HEADER_BYTES) {
    throw new Error(`canvas-polyfill: blob too small (${bytes.length} bytes)`);
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== MAGIC_HI || dv.getUint32(4, true) !== MAGIC_LO) {
    throw new Error("canvas-polyfill: blob missing magic header");
  }
  const width = dv.getUint32(8, true);
  const height = dv.getUint32(12, true);
  const pixels = new Uint8ClampedArray(bytes.length - HEADER_BYTES);
  pixels.set(bytes.subarray(HEADER_BYTES));
  return new StubImageBitmap(width, height, pixels);
};
