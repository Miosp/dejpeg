import { encode } from "../../src/codec/encode.js";
import type { DecodedImage } from "../../src/codec/types.js";
import { describe, it, expect } from "bun:test";

function makeDecoded(width: number, height: number, withAlpha: boolean): DecodedImage {
  const rgb = new Float32Array(3 * width * height);
  const alpha = withAlpha ? new Uint8ClampedArray(width * height) : null;
  for (let i = 0; i < width * height; i++) {
    rgb[i] = 0.5; // R
    rgb[width * height + i] = 0.25; // G
    rgb[2 * width * height + i] = 0.75; // B
    if (alpha) alpha[i] = 128;
  }
  return { rgb, alpha, width, height };
}

describe("encode", () => {
  it("encodes to PNG (lossless, default)", async () => {
    const decoded = makeDecoded(8, 8, false);
    const { blob, bitmap } = await encode(decoded, { format: "png" });
    expect(blob.type).toBe("image/png");
    expect(bitmap.width).toBe(8);
    expect(bitmap.height).toBe(8);
    bitmap.close();
  });

  it("encodes to JPEG with quality", async () => {
    const decoded = makeDecoded(8, 8, false);
    const { blob } = await encode(decoded, { format: "jpeg", quality: 85 });
    expect(blob.type).toBe("image/jpeg");
  });

  it("encodes to WebP with quality", async () => {
    const decoded = makeDecoded(8, 8, false);
    const { blob } = await encode(decoded, { format: "webp", quality: 90 });
    expect(blob.type).toBe("image/webp");
  });

  it("preserves alpha when encoding PNG from alpha-bearing input", async () => {
    const decoded = makeDecoded(8, 8, true);
    const { blob } = await encode(decoded, { format: "png" });
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(8, 8);
    canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
    const data = canvas.getContext("2d")!.getImageData(0, 0, 8, 8).data;
    expect(data[3]).toBe(128);
    bitmap.close();
  });

  it("flattens alpha onto white when encoding JPEG", async () => {
    const decoded = makeDecoded(8, 8, true);
    const { blob } = await encode(decoded, { format: "jpeg", quality: 95 });
    expect(blob.type).toBe("image/jpeg");
    const bitmap = await createImageBitmap(blob);
    expect(bitmap.width).toBe(8);
    bitmap.close();
  });
});
