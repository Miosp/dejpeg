import { decode } from "../../src/codec/decode.js";
import { describe, it, expect } from "bun:test";

async function makePngBlob(width: number, height: number, withAlpha: boolean): Promise<Blob> {
  // Build an ImageData, draw to canvas, export as PNG.
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.createImageData(width, height);
  // Fill with a deterministic pattern: red=255 for pixel (0,0), gradient across.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      imageData.data[i] = (x * 255) / width; // R
      imageData.data[i + 1] = (y * 255) / height; // G
      imageData.data[i + 2] = 0; // B
      imageData.data[i + 3] = withAlpha ? (x === 0 ? 255 : (x * 255) / width) : 255; // A
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return await canvas.convertToBlob({ type: "image/png" });
}

describe("decode", () => {
  it("decodes an opaque PNG to RGB Float32Array with null alpha", async () => {
    const blob = await makePngBlob(8, 8, false);
    const file = new File([blob], "test.png", { type: "image/png" });
    const decoded = await decode(file, 3);
    expect(decoded.width).toBe(8);
    expect(decoded.height).toBe(8);
    expect(decoded.rgb.length).toBe(8 * 8 * 3);
    expect(decoded.alpha).toBeNull();
    // First channel should be the red gradient.
    expect(decoded.rgb[0]).toBeCloseTo(0, 1);
  });

  it("decodes an RGBA PNG and splits alpha", async () => {
    const blob = await makePngBlob(8, 8, true);
    const file = new File([blob], "test.png", { type: "image/png" });
    const decoded = await decode(file, 3);
    expect(decoded.rgb.length).toBe(8 * 8 * 3);
    expect(decoded.alpha).not.toBeNull();
    expect(decoded.alpha!.length).toBe(8 * 8);
    // Pixel (0,0) is opaque (a=255) — must be stored as 255, not 0.
    expect(decoded.alpha![0]).toBe(255);
  });

  it("decodes a grayscale image into a 1-channel rgb array when channels=1", async () => {
    const blob = await makePngBlob(8, 8, false);
    const file = new File([blob], "test.png", { type: "image/png" });
    const decoded = await decode(file, 1);
    expect(decoded.rgb.length).toBe(8 * 8 * 1);
    // BT.601 luma: 0.299*R + 0.587*G + 0.114*B, normalized to [0,1].
    // Pixel (0,0): R=0, G=0 → luma=0.
    expect(decoded.rgb[0]).toBeCloseTo(0, 2);
    // Pixel (7,0): R=223, G=0 → luma=0.299*223/255.
    expect(decoded.rgb[7]).toBeCloseTo((0.299 * 223) / 255, 2);
    // Pixel (0,7): R=0, G=223 → luma=0.587*223/255.
    expect(decoded.rgb[56]).toBeCloseTo((0.587 * 223) / 255, 2);
  });
});
