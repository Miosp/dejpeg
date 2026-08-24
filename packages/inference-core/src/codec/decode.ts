import type { DecodedImage } from "./types.js";
import { getWasmLoader } from "./registry.js";
import { ImageDecodeError } from "../errors.js";

/**
 * Decode a File into a DecodedImage. Native first (createImageBitmap);
 * WASM fallback via the registry if native throws.
 *
 * Alpha is always split off. The pipeline processes RGB only; encode
 * recombines alpha if the output format supports it.
 */
export async function decode(file: File, channels: 1 | 3): Promise<DecodedImage> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch (nativeErr) {
    const loader = getWasmLoader(file.type.split("/")[1] ?? "");
    if (!loader) {
      throw new ImageDecodeError({ filename: file.name, cause: nativeErr });
    }
    // WASM codecs typically hand back an ImageBitmap or ImageData; normalize.
    const mod = (await loader()) as { default: (buf: ArrayBuffer) => Promise<ImageBitmap> };
    const buf = await file.arrayBuffer();
    bitmap = await mod.default(buf);
  }

  // Draw to OffscreenCanvas to read pixels back as ImageData.
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();

  return splitAlpha(imageData, channels);
}

function splitAlpha(imageData: ImageData, channels: 1 | 3): DecodedImage {
  const { width, height, data } = imageData; // data is Uint8ClampedArray RGBA HWC
  const pixelCount = width * height;
  const rgb = new Float32Array(channels * pixelCount);
  const hasAlpha = channels === 3 && data.some((v, i) => i % 4 === 3 && v < 255);
  const alpha = hasAlpha ? new Uint8ClampedArray(pixelCount) : null;

  for (let i = 0; i < pixelCount; i++) {
    const r = data[i * 4]!;
    const g = data[i * 4 + 1]!;
    const b = data[i * 4 + 2]!;
    const a = data[i * 4 + 3]!;

    if (channels === 1) {
      // ITU-R BT.601 luma.
      rgb[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    } else {
      rgb[i] = r / 255;
      rgb[pixelCount + i] = g / 255;
      rgb[2 * pixelCount + i] = b / 255;
      if (alpha) alpha[i] = a;
    }
  }

  return { rgb, alpha, width, height };
}
