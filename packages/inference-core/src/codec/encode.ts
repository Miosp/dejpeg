import type { DecodedImage, EncodeOpts, EncodeFormat } from "./types.js";
import { NATIVE_ENCODE_FORMATS } from "./types.js";

export interface EncodeResult {
  blob: Blob;
  bitmap: ImageBitmap;
}

/**
 * Encode a DecodedImage back to a Blob + ImageBitmap.
 * Native via OffscreenCanvas.convertToBlob for png/jpeg/webp.
 * AVIF via lazy-loaded @jsquash/avif.
 */
export async function encode(decoded: DecodedImage, opts: EncodeOpts): Promise<EncodeResult> {
  const imageData = recombineAlpha(decoded, opts.format);
  const canvas = new OffscreenCanvas(decoded.width, decoded.height);
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(imageData, 0, 0);

  const blob = await encodeFromCanvas(canvas, opts);
  const bitmap = await createImageBitmap(canvas);
  return { blob, bitmap };
}

function recombineAlpha(decoded: DecodedImage, format: EncodeFormat): ImageData {
  const { rgb, alpha, width, height } = decoded;
  const pixelCount = width * height;
  const data = new Uint8ClampedArray(pixelCount * 4);
  const channels = rgb.length === pixelCount ? 1 : 3;
  const formatSupportsAlpha = format === "png" || format === "webp";

  for (let i = 0; i < pixelCount; i++) {
    if (channels === 1) {
      const v = rgb[i]! * 255;
      data[i * 4] = v;
      data[i * 4 + 1] = v;
      data[i * 4 + 2] = v;
    } else {
      data[i * 4] = rgb[i]! * 255;
      data[i * 4 + 1] = rgb[pixelCount + i]! * 255;
      data[i * 4 + 2] = rgb[2 * pixelCount + i]! * 255;
    }
    if (formatSupportsAlpha && alpha) {
      data[i * 4 + 3] = alpha[i]!;
    } else if (formatSupportsAlpha) {
      data[i * 4 + 3] = 255;
    } else {
      // JPEG / AVIF: composite alpha over white background.
      if (alpha) {
        const a = alpha[i]! / 255;
        data[i * 4] = data[i * 4]! * a + 255 * (1 - a);
        data[i * 4 + 1] = data[i * 4 + 1]! * a + 255 * (1 - a);
        data[i * 4 + 2] = data[i * 4 + 2]! * a + 255 * (1 - a);
      }
      data[i * 4 + 3] = 255;
    }
  }

  return new ImageData(data, width, height);
}

async function encodeFromCanvas(canvas: OffscreenCanvas, opts: EncodeOpts): Promise<Blob> {
  if (NATIVE_ENCODE_FORMATS.includes(opts.format)) {
    const mime = opts.format === "jpeg" ? "image/jpeg" : `image/${opts.format}`;
    const convertOpts: { type: string; quality?: number } = { type: mime };
    if (opts.quality !== undefined) convertOpts.quality = opts.quality / 100;
    const blob = await canvas.convertToBlob(convertOpts);
    return blob;
  }

  if (opts.format === "avif") {
    const mod = (await import("@jsquash/avif")).default as {
      encode: (img: ImageData, opts?: { quality?: number }) => Promise<ArrayBuffer>;
    };
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const encOpts: { quality?: number } = {};
    if (opts.quality !== undefined) encOpts.quality = opts.quality;
    const buffer = await mod.encode(imageData, encOpts);
    return new Blob([buffer], { type: "image/avif" });
  }

  throw new Error(`Unsupported encode format: ${opts.format}`);
}
