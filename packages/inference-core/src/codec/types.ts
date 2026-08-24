// Codec types. See spec section "Codec layer".

/** RGB in CHW layout, [0,1]. Alpha (if present) is split off separately. */
export interface DecodedImage {
  rgb: Float32Array;
  alpha: Uint8ClampedArray | null; // HW [0,255]; null if source had no alpha
  width: number;
  height: number;
}

export type EncodeFormat = "png" | "jpeg" | "webp" | "avif";

export interface EncodeOpts {
  format: EncodeFormat;
  quality?: number; // 0..100; ignored for png
}

/** Native encode depends on OffscreenCanvas; some formats need WASM fallback. */
export const NATIVE_ENCODE_FORMATS: readonly EncodeFormat[] = ["png", "jpeg", "webp"] as const;
export const WASM_ENCODE_FORMATS: readonly EncodeFormat[] = ["avif"] as const;
