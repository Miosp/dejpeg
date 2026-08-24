// Native-first codec registry. WASM loaders are dynamic imports, resolved on
// demand the first time a non-native format arrives.

export interface WasmCodecLoader {
  readonly format: string;
  readonly loader: () => Promise<unknown>;
}

/**
 * WASM codec loaders. Lazy: the dynamic import only fires when the matching
 * format actually arrives. Adding a new WASM codec = adding one entry here.
 */
export const WASM_CODECS: readonly WasmCodecLoader[] = [
  { format: "heic", loader: () => import("@jsquash/heic") },
  { format: "tiff", loader: () => import("@jsquash/tiff") },
  { format: "avif", loader: () => import("@jsquash/avif") },
];

const wasmByFormat = new Map(WASM_CODECS.map((c) => [c.format, c.loader]));

/** Returns the lazy loader for a WASM-only format, or undefined if native handles it. */
export function getWasmLoader(format: string): (() => Promise<unknown>) | undefined {
  return wasmByFormat.get(format);
}

/** Does this format need WASM (i.e., no reliable native support)? */
export function requiresWasm(format: string): boolean {
  return wasmByFormat.has(format);
}
