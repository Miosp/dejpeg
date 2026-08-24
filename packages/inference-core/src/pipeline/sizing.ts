import { Effect } from "effect";
import type { InferenceEngine, Tensor } from "../engine/types.js";
import type { ModelDef } from "../models/types.js";
import { TileAllocationFailure, TileFloorExceeded } from "../errors.js";

export const TILE_SIZE_DEFAULT = 512;
export const TILE_SIZE_MIN = 64;

/**
 * Detect whether an error is an allocation-failure class that should trigger
 * tile shrinkage. We accept either:
 *   - The tagged error TileAllocationFailure (raised by OnnxEngine)
 *   - Plain errors whose message matches WebGPU/WASM OOM signatures
 *
 * Declared as a type guard so callers (and the probe below) can narrow; the
 * guard is intentionally liberal — when in doubt, treat as an allocation
 * failure so the sizer shrinks and retries rather than crashing.
 */
export function isAllocationFailure(e: unknown): e is TileAllocationFailure {
  if (e instanceof Error && (e as { _tag?: unknown })._tag === "TileAllocationFailure") {
    return true;
  }
  if (e instanceof Error) {
    const msg = e.message.toLowerCase();
    if (/out of memory|oom|gpuoutofmemory/.test(msg)) return true;
    if (/rangeerror.*array|invalid array length/.test(msg)) return true;
    if (/alloc|buffer.*memory/.test(msg)) return true;
  }
  return false;
}

/**
 * Settle the working tile size for the active model + backend.
 *
 * Strategy: start at the model's default tile size, probe via a zero-filled
 * forward pass, halve on allocation failure, repeat down to TILE_SIZE_MIN.
 * Fails with TileFloorExceeded if even the floor fails.
 *
 * Non-allocation errors during the probe (e.g. a model-def shape mismatch) are
 * not shrunk against — they surface as defects, since shrinking will not fix
 * them. The successful size should be cached by the caller per (modelId,
 * backend).
 */
export function settleTileSize(
  engine: InferenceEngine,
  def: ModelDef,
): Effect.Effect<number, TileFloorExceeded, never> {
  const align = def.alignment ?? 1;
  return Effect.gen(function* () {
    let size = def.tileSizeDefault ?? TILE_SIZE_DEFAULT;
    size = Math.max(size, align);

    while (size >= TILE_SIZE_MIN) {
      const ok = yield* probe(engine, def, size).pipe(
        Effect.as(true),
        Effect.catchTag("TileAllocationFailure", (e) =>
          Effect.logWarning(`tile ${e.attemptedSize} OOM, shrinking`).pipe(
            Effect.zipRight(Effect.succeed(false)),
          ),
        ),
      );
      if (ok) return size;
      if (size <= TILE_SIZE_MIN) break;
      size = Math.max(TILE_SIZE_MIN, size >> 1);
    }

    return yield* Effect.fail(
      new TileFloorExceeded({ modelId: def.id, floor: TILE_SIZE_MIN }),
    );
  });
}

/**
 * Run one forward pass on a zero-filled tile to test whether the size works.
 * Allocation-class failures become a typed TileAllocationFailure (so the
 * caller can shrink). Any other failure is a defect — the probe uses valid
 * shapes, so non-OOM errors indicate a model-definition bug.
 */
function probe(
  engine: InferenceEngine,
  def: ModelDef,
  size: number,
): Effect.Effect<void, TileAllocationFailure, never> {
  return Effect.tryPromise({
    try: async () => {
      const zeros = new Float32Array(def.channels * size * size);
      const feeds: Record<string, Tensor> = {
        input: { data: zeros, shape: [1, def.channels, size, size] },
      };
      for (const [name, binding] of Object.entries(def.inputs)) {
        if (binding === "image") continue;
        const param = def.params[binding.param];
        if (param && param.kind === "range") {
          feeds[name] = { data: new Float32Array([param.default]), shape: [1, 1] };
        }
      }
      await engine.run(feeds);
    },
    catch: (e: unknown): unknown => e,
  }).pipe(
    Effect.catchAll((e) =>
      isAllocationFailure(e)
        ? Effect.fail(
            new TileAllocationFailure({
              attemptedSize: size,
              backend: engine.backend,
              cause: e,
            }),
          )
        : Effect.die(e),
    ),
  );
}
