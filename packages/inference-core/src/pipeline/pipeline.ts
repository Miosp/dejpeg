import { Chunk, Effect, Stream } from "effect";
import { EngineEnv } from "./runtime.js";
import { settleTileSize, isAllocationFailure } from "./sizing.js";
import { computePlan, type Tile, type TilePlan } from "../tiling/slicer.js";
import { combineTiles, finalizeBlend } from "../tiling/blender.js";
import type { Tensor } from "../engine/types.js";
import type { InputBinding, ModelDef } from "../models/types.js";
import {
  CanvasCapExceeded,
  Cancelled,
  InvalidOutput,
  type ModelError,
} from "../errors.js";

export interface PipelineInput {
  itemId: string;
  /** Decoded image data, CHW layout, float32 in [0,1]. */
  image: {
    data: Float32Array;
    channels: number;
    width: number;
    height: number;
  };
}

export interface PipelineOutput {
  itemId: string;
  /** Restored image data, same layout as input. */
  data: Float32Array;
  channels: number;
  width: number;
  height: number;
  /** Model-predicted JPEG quality factor, if the model outputs one. */
  qfPredicted?: number;
}

export type ProgressSink = (event: {
  kind: "image";
  itemId: string;
  stage: "plan" | "tile" | "blend" | "finalize";
  done?: number;
  total?: number;
  ms?: number;
}) => void;

const BROWSER_CANVAS_CAP_PIXELS = 268_000_000;

/**
 * Tiles merged into one engine.run() batch. 1 disables batching. Measured on
 * Intel Xe-LP (gen-12lp): batching 4 tiles is ~2x SLOWER per tile than single
 * runs (activation spilling on the shared-memory iGPU), so the default is 1.
 * The machinery stays for GPUs with dedicated VRAM; tune via ProcessImageOpts.
 */
export const TILE_BATCH = 1;

/**
 * Full image processing pipeline. Caller supplies the decoded image and a
 * sink for progress events; the pipeline settles tile size, runs the tile
 * loop, blends, and returns the restored Float32Array.
 */
export interface ProcessImageOpts {
  tileSizeOverride?: number;
  /** Overrides TILE_BATCH for this run. */
  tileBatch?: number;
  /** Abort signal, checked between tiles and before blend/encode. */
  signal?: AbortSignal;
}

export function processImage(
  input: PipelineInput,
  def: ModelDef,
  params: Record<string, number | string | boolean>,
  sink: ProgressSink,
  opts?: ProcessImageOpts,
): Effect.Effect<PipelineOutput, ModelError, EngineEnv> {
  return Effect.gen(function* () {
    const { engine } = yield* EngineEnv;
    let qfPredicted: number | undefined;

    const pixels = input.image.width * input.image.height;
    if (pixels > BROWSER_CANVAS_CAP_PIXELS) {
      yield* Effect.fail(
        new CanvasCapExceeded({
          width: input.image.width,
          height: input.image.height,
          browserCap: BROWSER_CANVAS_CAP_PIXELS,
        }),
      );
    }

    const tileSize = opts?.tileSizeOverride ?? (yield* settleTileSize(engine, def));
    const overlap = Math.max(8, tileSize >> 3);
    const signal = opts?.signal;
    const checkAborted = Effect.gen(function* () {
      if (signal?.aborted) {
        return yield* Effect.fail(new Cancelled({ id: input.itemId }));
      }
    });

    const plan = computePlan(
      input.image.width,
      input.image.height,
      tileSize,
      overlap,
    );
    sink({
      kind: "image",
      itemId: input.itemId,
      stage: "plan",
      total: plan.tiles.length,
    });

    /**
     * Run one chunk of tiles through the engine as a single batched
     * forward pass, splitting the batched output back into per-tile rows.
     */
    const runChunk = (
      chunk: ReadonlyArray<Tile>,
    ): Effect.Effect<ReadonlyArray<readonly [number, Float32Array]>, ModelError> =>
      Effect.gen(function* () {
        const feeds = extractBatchedFeeds(input.image, chunk, def, params);
        const outputs = yield* Effect.tryPromise({
          try: () => engine.run(feeds),
          catch: (e): ModelError => e as unknown as ModelError,
        });
        const out = outputs["output"];
        const first = chunk[0]!;
        if (first.index === 0) {
          const qfOut = outputs["qf_predicted"];
          if (qfOut && qfOut.data.length > 0) {
            qfPredicted = Math.round(qfOut.data[0]! * 100);
          }
        }
        if (!out) {
          return yield* Effect.fail(
            new InvalidOutput({
              modelId: def.id,
              reason: "shape",
              details: "engine returned no 'output' tensor",
            }),
          );
        }
        const per = def.channels * first.h * first.w;
        for (const v of out.data) {
          if (!Number.isFinite(v)) {
            return yield* Effect.fail(
              new InvalidOutput({ modelId: def.id, reason: "nan" }),
            );
          }
        }
        if (out.data.length !== chunk.length * per) {
          return yield* Effect.fail(
            new InvalidOutput({
              modelId: def.id,
              reason: "shape",
              details: `expected [${chunk.length},${def.channels},${first.h},${first.w}] output, got [${out.shape.join(",")}]`,
            }),
          );
        }
        const results: Array<readonly [number, Float32Array]> = [];
        for (let j = 0; j < chunk.length; j++) {
          const slice = out.data.subarray(j * per, (j + 1) * per);
          if (first.index === 0 && j === 0) {
            (globalThis as Record<string, unknown>).__debugTileOutput = {
              data: slice.slice(),
              shape: [...out.shape],
              inputShape: [chunk.length, def.channels, first.h, first.w],
            };
          }
          results.push([chunk[j]!.index, slice as Float32Array]);
        }
        return results;
      });

    const batchSize = Math.max(1, opts?.tileBatch ?? TILE_BATCH);
    const chunks: Tile[][] = [];
    for (let i = 0; i < plan.tiles.length; i += batchSize) {
      chunks.push(plan.tiles.slice(i, i + batchSize));
    }

    const tileOutputs = yield* Stream.fromIterable(chunks).pipe(
      Stream.mapEffect((chunk) =>
        Effect.gen(function* () {
          yield* checkAborted;
          const t0 = performance.now();
          const outputs = yield* runChunk(chunk).pipe(
            // A batched run holds batchSize times the per-tile activations;
            // when the device refuses the allocation, retry the chunk one
            // tile at a time (the size settleTileSize approved).
            Effect.catchIf(
              (e) => isAllocationFailure(e),
              () =>
                Effect.gen(function* () {
                  const singles: Array<readonly [number, Float32Array]> = [];
                  for (const tile of chunk) {
                    singles.push(...(yield* runChunk([tile])));
                  }
                  return singles;
                }),
            ),
          );
          const ms = (performance.now() - t0) / chunk.length;
          for (const [index] of outputs) {
            sink({
              kind: "image",
              itemId: input.itemId,
              stage: "tile",
              done: index + 1,
              total: plan.tiles.length,
              ms,
            });
          }
          return outputs;
        }),
      ),
      Stream.runCollect,
    ).pipe(
      Effect.map((chunkResults) =>
        Chunk.toReadonlyArray(chunkResults).flat(),
      ),
    );

    yield* checkAborted;
    sink({ kind: "image", itemId: input.itemId, stage: "blend" });
    const tileMap = new Map<number, Float32Array>(Array.from(tileOutputs));
    const accumulated = combineTiles(plan, input.image.channels, tileMap);
    const finalData = finalizeBlend(
      accumulated,
      input.image.channels,
      input.image.width * input.image.height,
    );

    // Restoration output is unbounded (no final sigmoid). The Python
    // reference clips to [0,1] via np.clip(output, 0, 1). Without this,
    // out-of-range values produce green/black/white artifacts when cast to
    // uint8.
    for (let i = 0; i < finalData.length; i++) {
      const v = finalData[i]!;
      finalData[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }

    sink({ kind: "image", itemId: input.itemId, stage: "finalize" });
    return {
      itemId: input.itemId,
      data: finalData,
      channels: input.image.channels,
      width: input.image.width,
      height: input.image.height,
      qfPredicted,
    };
  });
}

/**
 * Build the feeds map for one tile. Image binding extracts the tile region in
 * CHW layout from the full image (zero-padding where the tile overhangs the
 * image, which computePlan allows for last-tile snap). Param bindings pull
 * from caller-supplied params or the model's default.
 */
function extractTileFeeds(
  image: PipelineInput["image"],
  tile: Tile,
  def: ModelDef,
  params: Record<string, number | string | boolean>,
): Record<string, Tensor> {
  const feeds: Record<string, Tensor> = {};

  for (const [name, binding] of Object.entries(def.inputs) as Array<
    [string, InputBinding]
  >) {
    if (binding === "image") {
      feeds[name] = extractImageTile(image, tile);
    } else {
      const value =
        params[binding.param] ?? def.params[binding.param]?.default ?? 0;
      // A "qf" param is fed pre-normalized: 1 - qf/100. Models with a
      // QF-embedding head are trained on [0, 1] inputs, not raw quality
      // factors.
      const feedValue = binding.param === "qf"
        ? 1 - Number(value) / 100
        : Number(value);
      feeds[name] = {
        data: new Float32Array([feedValue]),
        shape: [1, 1],
      };
    }
  }

  return feeds;
}

function extractImageTile(
  image: PipelineInput["image"],
  tile: Tile,
): Tensor {
  const { data: src, channels, width: imgW, height: imgH } = image;
  const tileData = new Float32Array(channels * tile.w * tile.h);
  for (let c = 0; c < channels; c++) {
    const srcChanOff = c * imgW * imgH;
    const dstChanOff = c * tile.w * tile.h;
    for (let dy = 0; dy < tile.h; dy++) {
      const srcY = tile.y0 + dy;
      if (srcY < 0 || srcY >= imgH) continue;
      const srcRow = srcChanOff + srcY * imgW;
      const dstRow = dstChanOff + dy * tile.w;
      for (let dx = 0; dx < tile.w; dx++) {
        const srcX = tile.x0 + dx;
        if (srcX < 0 || srcX >= imgW) continue;
        tileData[dstRow + dx] = src[srcRow + srcX]!;
      }
    }
  }
  return { data: tileData, shape: [1, channels, tile.h, tile.w] };
}

/**
 * Build one feeds map for a whole chunk: per-tile tensors concatenated along
 * a new leading batch axis ([B,C,H,W] for image bindings, [B,1] for scalar
 * params). A single-tile chunk skips the copy and reuses the per-tile feeds.
 */
function extractBatchedFeeds(
  image: PipelineInput["image"],
  chunk: ReadonlyArray<Tile>,
  def: ModelDef,
  params: Record<string, number | string | boolean>,
): Record<string, Tensor> {
  if (chunk.length === 1) {
    return extractTileFeeds(image, chunk[0]!, def, params);
  }
  const perTile = chunk.map((t) => extractTileFeeds(image, t, def, params));
  const feeds: Record<string, Tensor> = {};
  for (const name of Object.keys(perTile[0]!)) {
    const first = perTile[0]![name]!;
    const data = new Float32Array(first.data.length * chunk.length);
    for (let j = 0; j < chunk.length; j++) {
      data.set(perTile[j]![name]!.data, j * first.data.length);
    }
    feeds[name] = { data, shape: [chunk.length, ...first.shape.slice(1)] };
  }
  return feeds;
}

export type { TilePlan };
