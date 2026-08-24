import { Effect, Stream } from "effect";
import { EngineEnv } from "./runtime.js";
import { settleTileSize } from "./sizing.js";
import { computePlan, type Tile, type TilePlan } from "../tiling/slicer.js";
import { combineTiles, finalizeBlend } from "../tiling/blender.js";
import type { Tensor } from "../engine/types.js";
import type { InputBinding, ModelDef } from "../models/types.js";
import {
  CanvasCapExceeded,
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
 * Full image processing pipeline. Caller supplies the decoded image and a
 * sink for progress events; the pipeline settles tile size, runs the tile
 * loop, blends, and returns the restored Float32Array.
 */
export interface ProcessImageOpts {
  tileSizeOverride?: number;
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

    const tileOutputs = yield* Stream.fromIterable(plan.tiles).pipe(
      Stream.zipWithIndex,
      Stream.mapEffect(([tile, i]) =>
        Effect.gen(function* () {
          const t0 = performance.now();
          const feeds = extractTileFeeds(input.image, tile, def, params);
          const outputs = yield* Effect.tryPromise({
            try: () => engine.run(feeds),
            catch: (e): ModelError => e as unknown as ModelError,
          });
          const out = outputs["output"];
          if (i === 0) {
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
          for (const v of out.data) {
            if (!Number.isFinite(v)) {
              return yield* Effect.fail(
                new InvalidOutput({ modelId: def.id, reason: "nan" }),
              );
            }
          }
          // Diagnostic: save first tile's raw model output for debugging.
          if (i === 0) {
            const tileCopy = out.data.slice();
            (globalThis as Record<string, unknown>).__debugTileOutput = {
              data: tileCopy,
              shape: [...out.shape],
              inputShape: [1, def.channels, tile.h, tile.w],
            };
          }
          const ms = performance.now() - t0;
          sink({
            kind: "image",
            itemId: input.itemId,
            stage: "tile",
            done: i + 1,
            total: plan.tiles.length,
            ms,
          });
          return [tile.index, out.data] as const;
        }),
      ),
      Stream.runCollect,
    );

    sink({ kind: "image", itemId: input.itemId, stage: "blend" });
    const tileMap = new Map<number, Float32Array>(Array.from(tileOutputs));
    const accumulated = combineTiles(plan, input.image.channels, tileMap);
    const finalData = finalizeBlend(
      accumulated,
      input.image.channels,
      input.image.width * input.image.height,
    );

    // FBCNN output is unbounded (no final sigmoid). The Python reference
    // clips to [0,1] via np.clip(output, 0, 1). Without this, out-of-range
    // values produce green/black/white artifacts when cast to uint8.
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
      // FBCNN expects pre-normalized QF: 1 - qf/100 (see original code:
      // qf_input = torch.tensor([[1 - QF_set/100]])). The qf_embed MLP was
      // trained on [0, 1] inputs, not raw quality factors.
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

export type { TilePlan };
