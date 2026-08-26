import { test, expect } from "bun:test";
import { Effect } from "effect";
import {
  processImage,
  type PipelineInput,
  type ProgressSink,
} from "../src/pipeline/pipeline.js";
import {
  EngineEnv,
  makeEngineEnv,
} from "../src/pipeline/runtime.js";
import { MockEngine } from "../src/engine/mock.js";
import { dejpegC40 } from "../src/models/dejpegC40.js";
import type { ModelDef } from "../src/models/types.js";
import {
  CanvasCapExceeded,
  InvalidOutput,
  TileFloorExceeded,
} from "../src/index.js";
import type { Tensor } from "../src/engine/types.js";

type ImageProgress = Parameters<ProgressSink>[0];

function makeSyntheticImage(
  width: number,
  height: number,
  channels = 3,
  fill = 0.5,
): PipelineInput["image"] {
  return {
    data: new Float32Array(channels * width * height).fill(fill),
    channels,
    width,
    height,
  };
}

function run(
  image: PipelineInput["image"],
  engine: MockEngine,
  events: Parameters<ProgressSink>[0][],
  params: Record<string, number | string | boolean> = {},
) {
  return Effect.runPromiseExit(
    processImage({ itemId: "x", image }, dejpegC40, params, (e) =>
      events.push(e),
    ).pipe(Effect.provideService(EngineEnv, makeEngineEnv(engine))),
  );
}

test("pipeline runs end-to-end on a small image with MockEngine", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const exit = await run(makeSyntheticImage(128, 128), engine, events);
  expect(exit._tag).toBe("Success");
  if (exit._tag !== "Success") return;
  const out = exit.value;
  expect(out.width).toBe(128);
  expect(out.height).toBe(128);
  expect(out.channels).toBe(3);
  expect(out.data.length).toBe(3 * 128 * 128);

  const stages = new Set(events.map((e) => e.stage));
  expect(stages.has("plan")).toBe(true);
  expect(stages.has("tile")).toBe(true);
  expect(stages.has("blend")).toBe(true);
  expect(stages.has("finalize")).toBe(true);

  expect(engine.calls.length).toBeGreaterThan(0);
});

test("progress events fire in order: plan, tile(1..N), blend, finalize", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  await run(makeSyntheticImage(512, 512), engine, events);

  const stages = events.map((e) => e.stage);
  const planIdx = stages.indexOf("plan");
  const firstTileIdx = stages.indexOf("tile");
  const blendIdx = stages.indexOf("blend");
  const finalizeIdx = stages.indexOf("finalize");
  expect(planIdx).toBeLessThan(firstTileIdx);
  expect(firstTileIdx).toBeLessThan(blendIdx);
  expect(blendIdx).toBeLessThan(finalizeIdx);

  const tileEvents = events.filter((e) => e.stage === "tile");
  const dones = tileEvents.map((e) => e.done);
  expect(dones).toEqual(tileEvents.map((_, i) => i + 1));
  expect(tileEvents.at(-1)?.total).toBe(tileEvents.length);
});

test("single-tile case: image smaller than tile size still produces full output", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const exit = await run(makeSyntheticImage(64, 64), engine, events);

  expect(exit._tag).toBe("Success");
  if (exit._tag !== "Success") return;
  expect(exit.value.data.length).toBe(3 * 64 * 64);

  const planEvent = events.find((e) => e.stage === "plan");
  expect(planEvent?.total).toBe(1);
  const tileEvents = events.filter((e) => e.stage === "tile");
  expect(tileEvents.length).toBe(1);
});

test("multi-tile case: 1024x1024 emits >1 tile and produces correct shape", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const exit = await run(makeSyntheticImage(1024, 1024), engine, events);

  expect(exit._tag).toBe("Success");
  if (exit._tag !== "Success") return;
  expect(exit.value.data.length).toBe(3 * 1024 * 1024);

  const tileEvents = events.filter((e) => e.stage === "tile");
  expect(tileEvents.length).toBeGreaterThan(1);
});

test("pipeline preserves identity values when MockEngine passes input through", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const fill = 0.42;
  const exit = await run(makeSyntheticImage(128, 128, 3, fill), engine, events);

  expect(exit._tag).toBe("Success");
  if (exit._tag !== "Success") return;
  const out = exit.value.data;
  for (let i = 0; i < out.length; i++) {
    expect(out[i]).toBeCloseTo(fill, 5);
  }
});

test("pipeline rejects with InvalidOutput when engine emits NaN", async () => {
  const nanTensor: Tensor = {
    data: new Float32Array([NaN]),
    shape: [1, 1],
  };
  const engine = new MockEngine({
    produce: () => ({ output: nanTensor }),
  });
  const events: ImageProgress[] = [];
  const exit = await run(makeSyntheticImage(128, 128), engine, events);

  expect(exit._tag).toBe("Failure");
  if (exit._tag !== "Failure") return;
  const fail = exit.cause;
  expect(fail._tag).toBe("Fail");
  if (fail._tag !== "Fail") return;
  expect(fail.error).toBeInstanceOf(InvalidOutput);
  expect((fail.error as InvalidOutput).reason).toBe("nan");
});

test("pipeline rejects with CanvasCapExceeded on absurdly large image", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const exit = await run(
    {
      data: new Float32Array(0),
      channels: 3,
      width: 20000,
      height: 20000,
    },
    engine,
    events,
  );

  expect(exit._tag).toBe("Failure");
  if (exit._tag !== "Failure") return;
  const fail = exit.cause;
  expect(fail._tag).toBe("Fail");
  if (fail._tag !== "Fail") return;
  expect(fail.error).toBeInstanceOf(CanvasCapExceeded);
  expect(events.length).toBe(0);
});

test("pipeline surfaces TileFloorExceeded when settleTileSize cannot probe any size", async () => {
  const engine = new MockEngine({
    failWith: () => {
      const err = new Error("out of memory") as Error & { _tag: string };
      err._tag = "TileAllocationFailure";
      return err;
    },
  });
  const events: ImageProgress[] = [];
  const exit = await run(makeSyntheticImage(128, 128), engine, events);

  expect(exit._tag).toBe("Failure");
  if (exit._tag !== "Failure") return;
  const fail = exit.cause;
  expect(fail._tag).toBe("Fail");
  if (fail._tag !== "Fail") return;
  expect(fail.error).toBeInstanceOf(TileFloorExceeded);
});

test("plan progress total matches actual tile count", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  await run(makeSyntheticImage(512, 512), engine, events);
  const planEvent = events.find((e) => e.stage === "plan");
  const tileEvents = events.filter((e) => e.stage === "tile");
  expect(planEvent?.total).toBe(tileEvents.length);
});

test("tileSizeOverride bypasses the settleTileSize probe", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const exit = await Effect.runPromiseExit(
    processImage(
      { itemId: "x", image: makeSyntheticImage(64, 64) },
      dejpegC40,
      {},
      (e) => events.push(e),
      { tileSizeOverride: 32 },
    ).pipe(Effect.provideService(EngineEnv, makeEngineEnv(engine))),
  );

  expect(exit._tag).toBe("Success");
  if (exit._tag !== "Success") return;
  expect(exit.value.data.length).toBe(3 * 64 * 64);

  const planEvent = events.find((e) => e.stage === "plan");
  expect(planEvent?.total).toBeGreaterThan(1);
});

test("tileSizeOverride absent keeps existing probe behavior", async () => {
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const exit = await Effect.runPromiseExit(
    processImage(
      { itemId: "x", image: makeSyntheticImage(64, 64) },
      dejpegC40,
      {},
      (e) => events.push(e),
    ).pipe(Effect.provideService(EngineEnv, makeEngineEnv(engine))),
  );

  expect(exit._tag).toBe("Success");
  if (exit._tag !== "Success") return;
  expect(exit.value.data.length).toBe(3 * 64 * 64);

  const planEvent = events.find((e) => e.stage === "plan");
  expect(planEvent?.total).toBe(1);
});

test("param binding feeds qf pre-normalized to the engine", async () => {
  const qfModel: ModelDef = {
    id: "test-qf",
    name: "Test QF",
    description: "",
    task: "jpeg-artifact-removal",
    url: "file://test.onnx",
    sizeBytes: 1,
    channels: 3,
    alignment: 1,
    inputs: { input: "image", qf_input: { param: "qf" } },
    outputs: [{ name: "output" }],
    params: {
      qf: { kind: "range", min: 10, max: 100, step: 1, default: 40, label: "QF", help: "" },
    },
  };
  const engine = new MockEngine();
  const events: ImageProgress[] = [];
  const exit = await Effect.runPromiseExit(
    processImage(
      { itemId: "x", image: makeSyntheticImage(32, 32) },
      qfModel,
      { qf: 40 },
      (e) => events.push(e),
      { tileSizeOverride: 32 },
    ).pipe(Effect.provideService(EngineEnv, makeEngineEnv(engine))),
  );

  expect(exit._tag).toBe("Success");
  const feeds = engine.calls[0]!.feeds;
  expect(feeds.qf_input!.data[0]).toBeCloseTo(0.6, 6);
});
