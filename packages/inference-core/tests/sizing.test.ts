import { test, expect } from "bun:test";
import { Effect } from "effect";
import { settleTileSize, TILE_SIZE_MIN, isAllocationFailure } from "../src/pipeline/sizing.js";
import { MockEngine } from "../src/engine/mock.js";
import { fbcnnColorReal } from "../src/models/fbcnnColorReal.js";
import { TileFloorExceeded } from "../src/errors.js";

test("settleTileSize returns default when probe succeeds first try", async () => {
  const e = new MockEngine();
  const size = await Effect.runPromise(settleTileSize(e, fbcnnColorReal));
  expect(size).toBe(512);
});

test("settleTileSize shrinks when default size OOMs", async () => {
  // Fail at 512 and 256; pass at 128.
  const e = new MockEngine({ failAtTileSizes: [512, 256] });
  const size = await Effect.runPromise(settleTileSize(e, fbcnnColorReal));
  expect(size).toBe(128);
});

test("settleTileSize fails with TileFloorExceeded when floor still OOMs", async () => {
  // Fail at every size the loop tries (256, 128, 64, 32).
  const e = new MockEngine({
    failWith: () => {
      const err = new Error("out of memory") as Error & { _tag: string };
      err._tag = "TileAllocationFailure";
      return err;
    },
  });
  const result = await Effect.runPromiseExit(settleTileSize(e, fbcnnColorReal));
  expect(result._tag).toBe("Failure");
  if (result._tag === "Failure") {
    const cause = result.cause;
    expect(cause._tag).toBe("Fail");
    if (cause._tag === "Fail") {
      expect(cause.error).toBeInstanceOf(TileFloorExceeded);
      expect((cause.error as TileFloorExceeded).floor).toBe(TILE_SIZE_MIN);
    }
  }
});

test("isAllocationFailure detects WebGPU OOM message", () => {
  expect(isAllocationFailure(new Error("GPU out of memory"))).toBe(true);
  expect(isAllocationFailure(new Error("allocation failed"))).toBe(true);
});

test("isAllocationFailure rejects non-OOM errors", () => {
  expect(isAllocationFailure(new Error("bad input shape"))).toBe(false);
  expect(isAllocationFailure(new Error("unsupported op"))).toBe(false);
});
