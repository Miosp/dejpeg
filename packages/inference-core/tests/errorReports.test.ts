import { test, expect } from "bun:test";
import {
  describeError,
  errorSummary,
  InvalidOutput,
  TileAllocationFailure,
  TileFloorExceeded,
  UnknownError,
  WorkerCrash,
} from "../src/errors.js";
import { serializeError } from "../src/host/protocol.js";
import { rehydrateError } from "../src/client/rehydrateError.js";

test("errors without a message still get a readable reason", () => {
  for (const e of [
    new InvalidOutput({ modelId: "m", reason: "nan" }),
    new TileFloorExceeded({ modelId: "m", floor: 64 }),
    new TileAllocationFailure({ attemptedSize: 512, backend: "webgpu" }),
    new WorkerCrash({ workerId: "main" }),
  ]) {
    expect(e.message).toBe("");
    expect(describeError(e).length).toBeGreaterThan(10);
  }
});

test("describeError never returns an empty string", () => {
  for (const e of [undefined, null, "", new Error(""), {}]) {
    expect(describeError(e).length).toBeGreaterThan(0);
  }
  expect(describeError(new Error("boom"))).toBe("boom");
  expect(describeError(new UnknownError({ message: "odd" }))).toBe("odd");
});

test("the cause of an error survives the trip across the worker", () => {
  const sent = serializeError(
    new TileAllocationFailure({ attemptedSize: 512, backend: "webgpu", cause: new Error("device lost") }),
  );
  const e = rehydrateError(JSON.parse(JSON.stringify(sent)));
  expect(e).toBeInstanceOf(TileAllocationFailure);
  expect(describeError(e)).toContain("device lost");
});

test("errorSummary names the error, reason, model and device", () => {
  const e = rehydrateError({
    ...serializeError(new InvalidOutput({ modelId: "m", reason: "nan" })),
    device: "webgpu: intel gen-12lp",
  });
  expect(errorSummary(e)).toBe("InvalidOutput | nan | m | webgpu: intel gen-12lp");
});
