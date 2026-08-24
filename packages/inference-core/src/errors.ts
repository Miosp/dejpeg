import { Data } from "effect";

// --- Network / IO ---

export class NetworkError extends Data.TaggedError("NetworkError")<{
  url: string;
  status?: number | undefined;
  cause?: unknown;
}> {}

// --- Backend / engine ---

export class BackendUnavailable extends Data.TaggedError("BackendUnavailable")<{
  backend: "webgpu" | "wasm";
  cause?: unknown;
}> {}

export class UnsupportedOp extends Data.TaggedError("UnsupportedOp")<{
  opType: string;
  modelId: string;
}> {}

// --- Tiling ---

export class TileAllocationFailure extends Data.TaggedError(
  "TileAllocationFailure",
)<{
  attemptedSize: number;
  backend: "webgpu" | "wasm";
  cause?: unknown;
}> {}

export class TileFloorExceeded extends Data.TaggedError("TileFloorExceeded")<{
  modelId: string;
  floor: number;
}> {}

// --- Image / decode ---

export class ImageDecodeError extends Data.TaggedError("ImageDecodeError")<{
  filename: string;
  cause?: unknown;
}> {}

export class CanvasCapExceeded extends Data.TaggedError("CanvasCapExceeded")<{
  width: number;
  height: number;
  browserCap: number;
}> {}

// --- Worker ---

export class WorkerCrash extends Data.TaggedError("WorkerCrash")<{
  workerId: string;
  cause?: unknown;
}> {}

// --- Model output ---

export class InvalidOutput extends Data.TaggedError("InvalidOutput")<{
  modelId: string;
  reason: "nan" | "inf" | "shape";
  details?: string | undefined;
}> {}

// --- Cancellation ---

export class Cancelled extends Data.TaggedError("Cancelled")<{
  id: string;
}> {}

// --- Client-side ---

export class NoModelLoaded extends Data.TaggedError("NoModelLoaded")<{
  operation: "process" | "encode";
}> {}

// --- Defensive escape hatch ---

export class UnknownError extends Data.TaggedError("UnknownError")<{
  message: string;
  cause?: unknown;
}> {}

// --- Concurrency guard (client-local; never crosses the wire) ---

export class ClientBusy extends Data.TaggedError("ClientBusy")<{
  operation: "process" | "encode" | "load-model";
}> {}

export type ModelError =
  | NetworkError
  | BackendUnavailable
  | UnsupportedOp
  | TileAllocationFailure
  | TileFloorExceeded
  | ImageDecodeError
  | CanvasCapExceeded
  | WorkerCrash
  | InvalidOutput
  | Cancelled
  | NoModelLoaded
  | UnknownError
  | ClientBusy;
