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

function causeText(cause: unknown): string {
  if (cause === undefined || cause === null) return "";
  const text = cause instanceof Error ? cause.message : String(cause);
  return text && text !== "{}" && text !== "[object Object]" ? `: ${text}` : "";
}

/** A sentence for the person using the app. Never empty, whatever `e` is. */
export function describeError(e: unknown): string {
  const err = e as ModelError & { cause?: unknown };
  switch (err?._tag) {
    case "NetworkError":
      return `Could not download the model (status ${err.status ?? "?"}).`;
    case "BackendUnavailable":
      return `No suitable backend (${err.backend})${causeText(err.cause)}.`;
    case "UnsupportedOp":
      return `${err.modelId} needs the ${err.opType} operation, which this browser cannot run.`;
    case "TileAllocationFailure":
      return `Ran out of memory at tile size ${err.attemptedSize}${causeText(err.cause)}.`;
    case "TileFloorExceeded":
      return `${err.modelId} ran out of memory even at the smallest tile size (${err.floor}).`;
    case "ImageDecodeError":
      return `Could not decode ${err.filename}.`;
    case "CanvasCapExceeded":
      return `This image (${err.width}x${err.height}) is too large for this browser.`;
    case "WorkerCrash":
      return `The processing worker crashed${causeText(err.cause)}.`;
    case "InvalidOutput":
      return err.reason === "shape"
        ? `${err.modelId} returned output of the wrong shape${err.details ? `: ${err.details}` : ""}.`
        : `${err.modelId} produced invalid values on this device${err.details ? ` (${err.details})` : ""}. Try another model.`;
    case "Cancelled":
      return "Cancelled.";
    case "NoModelLoaded":
      return "No model is loaded yet. Pick a model first.";
    case "ClientBusy":
      return "Still working on the previous request.";
  }
  const message = e instanceof Error ? e.message : e == null ? "" : String(e);
  return message || "Something went wrong, but no reason was reported.";
}

/** The technical facts to include in a bug report: error type, model, reason, device. */
export function errorSummary(e: unknown): string {
  const err = (e ?? {}) as { _tag?: string; name?: string; modelId?: string; reason?: string; backend?: string; device?: string };
  return [err._tag ?? err.name ?? "Unknown", err.reason, err.modelId, err.device ?? err.backend]
    .filter(Boolean)
    .join(" | ");
}
