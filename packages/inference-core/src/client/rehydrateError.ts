import {
  NetworkError,
  BackendUnavailable,
  UnsupportedOp,
  TileAllocationFailure,
  TileFloorExceeded,
  ImageDecodeError,
  CanvasCapExceeded,
  WorkerCrash,
  InvalidOutput,
  Cancelled,
  NoModelLoaded,
  UnknownError,
  type ModelError,
} from "../errors.js";
import type { SerializedError } from "../host/protocol.js";

type ErrorCtor = new (fields: Record<string, unknown>) => ModelError;

const ERROR_REGISTRY: Record<string, ErrorCtor> = {
  NetworkError: NetworkError as unknown as ErrorCtor,
  BackendUnavailable: BackendUnavailable as unknown as ErrorCtor,
  UnsupportedOp: UnsupportedOp as unknown as ErrorCtor,
  TileAllocationFailure: TileAllocationFailure as unknown as ErrorCtor,
  TileFloorExceeded: TileFloorExceeded as unknown as ErrorCtor,
  ImageDecodeError: ImageDecodeError as unknown as ErrorCtor,
  CanvasCapExceeded: CanvasCapExceeded as unknown as ErrorCtor,
  WorkerCrash: WorkerCrash as unknown as ErrorCtor,
  InvalidOutput: InvalidOutput as unknown as ErrorCtor,
  Cancelled: Cancelled as unknown as ErrorCtor,
  NoModelLoaded: NoModelLoaded as unknown as ErrorCtor,
  UnknownError: UnknownError as unknown as ErrorCtor,
};

export function rehydrateError(s: SerializedError): ModelError {
  const Ctor = ERROR_REGISTRY[s._tag] ?? (UnknownError as unknown as ErrorCtor);
  const { _tag, ...fields } = s;
  void _tag;
  return new Ctor(fields);
}
