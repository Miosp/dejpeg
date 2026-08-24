// Framework-free inference library for dejpeg.
// Default public surface — main-thread only. Effect-TS not in transitive graph.
export * from "./client/index.js";
export type { ModelDef, ModelParam, InputBinding, ModelTask } from "./models/types.js";
export { MODELS, MODELS_BY_ID, type ModelId } from "./models/index.js";
export {
  NoModelLoaded,
  UnknownError,
  ClientBusy,
  type ModelError,
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
} from "./errors.js";
