import { Context } from "effect";
import type { InferenceEngine } from "../engine/types.js";

export interface EngineEnv {
  readonly engine: InferenceEngine;
}

export const EngineEnv = Context.GenericTag<EngineEnv>("EngineEnv");

export const makeEngineEnv = (engine: InferenceEngine): EngineEnv => ({ engine });
