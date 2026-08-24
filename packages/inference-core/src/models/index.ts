import { fbcnnColorReal } from "./fbcnnColorReal.js";
import { fbcnnGray } from "./fbcnnGray.js";
import { fbcnnGrayDouble } from "./fbcnnGrayDouble.js";
import type { ModelDef } from "./types.js";

export const MODELS: readonly ModelDef[] = [fbcnnColorReal, fbcnnGray, fbcnnGrayDouble] as const;

export const MODELS_BY_ID: Record<string, ModelDef> = Object.fromEntries(
  MODELS.map((m) => [m.id, m] as const),
);

export type ModelId = (typeof MODELS)[number]["id"];
