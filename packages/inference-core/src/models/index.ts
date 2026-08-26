import { dejpegC40 } from "./dejpegC40.js";
import type { ModelDef } from "./types.js";

export const MODELS: readonly ModelDef[] = [dejpegC40] as const;

export const MODELS_BY_ID: Record<string, ModelDef> = Object.fromEntries(
  MODELS.map((m) => [m.id, m] as const),
);

export type ModelId = (typeof MODELS)[number]["id"];
