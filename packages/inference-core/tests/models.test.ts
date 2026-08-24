import { test, expect } from "bun:test";
import { MODELS, MODELS_BY_ID } from "../src/models/index.js";

test("every model has a unique id", () => {
  const ids = MODELS.map((m) => m.id);
  expect(new Set(ids).size).toBe(ids.length);
});

test("MODELS_BY_ID contains every model", () => {
  for (const m of MODELS) {
    expect(MODELS_BY_ID[m.id]).toBe(m);
  }
});

test("every input binding resolves to 'image' or a declared param", () => {
  for (const m of MODELS) {
    for (const [name, binding] of Object.entries(m.inputs)) {
      if (binding === "image") continue;
      expect(
        binding.param in m.params,
        `model ${m.id}: input "${name}" binds to unknown param "${binding.param}"`,
      ).toBe(true);
    }
  }
});

test("every model has at least one input bound to 'image'", () => {
  for (const m of MODELS) {
    const hasImage = Object.values(m.inputs).some((b) => b === "image");
    expect(hasImage, `model ${m.id} has no 'image' input`).toBe(true);
  }
});

test("every model declares at least one output", () => {
  for (const m of MODELS) {
    expect(m.outputs.length).toBeGreaterThan(0);
  }
});
