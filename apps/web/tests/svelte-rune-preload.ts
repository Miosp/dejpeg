// Polyfill Svelte 5 runes so `.svelte.ts` modules can be imported by Bun's
// test runner, which does not run the Svelte compiler. These are pass-through
// shims that preserve value semantics; runtime reactivity is not exercised by
// the unit tests in this package.
//
// Loaded via [test].preload in the repo-root bunfig.toml.

(globalThis as any).$state ??= <T>(value: T): T => value;
(globalThis as any).$derived ??= <T>(compute: () => T): T => compute();
const effectFn = ((_fn: () => unknown) => {}) as unknown as (() => void) & {
  root: (fn: () => unknown) => unknown;
};
(globalThis as any).$effect ??= effectFn;
(globalThis as any).$effect.root ??= (fn: () => unknown) => fn();
