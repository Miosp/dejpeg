// Minimal DOM harness for Bun's test runner, which ships without a real DOM.
// Installs localStorage (Map-backed), document.documentElement attribute
// storage, and window.matchMedia. Only installed when the global is absent so
// this never clobbers a real browser/JSDOM environment.
//
// Loaded via [test].preload in the repo-root bunfig.toml.

const g = globalThis as any;

// ---- localStorage (Map-backed) ----
if (typeof g.localStorage === "undefined") {
  const store = new Map<string, string>();
  g.localStorage = {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => { store.set(k, String(v)); },
    removeItem: (k: string) => { store.delete(k); },
    clear: () => { store.clear(); },
    key: (_i: number) => null,
    get length() { return store.size; },
  };
}

// ---- document.documentElement attribute storage ----
if (typeof g.document === "undefined") {
  const attrs = new Map<string, string>();
  g.document = {
    documentElement: {
      getAttribute: (name: string) => (attrs.has(name) ? attrs.get(name)! : null),
      setAttribute: (name: string, value: string) => { attrs.set(name, String(value)); },
      removeAttribute: (name: string) => { attrs.delete(name); },
    },
  };
}

// ---- window.matchMedia (prefers-color-scheme unsupported → false) ----
if (typeof g.window === "undefined") {
  g.window = {};
}
if (typeof g.window.matchMedia === "undefined") {
  g.window.matchMedia = (_q: string) => ({
    matches: false,
    media: _q,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  });
}
