import { defineConfig } from "astro/config";
import * as fs from "node:fs";
import svelte from "@astrojs/svelte";
import { serwist } from "./src/service-worker/integration.ts";

// Vite middleware that sets COOP/COEP headers on ALL responses (including
// Astro-rendered HTML). Astro's own server middleware bypasses Vite's
// server.headers config, so we need this plugin to ensure the main page
// gets the headers required for SharedArrayBuffer / multi-threaded WASM.
function crossOriginIsolation() {
  const setHeaders = (res) => {
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  };
  return {
    name: "cross-origin-isolation",
    configureServer(server) {
      server.middlewares.use((_req, res, next) => { setHeaders(res); next(); });
    },
    configurePreviewServer(server) {
      server.middlewares.use((_req, res, next) => { setHeaders(res); next(); });
    },
  };
}

// The ORT WASM binary (~26 MiB) exceeds Cloudflare Workers' 25 MiB per-file
// asset limit; the engine fetches it from the CDN at runtime instead
// (env.wasm.wasmPaths in inference-core). Drop it from the build output so
// deploys are never rejected on file size. closeBundle runs after the worker
// sub-bundle has written its assets, which generateBundle never sees.
function dropOrtWasmAssets() {
  return {
    name: "drop-ort-wasm-assets",
    closeBundle() {
      const dir = "dist/_astro";
      if (!fs.existsSync(dir)) return;
      for (const f of fs.readdirSync(dir)) {
        if (/^ort-wasm.*\.wasm$/.test(f)) fs.unlinkSync(`${dir}/${f}`);
      }
    },
  };
}


export default defineConfig({
  output: "static",
  adapter: undefined,
  // The dev toolbar's audit panel re-runs its lints on every DOM update and
  // floods the console; astro check covers diagnostics in CI instead.
  devToolbar: { enabled: false },
  integrations: [svelte(), serwist()],
  vite: {
    plugins: [crossOriginIsolation(), dropOrtWasmAssets()],
    worker: { format: "es" },
    optimizeDeps: {
      exclude: ["onnxruntime-web"],
    },
  },
});
