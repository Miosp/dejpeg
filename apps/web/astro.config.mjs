import { defineConfig } from "astro/config";
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

export default defineConfig({
  output: "static",
  adapter: undefined,
  integrations: [svelte(), serwist()],
  vite: {
    plugins: [crossOriginIsolation()],
    worker: { format: "es" },
    optimizeDeps: {
      exclude: ["onnxruntime-web"],
    },
    build: {
      rollupOptions: {
        external: ["onnxruntime-web"],
      },
    },
  },
});
