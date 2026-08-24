/// <reference lib="webworker" />

import { Serwist, CacheFirst } from "serwist";
import type { PrecacheEntry, SerwistGlobalConfig } from "serwist";

declare global {
  interface WorkerGlobalScope extends SerwistGlobalConfig {
    __SW_MANIFEST: (PrecacheEntry | string)[] | undefined;
  }
}

declare const self: ServiceWorkerGlobalScope;

const serwist = new Serwist({
  precacheEntries: self.__SW_MANIFEST ?? [],
  skipWaiting: true,
  clientsClaim: true,
  navigationPreload: true,
  runtimeCaching: [
    {
      matcher: ({ url }) =>
        url.pathname.includes("ort-wasm") ||
        url.pathname.endsWith(".wasm") ||
        url.pathname.includes("ort."),
      handler: new CacheFirst({ cacheName: "dejpeg-wasm" }),
    },
  ],
});

serwist.addEventListeners();
