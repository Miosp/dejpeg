# web

Astro + Svelte 5 frontend for dejpeg.

## Dev

    bun run --filter web dev

## Build

    bun run --filter web build

Outputs to `dist/`.

## Deploy (Cloudflare Workers)

    cd apps/web
    bun run build
    bunx wrangler deploy

Serves static assets from `dist/` on a `*.workers.dev` URL
(`wrangler.jsonc`). COOP/COEP headers come from `public/_headers`, which
Workers Static Assets honors. The model files must exist in `public/models/`
at build time (see [Model files](#model-files) below).

Two Workers-specific size notes:

- The 25 MiB per-file asset limit. The ORT WASM binary (~26 MiB) exceeds
  it, so the build drops the wasm and `inference-core` fetches it from the
  jsdelivr CDN at runtime (`env.wasm.wasmPaths` in `engine/onnx.ts`).
  Keep that pinned version in sync with the lockfile.
- Keep bench artifacts out of `public/models/`; everything in `public/`
  deploys.

## Production model hosting

Not required: every model is under the 25 MiB per-file limit (about 15 MB
for each v2 file, 5.4 MB for v1), so they ship with the build. If a future
model exceeds the limit,
host it on object storage with permissive CORS (e.g. Cloudflare R2:
`wrangler r2 bucket create`, upload, enable public access, `wrangler r2
bucket cors put` with your site's origin) and point the model def's `url`
at the public bucket URL.

## Model files

The files are gitignored. Do not commit them. Download them from the GitHub
releases into `public/models/`:

    cd apps/web/public/models
    gh release download v2.0.0 -R Miosp/dejpeg -p 'dejpeg-v2*.onnx'
    gh release download v2.0.0 -R Miosp/dejpeg -p dejpeg-c40.onnx

## Custom domain

To map a custom domain to the Worker:

1. Cloudflare dashboard → Workers & Pages → dejpeg-web → Settings → Domains & Routes → Add → Custom domain
2. Enter the domain; Cloudflare auto-provisions SSL

DNS propagation typically takes a few minutes; SSL provisioning up to an hour.
