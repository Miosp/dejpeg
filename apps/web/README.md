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
Workers Static Assets honors. The model file (`dejpeg-c40.onnx`, ~5.4MB)
must exist in `public/models/` at build time (see
[Model file](#model-file) below).

Two Workers-specific size notes:

- The 25 MiB per-file asset limit. The ORT WASM binary (~26 MiB) exceeds
  it, so the build drops the wasm and `inference-core` fetches it from the
  jsdelivr CDN at runtime (`env.wasm.wasmPaths` in `engine/onnx.ts`).
  Keep that pinned version in sync with the lockfile.
- Keep bench artifacts out of `public/models/`; everything in `public/`
  deploys.

## Production model hosting

Not required: `dejpeg-c40.onnx` is ~5.4MB, under the 25 MiB per-file
limit, so it ships with the build. If a future model exceeds the limit,
host it on object storage with permissive CORS (e.g. Cloudflare R2:
`wrangler r2 bucket create`, upload, enable public access, `wrangler r2
bucket cors put` with your site's origin) and point the model def's `url`
at the public bucket URL.

## Model file

Copy `dejpeg-c40.onnx` into `public/models/` for local dev:

    cp <export-dir>/dejpeg-c40.onnx apps/web/public/models/

The file is gitignored. Do not commit it. Generate it with
`models/dejpeg/scripts/export_onnx.py --dynamic` if missing.

## Custom domain

To map a custom domain to the Worker:

1. Cloudflare dashboard → Workers & Pages → dejpeg-web → Settings → Domains & Routes → Add → Custom domain
2. Enter the domain; Cloudflare auto-provisions SSL

DNS propagation typically takes a few minutes; SSL provisioning up to an hour.
