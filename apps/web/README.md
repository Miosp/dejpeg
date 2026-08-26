# web

Astro + Svelte 5 frontend for dejpeg.

## Dev

    bun run --filter web dev

## Build

    bun run --filter web build

Outputs to `dist/`.

## Deploy (Cloudflare Pages)

1. Build locally: `bun run --filter web build`
2. Drag-and-drop `dist/` to Cloudflare Pages, or wire CI to run the build.
3. The model file (`dejpeg-c40.onnx`, ~5.4MB) is under the 25MB per-file
   limit and deploys with `dist/` — it must exist in `public/models/` at
   build time (see [Model file](#model-file) below).

## Production model hosting

Not required: `dejpeg-c40.onnx` is ~5.4MB, under Cloudflare Pages' 25MB
per-file limit, so it ships with the build. If a future model exceeds the
limit, host it on object storage with permissive CORS (e.g. Cloudflare R2:
`wrangler r2 bucket create`, upload, enable public access, `wrangler r2
bucket cors put` with your site's origin) and point the model def's `url`
at the public bucket URL.

## Model file

Copy `dejpeg-c40.onnx` into `public/models/` for local dev:

    cp <export-dir>/dejpeg-c40.onnx apps/web/public/models/

The file is gitignored. Do not commit it. Generate it with
`models/dejpeg/scripts/export_onnx.py --dynamic` if missing.

## Custom domain

To map a custom domain to the Pages deployment:

1. Cloudflare dashboard → Pages → your project → Custom domains → Set up a custom domain
2. Enter the domain; Cloudflare auto-provisions SSL
3. Add the new origin to your bucket's CORS rules and re-apply them

DNS propagation typically takes a few minutes; SSL provisioning up to an hour.
