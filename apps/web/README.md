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
3. **Models are NOT deployed via Pages** — they exceed the 25MB per-file limit.
   Models are hosted on Cloudflare R2. See
   [Production model hosting](#production-model-hosting) below.

## Production model hosting

FBCNN ONNX models (~137MB FP16 each) exceed Cloudflare Pages' 25MB per-file
limit. Host them on any object storage with permissive CORS (e.g. Cloudflare
R2: `wrangler r2 bucket create`, upload, enable public access, `wrangler r2
bucket cors put` with your site's origin).

The FBCNN model stubs in `packages/inference-core/src/models/fbcnn*.ts`
reference `https://pub-PLACEHOLDER.r2.dev/<name>.onnx`. Replace `PLACEHOLDER`
with the real public bucket ID after the setup.

## Model file

Copy `fbcnn-color-real.onnx` into `public/models/` for local dev:

    cp /tmp/convert-out/fbcnn-color-real.onnx apps/web/public/models/

The file is gitignored (too large). Do not commit it.

## Custom domain

To map a custom domain to the Pages deployment:

1. Cloudflare dashboard → Pages → your project → Custom domains → Set up a custom domain
2. Enter the domain; Cloudflare auto-provisions SSL
3. Add the new origin to your bucket's CORS rules and re-apply them

DNS propagation typically takes a few minutes; SSL provisioning up to an hour.
