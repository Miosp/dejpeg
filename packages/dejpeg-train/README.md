# dejpeg-train

Research training pipeline for **DeJPEGNet**, the browser-deployable JPEG
artifact remover. NAFNet-derived, blind to QF, full RGB, tile-invariant,
FP16 ≤4MB for ONNX Runtime Web (WebGPU). The shipped model lives in
`models/dejpeg/`; this package contains the full corpus-building and
experiment tooling behind it.

## Layout

```
src/dejpeg_train/
  model/   blocks, prompt, student, degencoder, teacher, reparam
  data/    prepare, manifest, synthetic, sources, degrade, jpegmeta, controls, batcher
  loss/    perceptual, blockiness, ldl, contrastive, gan, distill
  train/   degencoder, oracle, teacher, student, ablate, finetune, schedule
  export/  onnx, verify
  eval/    metrics, sweep, panel
  bench/   latency
  paths    env-driven path configuration (see below)
configs/   per-phase yaml
scripts/   runnable experiments (corpus build, training phases, evals)
tests/     deterministic unit tests
```

## Path configuration

Nothing is machine-specific: every script resolves locations via
`src/dejpeg_train/paths.py` from two environment variables.

| Variable | Meaning | Default |
|---|---|---|
| `DEJPEG_WORK_ROOT` | outputs: venv, checkpoints, logs, phase dirs | `~/dejpeg-work` |
| `DEJPEG_DATA_ROOT` | datasets: shards, raw, manifest, eval sets, weights | `$DEJPEG_WORK_ROOT/data` |

Expected layout under `DEJPEG_DATA_ROOT`:

```
shards/            WebDataset tars: {div2k,flickr2k,liu4k_v2,user_raws}-NNNNNN.tar
raw/               source images (div2k, user_raws, user_raws_src, ...)
manifest.sqlite    global content+perceptual-hash dedup store
eval_sets/         twohalves/, realweb500/
weights/           FBCNN reference weights (fbcnn_color.pth), DISTS metric weights
fbcnn-upstream/    cloned FBCNN repo (testsets/Classic5, testsets/LIVE1_color)
```

## Environment

Standalone uv project (not a workspace member: uses CUDA torch; the root
workspace pins CPU torch for fbcnn). Code lives in the repo; the venv,
datasets, and run outputs should live on a native Linux filesystem (ext4),
never on drvfs mounts like `/mnt/c`, which starve the dataloader.

```
# from WSL2/Linux:
export DEJPEG_WORK_ROOT=/path/to/ext4/work      # venv + run outputs
export DEJPEG_DATA_ROOT=/path/to/datasets       # shards etc (big disk)
bash scripts/setup_env.sh                       # venv + uv sync + CUDA check
```

Corpus build: `scripts/fetch_datasets.py` (DIV2K/Flickr2K/LIU4K) →
`scripts/process_corpora.py` / `prepare_div2k.py` → `shards/`.
Personal photo ingest (RAW → lossless WebP): `scripts/ingest_user_raws.py`.

## Tests

```
UV_PROJECT_ENVIRONMENT=$DEJPEG_WORK_ROOT/.venv uv run pytest
```
