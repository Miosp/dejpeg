#!/usr/bin/env bash
# One-time environment setup: create ext4 venv, sync deps (CUDA torch), check GPU/bf16, pytest collect.
# Run from Windows via:  wsl.exe -d Ubuntu bash <repo>/packages/dejpeg-train/scripts/setup_env.sh
# Optional env: DEJPEG_WORK_ROOT (outputs/venv, default ~/dejpeg-work), DEJPEG_DATA_ROOT (datasets).
set -euo pipefail

export DEJPEG_WORK_ROOT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}"
mkdir -p "$DEJPEG_WORK_ROOT"
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="$DEJPEG_WORK_ROOT/.venv"

echo "=== UV SYNC ==="
uv sync

echo "=== CUDA CHECK ==="
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('bf16', torch.cuda.is_bf16_supported()); print('torch', torch.__version__)"

echo "=== PYTEST COLLECT ==="
uv run pytest --collect-only -q && rc=0 || rc=$?
echo "PYTEST_EXIT=$rc"
echo "=== DONE ==="
