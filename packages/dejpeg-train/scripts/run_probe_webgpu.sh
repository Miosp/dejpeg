#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv"
# Ensure chromium is installed for playwright (idempotent).
uv run playwright install chromium >/dev/null 2>&1 || true
uv run python scripts/probe_webgpu.py
