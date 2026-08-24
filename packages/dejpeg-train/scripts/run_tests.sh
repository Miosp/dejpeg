#!/usr/bin/env bash
# Reusable Phase-0 test runner. Usage: bash run_tests.sh [pytest target]
#   default target = "tests"
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv"
TEST="${1:-tests}"
uv run pytest "$TEST" -v
