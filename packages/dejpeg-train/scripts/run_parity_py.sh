set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv"
uv run python scripts/parity_phase05.py
