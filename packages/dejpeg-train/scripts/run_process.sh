set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv"
LOG="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/data/process.log"
mkdir -p "${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/data"
nohup uv run python scripts/process_corpora.py > "$LOG" 2>&1 &
echo "PID $!"
echo "log: $LOG"
