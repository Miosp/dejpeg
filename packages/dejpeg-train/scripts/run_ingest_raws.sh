set -uo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv"
mkdir -p ${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/data/raw/user_raws
# verify rawpy present before launching
uv run python -c "import rawpy; print('[launch] rawpy', rawpy.__version__)" || { echo "[launch] rawpy MISSING - run uv sync"; exit 1; }
nohup uv run python scripts/ingest_user_raws.py > ${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/data/ingest_raws.log 2>&1 &
echo "[launch] ingest PID $! -> ${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/data/ingest_raws.log"
sleep 5
echo "--- first log lines ---"
tail -n 8 ${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/data/ingest_raws.log 2>/dev/null
