#!/usr/bin/env bash
# Post-training orchestrator: waits for the 250k finish, runs the full final
# eval chain, then launches the C0=40 capacity arm (150k, early-annealed).
# Detached: setsid bash post_250k.sh &
set -u
LOG="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase2/train.log"
TRAIN="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/post250k.log"
echo "[orch] started $(date)" > "$TRAIN"

# 1. wait for trainer to exit AND it=250000 present in log
for i in $(seq 1 400); do   # 400 x 120s = 13.3h cap
  if ! pgrep -f "train_phase2_studen[t]" > /dev/null; then break; fi
  sleep 120
done
grep -q "it=250000" "$LOG" || { echo "[orch] FATAL: no it=250000 in log" >> "$TRAIN"; exit 1; }
sleep 30
echo "[orch] training done $(date)" >> "$TRAIN"

cd "$(dirname "$0")/.."
PY="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv/bin/python"
export MATRIX_CKPT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase2/student_p2_latest.pt"
export SHEET_CKPT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase2/student_p2_latest.pt"

echo "[orch] canonical matrix @250k $(date)" >> "$TRAIN"
$PY -u scripts/eval_perceptual_matrix.py >> "$TRAIN" 2>&1

echo "[orch] contact sheets $(date)" >> "$TRAIN"
$PY -u scripts/eval_contact_sheets.py >> "$TRAIN" 2>&1

echo "[orch] realweb NR input+student+fbcnn on cuda $(date)" >> "$TRAIN"
$PY -u scripts/eval_realweb_nr.py --variants all --device cuda >> "$TRAIN" 2>&1

# C0=40 capacity arm deliberately NOT auto-launched: an honest comparison needs
# a full annealed schedule; size ITERS to remaining wall-clock only after
# reviewing these evals. Launch manually:
#   OUT_DIR=${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase2_c040 ITERS=<fit-window> C0=40 setsid bash -c \
#     'cd <repo>/packages/dejpeg-train && export UV_PROJECT_ENVIRONMENT=${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv && \
#      uv run python -u scripts/train_phase2_student.py >> ${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase2_c040/train.log 2>&1'

echo "[orch] all evals done $(date)" >> "$TRAIN"
