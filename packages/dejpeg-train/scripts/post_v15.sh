#!/usr/bin/env bash
# V1.5 post-run chain: wait for completion -> gates -> canonical matrix ->
# realweb NR. Same shape as post_2b.sh but targets phase_v15.
set -u
PLOG="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase_v15/train.log"
OLOG="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/postv15.log"
PY="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv/bin/python"
cd "$(dirname "$0")/.."
echo "[postv15] armed $(date)" > "$OLOG"

for i in $(seq 1 300); do  # 300 x 120s = 10h cap beyond expected finish
  pgrep -f "train_phase2_studen[t]" > /dev/null || break
  sleep 120
done
grep -q "\[FINAL\]" "$PLOG" || { echo "[postv15] FATAL: no [FINAL] in log" >> "$OLOG"; exit 1; }
sleep 20
echo "[postv15] training complete $(date)" >> "$OLOG"

export PROBE_CKPT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase_v15/student_p2_latest.pt"
export GATE_CKPT="$PROBE_CKPT"
export MATRIX_CKPT="$PROBE_CKPT"
export NR_CKPT="$PROBE_CKPT"

echo "[postv15] chroma probe $(date)" >> "$OLOG"
$PY -u scripts/eval_chroma_probe.py >> "$OLOG" 2>&1

echo "[postv15] near-identity gate $(date)" >> "$OLOG"
$PY -u scripts/eval_near_identity.py >> "$OLOG" 2>&1

echo "[postv15] canonical matrix $(date)" >> "$OLOG"
$PY -u scripts/eval_perceptual_matrix.py >> "$OLOG" 2>&1

echo "[postv15] realweb NR all variants $(date)" >> "$OLOG"
$PY -u scripts/eval_realweb_nr.py --variants all --device cuda >> "$OLOG" 2>&1

echo "[postv15] complete $(date)" >> "$OLOG"
