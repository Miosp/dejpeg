#!/usr/bin/env bash
# Post-Phase-2b eval chain: waits for fine-tune completion, runs chroma probe,
# near-identity gate, canonical matrix, realweb NR (all sequential, exclusive GPU).
# Detached: setsid bash scripts/post_2b.sh &
set -u
PLOG="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase2b/train.log"
OLOG="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/post2b.log"
PY="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/.venv/bin/python"
cd "$(dirname "$0")/.."
echo "[post2b] armed $(date)" > "$OLOG"

for i in $(seq 1 200); do  # 200 x 120s = 6.7h cap
  pgrep -f "train_phase2_studen[t]" > /dev/null || break
  sleep 120
done
grep -q "\[FINAL\]" "$PLOG" || { echo "[post2b] FATAL: no [FINAL] in phase2b log" >> "$OLOG"; exit 1; }
sleep 20
echo "[post2b] training complete $(date)" >> "$OLOG"

export GATE_CKPT="${DEJPEG_WORK_ROOT:-$HOME/dejpeg-work}/phase2b/student_p2_latest.pt"
export MATRIX_CKPT="$GATE_CKPT"

echo "[post2b] chroma probe $(date)" >> "$OLOG"
$PY - << 'EOF' >> "$OLOG" 2>&1
import sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, "src")
from dejpeg_train.paths import phase_dir, testsets_dir
from dejpeg_train.model.student import DeJPEGNetS

net = DeJPEGNetS(cond_mode="none").cuda()
ck = torch.load(phase_dir("phase2b")/"student_p2_latest.pt",
                map_location="cuda", weights_only=False)
net.load_state_dict(ck.get("ema", ck["model"])); net.eval()
gt = cv2.imread(str(testsets_dir("Classic5")/"1.bmp"))
ok, enc = cv2.imencode(".jpg", gt, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
inp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
x = torch.from_numpy(inp).permute(2,0,1)[None].float().cuda()/255.
with torch.no_grad():
    o = net(x.contiguous(memory_format=torch.channels_last), torch.zeros(1,97,device="cuda"))
out = (o[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
lab_o = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(float)
chroma = float(np.abs(lab_o[...,1:] - 128).mean())
psnr = 10*np.log10(255**2/np.mean((out.astype(float)-gt.astype(float))**2))
print(f"[probe] inventedChroma={chroma:.2f} PSNR={psnr:.2f} "
      f"({'OK' if chroma <= 1.5 else 'STILL DRIFTED'})")
EOF

echo "[post2b] near-identity gate $(date)" >> "$OLOG"
$PY -u scripts/eval_near_identity.py >> "$OLOG" 2>&1

echo "[post2b] canonical matrix @2b $(date)" >> "$OLOG"
$PY -u scripts/eval_perceptual_matrix.py >> "$OLOG" 2>&1

echo "[post2b] realweb NR all variants $(date)" >> "$OLOG"
$PY -u scripts/eval_realweb_nr.py --variants all --device cuda >> "$OLOG" 2>&1

echo "[post2b] all evals complete $(date)" >> "$OLOG"
