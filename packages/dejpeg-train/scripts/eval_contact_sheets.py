"""Side-by-side contact sheets: input | FBCNN | ours, for eyeball checks.

Writes PNG strips to $DEJPEG_WORK_ROOT/phase07/sheets_<tag>/.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dejpeg.paths import phase_dir, testsets_dir, eval_sets_dir, weights_dir

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "fbcnn-py" / "src"))

CKPT = Path(os.environ.get(
    "SHEET_CKPT", phase_dir("phase2") / "student_p2_latest.pt"))
OUT = Path(os.environ.get(
    "SHEET_OUT", phase_dir("phase07") / "sheets_final"))
LIVE1 = testsets_dir("LIVE1_color")
REALWEB = eval_sets_dir() / "realweb500"
WEIGHTS_DIR = weights_dir()


def load_student(device="cuda"):
    import torch
    from dejpeg.model.student import DeJPEGNetS

    m = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "30"))).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    m.load_state_dict(ck.get("ema", ck["model"]))
    m.eval()
    return m


def restore(model, bgr):
    import torch
    x = torch.from_numpy(bgr).permute(2, 0, 1)[None].float().cuda() / 255.0
    _, _, h, w = x.shape
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        o = model(xp.contiguous(memory_format=torch.channels_last),
                  torch.zeros(1, 97, device="cuda"))[:, :, :h, :w]
    return (o[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[:, :, ::-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import fbcnn.config as fb_cfg
    import fbcnn.inference as fb_inf
    model = load_student()

    jobs = []
    for i, p in enumerate(sorted(LIVE1.glob("*.png"))[:4]):
        for qf in (10, 30):
            jobs.append((f"live1_{p.stem}_q{qf}", p, qf, False))
    for i, p in enumerate(sorted(REALWEB.glob("rw_*.jpg"))[::100][:6]):
        jobs.append((f"web_{p.stem}", p, None, True))

    for name, path, qf, is_web in jobs:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        if max(bgr.shape[:2]) > 1024:
            s = 1024 / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        if is_web:
            inp = bgr
        else:
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
            inp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        fb = fb_inf.run(inp, None, fb_cfg.COLOR_REAL, str(WEIGHTS_DIR))[0]
        st = restore(model, inp)
        strip = np.concatenate([inp, fb, st], axis=1)
        cv2.imwrite(str(OUT / f"{name}.png"), strip)
        print(f"[sheet] {name} in={inp.shape[1]}x{inp.shape[0]}", flush=True)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
