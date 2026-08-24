"""Phase 0.7.2 -- conditioning verdict on the saved P3 checkpoints (zero training).

Evaluates student_p3_{none,scalar,prompt}.pt on:
  A. Classic5 + LIVE1 @ QF {10,20,30,40}: PSNR / PSNR-B (+ input baseline)
  B. near-identity: 20 clean DIV2K-valid images, dropped conditioning (validity 0)
     -> PSNR(out, in); gate >= 45 dB (model must not touch clean input)
  C. QF sweep: 10 LIVE1 images, QF 5..95 step 5 -> per-bin PSNR gain
  D. two-halves (QF10 left / QF90 right), dropped conditioning
     -> per-half PSNR gain (spatially-adaptive restoration)

Writes $DEJPEG_WORK_ROOT/phase07/verdict.json and prints decision tables.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dejpeg.data.jpegmeta import parse_jpeg
from dejpeg.eval.dists import DISTS
from dejpeg.eval.metrics import psnr, psnr_b
from dejpeg.model.conditioning import quant_table_to_condition
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import phase_dir, testsets_dir, raw_dir, eval_sets_dir

DEV = "cuda"
CKPT_DIR = phase_dir("phase06") / "p3"
CLASSIC5 = testsets_dir("Classic5")
LIVE1 = testsets_dir("LIVE1_color")
CLEAN = raw_dir() / "div2k_valid/DIV2K_valid_HR"
TWOH = eval_sets_dir() / "twohalves"
OUT = phase_dir("phase07") / "verdict.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

_cond_dropped = quant_table_to_condition([0] * 64, validity=0.0).to(DEV)
_zeros32 = torch.zeros(1, 32, device=DEV)


def load_model(variant):
    m = DeJPEGNetS(cond_mode=variant).to(DEV, memory_format=torch.channels_last)
    ck = torch.load(CKPT_DIR / f"student_p3_{variant}.pt", map_location=DEV, weights_only=True)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def restore(model, rgb_u8, cond65=None):
    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
    _, _, h, w = t.shape
    t32 = F.pad(t, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))
    cond = _cond_dropped if cond65 is None else cond65
    ctx = build_ctx(cond.unsqueeze(0), _zeros32)
    o = model(t32.contiguous(memory_format=torch.channels_last), ctx)[:, :, :h, :w].clamp(0, 1)
    return (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def cond_from_jpeg(jpeg_bytes):
    meta = parse_jpeg(jpeg_bytes)
    qt = meta.quant_tables[0].values
    return quant_table_to_condition(qt, 1.0).to(DEV)


def read_rgb(p):
    bgr = cv2.imread(str(p))
    return bgr[:, :, ::-1].copy()


def jpeg_rgb(rgb, qf):
    bgr = rgb[:, :, ::-1].copy()
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    return dec, enc.tobytes()


# ------------------------------------------------------------------ eval A+C
def eval_classic(model, imgs, qfs):
    res = {}
    for qf in qfs:
        po, pb, pi = [], [], []
        for p in imgs:
            rgb = read_rgb(p)
            dec, jb = jpeg_rgb(rgb, qf)
            out = restore(model, dec, cond_from_jpeg(jb))
            po.append(psnr(rgb, out))
            pb.append(psnr_b(rgb, out))
            pi.append(psnr(rgb, dec))
        res[qf] = dict(psnr=float(np.mean(po)), psnrb=float(np.mean(pb)), psnr_in=float(np.mean(pi)))
    return res


def eval_sweep(model, imgs):
    gains, pins, pouts = {}, {}, {}
    for qf in range(5, 100, 5):
        po, pi = [], []
        for p in imgs:
            rgb = read_rgb(p)
            dec, jb = jpeg_rgb(rgb, qf)
            out = restore(model, dec, cond_from_jpeg(jb))
            po.append(psnr(rgb, out))
            pi.append(psnr(rgb, dec))
        pins[qf] = float(np.mean(pi))
        pouts[qf] = float(np.mean(po))
        gains[qf] = pouts[qf] - pins[qf]
    return dict(pin=pins, pout=pouts, gain=gains)


# ------------------------------------------------------------------ eval B
def eval_near_identity(model, imgs):
    vals = []
    for p in imgs:
        rgb = read_rgb(p)
        out = restore(model, rgb, None)  # dropped conditioning
        vals.append(psnr(rgb, out))
    return float(np.mean(vals)), float(np.min(vals))


# ------------------------------------------------------------------ eval D
def eval_twohalves(model):
    srcs = {p.stem.replace("_2h", ""): p for p in sorted(TWOH.glob("*_2h.png"))}
    gL, gR = [], []
    for stem, p in srcs.items():
        gt_p = LIVE1 / f"{stem}.bmp"
        if not gt_p.exists():
            continue
        deg = read_rgb(p)
        out = restore(model, deg, None)  # dropped conditioning (unreliable DQT)
        gt = read_rgb(gt_p)
        mid = gt.shape[1] // 2
        for half, bucket in ((0, gL), (1, gR)):
            gin = psnr(gt[:, mid * half:mid * (half + 1)], deg[:, mid * half:mid * (half + 1)])
            gout = psnr(gt[:, mid * half:mid * (half + 1)], out[:, mid * half:mid * (half + 1)])
            bucket.append(gout - gin)
    return dict(left_qf10_gain=float(np.mean(gL)), right_qf90_gain=float(np.mean(gR)),
                right_min_gain=float(np.min(gR)), n=len(gL))


def main():
    dists = DISTS().to(DEV)
    classic5 = sorted(CLASSIC5.glob("*.bmp"))
    live1 = sorted(LIVE1.glob("*.bmp"))
    sweep_imgs = live1[:10]
    clean_imgs = sorted(CLEAN.glob("*.png"))[:20]
    variants = ("none", "scalar", "prompt")
    results = {}
    for v in variants:
        model = load_model(v)
        r = results[v] = {}
        r["classic5"] = eval_classic(model, classic5, (10, 20, 30, 40))
        r["live1"] = eval_classic(model, live1, (10, 20, 30, 40))
        mean_all = np.mean([r[s][q]["psnr"] for s in ("classic5", "live1") for q in (10, 20, 30, 40)])
        r["mean_psnr_all"] = float(mean_all)
        ni_mean, ni_min = eval_near_identity(model, clean_imgs)
        r["near_identity"] = dict(psnr=ni_mean, worst=ni_min)
        r["sweep"] = eval_sweep(model, sweep_imgs)
        r["twohalves"] = eval_twohalves(model)
        del model
        torch.cuda.empty_cache()
        print(f"[{v}] meanPSNR={mean_all:.2f}  nearID={ni_mean:.1f}dB(worst {ni_min:.1f})  "
              f"2h L={r['twohalves']['left_qf10_gain']:+.2f} R={r['twohalves']['right_qf90_gain']:+.2f}",
              flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print("\n=== A. Classic5 / LIVE1 mean PSNR (restored) ===")
    print(f"{'variant':<8} {'C5@20':>7} {'L1@20':>7} {'C5@40':>7} {'L1@40':>7} {'ALL':>7}")
    for v in variants:
        r = results[v]
        print(f"{v:<8} {r['classic5'][20]['psnr']:>7.2f} {r['live1'][20]['psnr']:>7.2f} "
              f"{r['classic5'][40]['psnr']:>7.2f} {r['live1'][40]['psnr']:>7.2f} {r['mean_psnr_all']:>7.2f}")
    print("\n=== B. near-identity PSNR(out,in) dB [gate >= 45] ===")
    for v in variants:
        r = results[v]["near_identity"]
        print(f"{v:<8} mean={r['psnr']:.1f}  worst={r['worst']:.1f}  {'PASS' if r['worst'] >= 45 else 'FAIL'}")
    print("\n=== C. QF-sweep gain (dB over input): worst-bin / @QF5 / @QF95 ===")
    for v in variants:
        g = results[v]["sweep"]["gain"]
        worst = min(g, key=g.get)
        print(f"{v:<8} worst {worst}: {g[worst]:+.2f}   QF5 {g[5]:+.2f}   QF95 {g[95]:+.2f}")
    print("\n=== D. two-halves gain (dB): left QF10 / right QF90 [both should be > 0] ===")
    for v in variants:
        t = results[v]["twohalves"]
        print(f"{v:<8} L={t['left_qf10_gain']:+.2f}  R={t['right_qf90_gain']:+.2f} (worst {t['right_min_gain']:+.2f})  n={t['n']}")
    print(f"\n[verdict] json -> {OUT}")


if __name__ == "__main__":
    main()
