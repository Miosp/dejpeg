"""Inference-time quality levers on frozen weights, measured on the GT matrix.

Variants:
  base   - v15 single forward
  tta    - self-ensemble: x8 geometric (x4 flips when non-square)
  ens2   - average(v15, phase2d)
  tta_ens2 - ensemble of TTA'd outputs
  usm    - base + mild unsharp mask
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import lpips  # noqa: E402

from dejpeg_train.eval.dists import DISTS  # noqa: E402
from dejpeg_train.model.student import DeJPEGNetS  # noqa: E402
from dejpeg_train.paths import phase_dir, testsets_dir

DEV = "cuda"
V15 = Path(os.environ.get("V15_CKPT", phase_dir("phase_v15") / "student_v15_final.pt"))
LITE = Path(os.environ.get("LITE_CKPT", phase_dir("phase2d") / "student_p2_final.pt"))
SETS = {
    "classic5": testsets_dir("Classic5"),
    "live1": testsets_dir("LIVE1_color"),
}
OUT = phase_dir("phase07") / "v1_variants.json"

lalex = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
dists_m = DISTS().to(DEV).eval()


def load(path):
    m = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "40"))).to(DEV)
    c0_env = int(os.environ.get("C0", "40"))
    ck = torch.load(path, map_location=DEV, weights_only=False)
    sd = ck.get("ema", ck["model"])
    if next(iter(sd.values())).shape[0] == 30 or sd["shallow.weight"].shape[0] == 30:
        m = DeJPEGNetS(cond_mode="none", c0=30).to(DEV)
    m.load_state_dict(sd)
    m.eval()
    return m


def fwd(m, bgr_u8):
    x = torch.from_numpy(bgr_u8[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(DEV) / 255.0
    _, _, h, w = x.shape
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        o = m(xp.contiguous(memory_format=torch.channels_last),
              torch.zeros(1, 97, device=DEV))[:, :, :h, :w]
    return o.clamp(0, 1)


def tta(m, bgr_u8):
    """8-fold dihedral self-ensemble with explicit inverses (rot90-based).
    Falls back to 4 fold-flips when non-square."""
    x0 = torch.from_numpy(bgr_u8[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(DEV) / 255.0
    _, _, h, w = x0.shape
    pairs = []
    ks = (0, 1, 2, 3) if h == w else (0,)
    for k in ks:
        for fl in (False, True):
            def fwd_t(t, k=k, fl=fl):
                y = torch.rot90(t, k, dims=(-2, -1))
                return torch.flip(y, [-1]) if fl else y

            def inv_t(y, k=k, fl=fl):
                z = torch.flip(y, [-1]) if fl else y
                return torch.rot90(z, -k, dims=(-2, -1))

            pairs.append((fwd_t, inv_t))
    acc = torch.zeros_like(x0)
    with torch.no_grad():
        for f, finv in pairs:
            t = f(x0)
            _, _, th, tw = t.shape
            tp = F.pad(t, (0, (32 - tw % 32) % 32, 0, (32 - th % 32) % 32), mode="reflect")
            o = m(tp.contiguous(memory_format=torch.channels_last),
                  torch.zeros(1, 97, device=DEV))[:, :, :th, :tw]
            acc += finv(o)
    return (acc / len(pairs)).clamp(0, 1)


def usm(o_t, amount=0.15):
    img = o_t[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharp = np.clip(img + amount * (img - blur), 0, 1)
    return torch.from_numpy(sharp[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(DEV)


def scores(o_t, rgb):
    gt = torch.from_numpy(rgb[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(DEV) / 255.0
    d = float(dists_m(o_t, gt))
    l = float(lalex(o_t * 2 - 1, gt * 2 - 1))
    mse = float(((o_t - gt) ** 2).mean())
    p = 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)
    return d, l, p


def main():
    v15 = load(V15)
    lite = None
    if LITE.exists():
        lite = DeJPEGNetS(cond_mode="none", c0=30).to(DEV)
        ck = torch.load(LITE, map_location=DEV, weights_only=False)
        lite.load_state_dict(ck.get("ema", ck["model"]))
        lite.eval()

    results = {}
    for set_name, root in SETS.items():
        imgs = sorted(root.glob("*.bmp")) + sorted(root.glob("*.png"))
        rows = []
        for qf in (10, 20, 30, 40):
            agg = {}
            for p in imgs:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                rgb = bgr[:, :, ::-1].copy()  # pristine GT (pre-encode)
                b_base = fwd(v15, dec)
                b_tta = tta(v15, dec)
                outs = {"base": b_base, "tta": b_tta, "usm": usm(b_base)}
                if lite is not None:
                    l_out = fwd(lite, dec)
                    outs["ens2"] = ((b_base + l_out) / 2).clamp(0, 1)
                    outs["tta_ens2"] = ((b_tta + tta(lite, dec)) / 2).clamp(0, 1)
                for k, o in outs.items():
                    d, l, ps = scores(o, rgb)
                    agg.setdefault(k, []).append((d, l, ps))
            row = {k: tuple(round(float(np.mean([x[i] for x in v])), 4) for i in range(3))
                   for k, v in agg.items()}
            rows.append((qf, row))
            print(f"[{set_name} QF{qf}] " +
                  " | ".join(f"{k}: D={row[k][0]:.4f} L={row[k][1]:.4f} P={row[k][2]:.2f}"
                             for k in ("base", "tta", "ens2", "tta_ens2", "usm") if k in row),
                  flush=True)
        results[set_name] = {str(q): r for q, r in rows}

    import json
    OUT.write_text(json.dumps(results, indent=1))
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
