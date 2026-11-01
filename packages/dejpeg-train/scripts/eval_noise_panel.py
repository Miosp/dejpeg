"""V0 noise panel (SUPPLEMENT 2026-10-15, sections 7.1/8.1).

Measures the output fine-grain artifact of a DeJPEGNet candidate on LIVE1
probe crops and reproduces the study baseline for v15:
    notch_share   ~0.040  (fraction of luma error power at the pattern bins)
    flat_hf_ratio ~0.80   (output/GT high-pass RMS on GT-flat pixels)

Protocol (SUPPLEMENT V0): GT = LIVE1 BMP cropped to a multiple of 16, then a
384x384 crop at row (H-384)//16*8, col (W-384)//16*8; PIL JPEG quality=QF
subsampling=2 (4:2:0) of the crop; model; round to uint8; compare luma.

Usage (WSL, from packages/dejpeg-train):
  python scripts/eval_noise_panel.py --weights /path/student_v15_final.pt \
      --c0 40 --out /path/noise-panel.json [--pieapp] [--qf 10,30,50]
  python scripts/eval_noise_panel.py --onnx apps/web/public/models/dejpeg-c40.onnx ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dejpeg_train.eval.metrics import psnr, rgb2ycbcr, ssim  # noqa: E402
from dejpeg_train.paths import testsets_dir  # noqa: E402

PEAK_FREQS = ((0.25, 0.5), (0.5, 0.25), (-0.25, 0.5), (0.5, -0.25))  # (fy, fx) cycles/px
NOTCH_RADIUS = 0.006  # cycles/px; ~2.3 bins on a 384-px crop
FLAT_STD = 2.0        # grey levels, 7x7 window (probe definition)

PROBE_IMAGES = ("bikes", "building2", "caps", "lighthouse2", "monarch",
                "ocean", "parrots", "sailing2", "statue", "woman")


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def peak_mask(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    mask = np.zeros((h, w), bool)
    for py, px in PEAK_FREQS:
        for sy, sx in ((py, px), (-py, -px)):
            dy = (fy - sy + 0.5) % 1 - 0.5
            dx = (fx - sx + 0.5) % 1 - 0.5
            mask |= dy ** 2 + dx ** 2 <= NOTCH_RADIUS ** 2
    return mask


def notch_share(out_y: np.ndarray, gt_y: np.ndarray) -> float:
    spec = np.abs(np.fft.fft2(out_y - gt_y)) ** 2
    return float(spec[peak_mask(*spec.shape)].sum() / spec.sum())


def binomial_highpass(y: np.ndarray) -> np.ndarray:
    k = np.array([1.0, 2.0, 1.0]) / 4
    lo = convolve1d(convolve1d(y, k, axis=0, mode="mirror"), k, axis=1, mode="mirror")
    return y - lo


def flat_hf_ratio(out_y: np.ndarray, gt_y: np.ndarray) -> float:
    m = uniform_filter(gt_y, 7)
    flat = np.sqrt(np.maximum(uniform_filter(gt_y ** 2, 7) - m ** 2, 0)) < FLAT_STD
    ho, hg = binomial_highpass(out_y), binomial_highpass(gt_y)
    return float(np.sqrt((ho[flat] ** 2).mean() / (hg[flat] ** 2).mean()))


def peak_ratio(err_y: np.ndarray, size: int = 256) -> float:
    acc = np.zeros((size, size))
    win = np.outer(np.hanning(size), np.hanning(size))
    for r in range(0, err_y.shape[0] - size + 1, size):
        for c in range(0, err_y.shape[1] - size + 1, size):
            t = err_y[r:r + size, c:c + size]
            acc += np.abs(np.fft.fft2((t - t.mean()) * win)) ** 2
    bins = [(int(round(py * size)) % size, int(round(px * size)) % size) for py, px in PEAK_FREQS]
    return float(np.mean([acc[b] for b in bins]) / np.median(acc))


def probe_crop(img: Image.Image, size: int = 384) -> Image.Image:
    h, w = img.height, img.width
    img = img.crop((0, 0, w // 16 * 16, h // 16 * 16))
    r = (img.height - size) // 16 * 8
    c = (img.width - size) // 16 * 8
    return img.crop((c, r, c + size, r + size))


def make_runner(args):
    if args.onnx:
        import onnxruntime as ort

        ort.preload_dlls()
        sess = ort.InferenceSession(str(args.onnx), providers=[args.provider, "CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def run(arr: np.ndarray) -> np.ndarray:
            h, w = arr.shape[:2]
            ph, pw = (-h) % 32, (-w) % 32
            if ph or pw:
                a = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
            else:
                a = arr
            x = a.transpose(2, 0, 1)[None].astype(np.float32)
            out = sess.run(None, {name: x})[0][0].transpose(1, 2, 0)
            return np.clip(out[:h, :w], 0, 1)
    else:
        import torch

        from dejpeg_train.model.student import DeJPEGNetS

        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        model = DeJPEGNetS(cond_mode=args.cond_mode, c0=args.c0)
        model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
        model.eval().to(args.device)

        def run(arr: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(arr.transpose(2, 0, 1)[None]).to(args.device).float()
            h, w = arr.shape[:2]
            ph, pw = (-h) % 32, (-w) % 32
            if ph or pw:
                x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.no_grad():
                out = model(x, torch.zeros(1, 97, device=args.device))[0]
            return np.clip(out[:, :h, :w].cpu().numpy().transpose(1, 2, 0), 0, 1)
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--c0", type=int, default=40)
    ap.add_argument("--cond-mode", default="none")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    ap.add_argument("--qf", default="10,30,50")
    ap.add_argument("--images", choices=["probe", "all"], default="probe")
    ap.add_argument("--pieapp", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.weights or args.onnx, "need --weights or --onnx"

    live1 = sorted(testsets_dir("LIVE1_color").glob("*.bmp"))
    names = PROBE_IMAGES if args.images == "probe" else [p.stem for p in live1]
    files = {p.stem: p for p in live1}

    run = make_runner(args)

    pieapp = None
    if args.pieapp:
        import torch

        import pyiqa

        pieapp = pyiqa.create_metric("pieapp", device=args.device)

    rows = []
    for name in names:
        gt_img = probe_crop(Image.open(files[name]).convert("RGB"))
        gt = np.asarray(gt_img, dtype=np.float64)  # 0..255
        for qf in [int(q) for q in args.qf.split(",")]:
            buf = Path("/tmp") / f"probe_{name}_q{qf}.jpg"
            gt_img.save(buf, quality=qf, subsampling=2)
            jpeg = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)
            out = (run((jpeg / 255.0).astype(np.float32)) * 255.0 + 0.5).astype(np.uint8).astype(np.float64)
            gt_l, jp_l, out_l = luma(gt), luma(jpeg), luma(out)
            row = {
                "image": name, "qf": qf,
                "psnr_y": psnr(gt_l, out_l),
                "psnr_y_in": psnr(gt_l, jp_l),
                "ssim_y": ssim(gt_l, out_l),
                "ssim_y_in": ssim(gt_l, jp_l),
                "notch_share": notch_share(out_l, gt_l),
                "flat_hf_ratio": flat_hf_ratio(out_l, gt_l),
                "peak_ratio": peak_ratio(out_l - gt_l),
            }
            if pieapp is not None:
                import torch

                def t(a: np.ndarray) -> torch.Tensor:
                    return torch.from_numpy(a.transpose(2, 0, 1)[None] / 255.0).float().to(args.device)

                row["pieapp"] = float(pieapp(t(out), t(gt)))
                row["pieapp_in"] = float(pieapp(t(jpeg), t(gt)))
            rows.append(row)
            print(f"{name} qf{qf}: psnr_y={row['psnr_y']:.3f} notch={row['notch_share']:.4f} "
                  f"flat_hf={row['flat_hf_ratio']:.3f} peak={row['peak_ratio']:.1f} "
                  f"pieapp={row.get('pieapp', float('nan')):.3f}", flush=True)
    summary = {}
    for qf in sorted({r["qf"] for r in rows}):
        sub = [r for r in rows if r["qf"] == qf]
        summary[qf] = {f"mean_{k}": float(np.mean([r[k] for r in sub]))
                       for k in ("psnr_y", "notch_share", "flat_hf_ratio", "peak_ratio")}

    args.out.write_text(json.dumps({"rows": rows, "summary": summary,
                                    "weights": str(args.weights), "onnx": str(args.onnx)}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
