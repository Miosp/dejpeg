#!/usr/bin/env python
"""E1 metric panel: evaluate a restoration model over an eval-set manifest.

Metrics per image (uint8 in, uint8 out -- D8: quantize model output once):
  psnr / psnr_b / ssim (FBCNN-verbatim, dejpeg_train.eval.metrics)
  psnr_y / ssim_y      (luma channel via MATLAB rgb2ycbcr)
  lpips_alex           (eval-gate perceptual; patched normalize)
  topiq_fr / afine_fr / dists (pyiqa, full-reference)
  bef_in / bef_out     (blocking-effect factor, lower = less blocky)
  chroma_out           (chroma RMS, probes hue invention on gray input)
Sharpening control: same metrics for USM(input), amount 0.10 radius 1.0
(shipped default), so tables read relative to "just sharpen the input".

Model runners:
  --onnx PATH    onnxruntime session (deployed fp16 artifact), edge-pad to /32
  --weights PATH train-package student checkpoint (eval_p3_verdict.py contract:
                 DeJPEGNetS(cond_mode), DQT condition from the degraded JPEG
                 bytes, build_ctx(cond, zeros32))
  --sliding      torch mode: swap LSCA to sliding windows (A1)

Outputs: JSON (per-image rows + per-group aggregates + paired bootstrap CIs)
and a markdown table to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dejpeg_train.eval.metrics import LPIPSAlex, compute_bef, psnr, psnr_b, rgb2ycbcr, ssim  # noqa: E402


def unsharp_mask(x: np.ndarray, amount: float, radius: float = 1.0) -> np.ndarray:
    """Shipped USM on [0,1] float CHW numpy (matches models/dejpeg infer.py:93)."""
    import cv2

    if amount <= 0:
        return x
    hwc = np.moveaxis(x, 0, -1)
    blur = cv2.GaussianBlur(hwc, (0, 0), radius)
    return np.moveaxis(np.clip(hwc + amount * (hwc - blur), 0, 1), -1, 0)


def _pad_to_multiple(arr: np.ndarray, m: int = 32) -> tuple[np.ndarray, int, int]:
    h, w = arr.shape[:2]
    ph, pw = (h + m - 1) // m * m, (w + m - 1) // m * m
    out = np.pad(arr, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    return out, h, w


class OnnxRunner:
    def __init__(self, path: str):
        import onnxruntime as ort

        ort.preload_dlls()  # pip nvidia-* libs (cudnn etc.) for the CUDA EP
        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(path, providers=prov)

    def __call__(self, arr: np.ndarray, jpeg_bytes: bytes | None = None) -> np.ndarray:
        x, h, w = _pad_to_multiple(arr)
        t = x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        (y,) = self.sess.run(None, {self.sess.get_inputs()[0].name: t})
        y = np.clip(y[0], 0, 1).transpose(1, 2, 0)
        return y[:h, :w]

    def restore(self, arr: np.ndarray, jpeg_bytes: bytes | None = None, usm: float = 0.0) -> np.ndarray:
        """Run + optional shipped-default USM on the float output (deployed behavior)."""
        out = self(arr, jpeg_bytes)
        return unsharp_mask(out.transpose(2, 0, 1), usm).transpose(1, 2, 0) if usm > 0 else out


class TorchRunner:
    """Train-package student checkpoint runner (matches eval_p3_verdict.py)."""

    def __init__(self, path: str, sliding: bool, device: str, cond_mode: str | None = None, c0: int = 30):
        import torch

        from dejpeg_train.model.student import DeJPEGNetS

        self.torch = torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # legacy ckpts carry numpy globals
        if isinstance(ckpt, dict) and "model" in ckpt:
            cfg = {k: v for k, v in ckpt.items() if k not in {"model", "optimizer", "epoch"}}
            cond_mode = cond_mode or cfg.get("cond_mode", "prompt")
            sd = ckpt["model"]
        else:
            cond_mode = cond_mode or "prompt"
            sd = ckpt
            cfg = {}
        self.model = DeJPEGNetS(cond_mode=cond_mode, c0=c0 if c0 != 30 else int(cfg.get("c0", 30)))
        self.model.load_state_dict(sd)
        if sliding:
            from dejpeg_train.model.blocks import swap_lsca_sliding

            swap_lsca_sliding(self.model)
        self.model.eval().to(device, memory_format=torch.channels_last)
        self.device = device

    def __call__(self, arr: np.ndarray, jpeg_bytes: bytes | None = None) -> np.ndarray:
        from dejpeg_train.data.jpegmeta import parse_jpeg
        from dejpeg_train.model.conditioning import quant_table_to_condition
        from dejpeg_train.model.student import build_ctx

        torch = self.torch
        with torch.no_grad():
            x, h, w = _pad_to_multiple(arr, 32)
            t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device).float() / 255.0
            if jpeg_bytes is not None:
                qt = parse_jpeg(jpeg_bytes).quant_tables[0].values
                cond = quant_table_to_condition(qt, 1.0).to(self.device)
            else:
                cond = quant_table_to_condition([0] * 64, 0.0).to(self.device)
            ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=self.device))
            t = t.contiguous(memory_format=torch.channels_last)
            y = self.model(t, ctx)[:, :, :h, :w].clamp(0, 1)
            return y[0].permute(1, 2, 0).float().cpu().numpy()

    restore = OnnxRunner.restore


def to_u8(x: np.ndarray) -> np.ndarray:
    return (x * 255.0 + 0.5).astype(np.uint8)


def eval_image(gt: np.ndarray, pred: np.ndarray, deg: np.ndarray, iq: dict, lp: LPIPSAlex) -> dict:
    row = {
        "psnr": psnr(gt, pred), "psnr_b": psnr_b(gt, pred),
        "psnr_y": psnr(rgb2ycbcr(gt), rgb2ycbcr(pred)),
        "ssim": ssim(gt, pred),
        "ssim_y": ssim(rgb2ycbcr(gt), rgb2ycbcr(pred)),
        "bef_in": compute_bef(rgb2ycbcr(deg)), "bef_out": compute_bef(rgb2ycbcr(pred)),
        "lpips_alex": lp(pred, gt),
    }
    for name, fn in iq.items():
        row[name] = float(fn(pred, gt))
    ycc = rgb2ycbcr(pred, only_y=False).astype(np.float32)
    row["chroma_out"] = float(((ycc[..., 1:] - 128.0) ** 2).mean() ** 0.5)
    return row


def bootstrap_ci(deltas: list[float], n: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(deltas)
    if len(a) < 2:
        return [0.0, 0.0]
    idx = rng.integers(0, len(a), (n, len(a)))
    means = a[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--onnx")
    ap.add_argument("--weights")
    ap.add_argument("--sliding", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--c0", type=int, default=30, help="student base width (v15 shipped ckpt is 40)")
    ap.add_argument("--cond-mode", default=None, choices=["prompt", "scalar", "none"],
                    help="override ckpt cond mode (v15 shipped ckpt is none)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sets", default=None, help="comma filter on set field")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pyiqa", action="store_true")
    ap.add_argument("--usm", type=float, default=0.10)
    ap.add_argument("--model-usm", type=float, default=0.0,
                    help="USM applied to the MODEL OUTPUT (0.10 = shipped deployed behavior)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.sets:
        keep = set(args.sets.split(","))
        rows = [r for r in rows if r["set"] in keep]
    if args.limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for r in rows:
            if seen[r["set"]] < args.limit:
                kept.append(r)
                seen[r["set"]] += 1
        rows = kept
    print(f"{len(rows)} pairs from {args.manifest}", flush=True)

    iq: dict = {}
    if not args.no_pyiqa:
        import torch

        import pyiqa

        dev = args.device
        for name in ("topiq_fr", "afine_fr", "dists"):
            m = pyiqa.create_metric(name, device=dev)
            def _crop32(a, b):
                h = (min(a.shape[0], b.shape[0]) // 32) * 32
                w = (min(a.shape[1], b.shape[1]) // 32) * 32
                return a[:h, :w], b[:h, :w]

            iq[name] = lambda a, b, _m=m, _d=dev: float(_m(
                torch.from_numpy(_crop32(a, b)[0].transpose(2, 0, 1)[None].copy()).to(_d).float() / 255.0,
                torch.from_numpy(_crop32(a, b)[1].transpose(2, 0, 1)[None].copy()).to(_d).float() / 255.0,
            ).item())

    lp = LPIPSAlex(device=args.device)
    runner = OnnxRunner(args.onnx) if args.onnx else TorchRunner(
        args.weights, args.sliding, args.device, cond_mode=args.cond_mode, c0=args.c0)

    results, errors = [], 0
    for i, r in enumerate(rows):
        gt = np.asarray(Image.open(r["clean"]).convert("RGB"))
        deg = np.asarray(Image.open(r["degraded"]).convert("RGB"))
        jpeg_bytes = Path(r["degraded"]).read_bytes() if r["degraded"].lower().endswith((".jpg", ".jpeg")) else None
        try:
            pred = to_u8(runner.restore(deg, jpeg_bytes, args.model_usm))
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"[{i}] {r['degraded']}: {e}", file=sys.stderr, flush=True)
            continue
        usm_in = to_u8(unsharp_mask(deg.transpose(2, 0, 1) / 255.0, args.usm).transpose(1, 2, 0))
        res = eval_image(gt, pred, deg, iq, lp)
        res.update({"set": r["set"], "encoder": r.get("encoder", ""), "qf": r.get("qf", 0),
                    "tag": r.get("tag", ""), "name": r["degraded"],
                    "psnr_in": psnr(gt, deg), "psnr_b_in": psnr_b(gt, deg),
                    "psnr_in_usm": psnr(gt, usm_in)})
        for name, fn in iq.items():
            res[f"{name}_in"] = float(fn(deg, gt))
        results.append(res)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for res in results:
        groups[(res["set"], res["encoder"], res["qf"])].append(res)

    agg = {}
    for key, rs in sorted(groups.items()):
        keys = ["psnr", "psnr_y", "psnr_b", "ssim", "ssim_y", "lpips_alex", "bef_in", "bef_out",
                "chroma_out", "psnr_in", "psnr_b_in", "psnr_in_usm"] + list(iq.keys()) + \
               [f"{n}_in" for n in iq]
        m = {k: float(np.mean([r[k] for r in rs])) for k in keys if k in rs[0]}
        m["n"] = len(rs)
        m["ci_psnr_delta"] = bootstrap_ci([r["psnr"] - r["psnr_in"] for r in rs])
        if iq:
            first = next(iter(iq))
            m[f"ci_{first}_delta"] = bootstrap_ci([r[first] - r[f"{first}_in"] for r in rs])
        agg["|".join(str(k) for k in key)] = m

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": results, "agg": agg, "errors": errors,
                               "model": args.onnx or args.weights, "usm": args.usm,
                               "model_usm": args.model_usm}, indent=1))
    print(f"\nwrote {out} ({len(results)} rows, {errors} errors)\n")

    cols = ["n", "psnr", "psnr_in", "psnr_in_usm", "psnr_b", "ssim_y"] + list(iq)
    hdr = f"{'group':44s} " + " ".join(f"{c:>10s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for key, m in agg.items():
        print(f"{key:44s} " + " ".join(
            f"{m.get(c, float('nan')):10.3f}" if isinstance(m.get(c), float) else f"{m.get(c, ''):>10}"
            for c in cols))


if __name__ == "__main__":
    main()
