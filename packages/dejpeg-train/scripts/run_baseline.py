"""FBCNN baseline (Task 0.2, BLOCKING).

Mirrors FBCNN's main_test_fbcnn_color.py exactly: cv2.imencode('.jpg', qf)
on-the-fly compression (RGB->BGR->encode->decode->RGB, default 4:2:0), FBCNN
BLIND restore (no qf input -> model predicts), metrics PSNR / PSNR-B / SSIM /
LPIPS-Alex. Aggregates mean per (dataset, QF) and writes docs/research/fbcnn-
baseline.md. Every later gate is stated relative to these numbers.

Run: uv run python scripts/run_baseline.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]          # packages/dejpeg-train
FBCNN_PY = REPO.parent / "fbcnn-py" / "src"         # packages/fbcnn-py/src
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(FBCNN_PY))

from fbcnn.config import COLOR_REAL, build_model  # noqa: E402
from fbcnn.weights import load_pretrained  # noqa: E402

from dejpeg.eval.metrics import LPIPSAlex, compute_all  # noqa: E402
from dejpeg.paths import data_root, weights_dir, work_root  # noqa: E402

WEIGHTS_DIR = weights_dir()
TESTSETS = data_root() / "fbcnn-upstream" / "testsets"
DATASETS = {"Classic5": TESTSETS / "Classic5", "LIVE1": TESTSETS / "LIVE1_color"}
QFS = [10, 20, 30, 40]
DOC_OUT = Path(os.environ.get("FBCNN_BASELINE_DOC", str(work_root() / "docs" / "fbcnn-baseline.md")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def jpeg_compress(img_rgb: np.ndarray, qf: int) -> np.ndarray:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    t = t.detach().squeeze(0).clamp(0.0, 1.0)
    return (t.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def main() -> None:
    print(f"device={DEVICE}")
    net = build_model(COLOR_REAL).to(DEVICE).eval()
    load_pretrained(net, COLOR_REAL, WEIGHTS_DIR)
    lpips = LPIPSAlex(device=DEVICE)

    rows = []
    for ds_name, ds_dir in DATASETS.items():
        imgs = sorted([p for p in ds_dir.iterdir() if p.suffix.lower() in (".bmp", ".png", ".jpg")])
        print(f"\n{ds_name}: {len(imgs)} images")
        for qf in QFS:
            acc_in = {"psnr": [], "psnr_b": [], "ssim": [], "lpips_alex": []}
            acc_rest = {"psnr": [], "psnr_b": [], "ssim": [], "lpips_alex": []}
            used_qfs = []
            for p in imgs:
                gt = imread_rgb(p)
                jpeg = jpeg_compress(gt, qf)
                with torch.no_grad():
                    x = torch.from_numpy(jpeg).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
                    out_e, out_qf = net(x, None)
                    rest = to_uint8_rgb(out_e)
                    used_qfs.append(round((1.0 - float(out_qf.squeeze().item())) * 100.0))
                for acc, im in ((acc_in, jpeg), (acc_rest, rest)):
                    m = compute_all(gt, im, lpips_metric=lpips)
                    for k in acc:
                        acc[k].append(m[k])
            mean = lambda lst: float(np.mean(lst))
            def fmt(acc):
                return (
                    f"PSNR={mean(acc['psnr']):.2f} "
                    f"PSNR-B={mean(acc['psnr_b']):.2f} "
                    f"SSIM={mean(acc['ssim']):.4f} "
                    f"LPIPS-Alex={mean(acc['lpips_alex']):.4f}"
                )
            print(f"  QF{qf:2d}  IN  [{fmt(acc_in)}]")
            print(f"  QF{qf:2d}  FBCNN[{fmt(acc_rest)}]  pred_qf~{int(np.mean(used_qfs))}")
            rows.append(
                {
                    "ds": ds_name,
                    "qf": qf,
                    "in": {k: mean(v) for k, v in acc_in.items()},
                    "fbcnn": {k: mean(v) for k, v in acc_rest.items()},
                    "pred_qf": int(np.mean(used_qfs)),
                }
            )

    write_doc(rows)
    print(f"\nwrote {DOC_OUT}")


def write_doc(rows: list[dict]) -> None:
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FBCNN Baseline",
        "",
        "Reference numbers for every later gate (`>= FBCNN`).",
        "",
        "- Model: FBCNN color, single-JPEG (`fbcnn_color.pth`, **71.9M params**).",
        "- Protocol: cv2 JPEG compress at QF (4:2:0), FBCNN **blind** restore (QF predicted, not input).",
        "- Metrics: PSNR / PSNR-B / SSIM (FBCNN utils_image port) + LPIPS-Alex (eval-gate perceptual).",
        "- LPIPS-Alex lower is better; all others higher is better.",
        "- DISTS/MUSIQ/MANIQA/CLIPIQA DEFERRED (pyiqa broken; revisit for Real-web-500).",
        "",
        "## Classic5 + LIVE1",
        "",
        "| Dataset | QF | | PSNR | PSNR-B | SSIM | LPIPS-Alex | (input row in parens) |",
        "|---|---|---|---|---|---|---|",
    ]
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["ds"], []).append(r)
    for ds, rs in by_ds.items():
        for r in rs:
            i, f = r["in"], r["fbcnn"]
            lines.append(
                f"| {ds} | {r['qf']} | FBCNN | {f['psnr']:.2f} | {f['psnr_b']:.2f} "
                f"| {f['ssim']:.4f} | {f['lpips_alex']:.4f} | pred_qf~{r['pred_qf']} |"
            )
            lines.append(
                f"| {ds} | {r['qf']} | input | {i['psnr']:.2f} | {i['psnr_b']:.2f} "
                f"| {i['ssim']:.4f} | {i['lpips_alex']:.4f} | |"
            )
    lines += [
        "",
        "## Gates this baseline bounds",
        "",
        "- Phase 1.3 oracle: Classic5 QF{10,20,30,40} PSNR>=FBCNN AND PSNR-B>=FBCNN AND LPIPS-Alex+DISTS strictly better.",
        "- Phase 2.2 student: same Classic5 gates; Real-web-500 blockiness+MUSIQ+MANIQA+CLIPIQA all > FBCNN.",
        "",
        "_Generated by packages/dejpeg-train/scripts/run_baseline.py._",
    ]
    DOC_OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
