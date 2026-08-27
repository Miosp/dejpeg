"""Recover P1 eval: load student_p1.pt (EMA), run Classic5 QF20/30 + contact sheet.
The trainer crashed in the contact sheet (missing no_grad); this re-runs the eval."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch, torch.nn.functional as F
from dejpeg_train.data.sources import DegradedBatchSource
from dejpeg_train.data.jpegmeta import parse_jpeg
from dejpeg_train.data.loader import sample_condition
from dejpeg_train.eval.metrics import psnr, psnr_b
from dejpeg_train.eval.panel import contact_sheet
from dejpeg_train.model.conditioning import quant_table_to_condition
from dejpeg_train.model.student import DeJPEGNetS, build_ctx
from dejpeg_train.paths import phase_dir, testsets_dir, shards_dir

DEV = "cuda"
OUT = phase_dir("phase06") / "p1"
CLASSIC5 = testsets_dir("Classic5")
SHARDS = [str(shards_dir() / "div2k-*.tar"),
          str(shards_dir() / "flickr2k-*.tar")]
FBCNN = {20: (32.01, 31.59), 30: (33.27, 32.70)}

model = DeJPEGNetS().to(DEV, memory_format=torch.channels_last)
ckpt = torch.load(OUT / "student_p1.pt", map_location=DEV, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
print("[eval] loaded student_p1.pt (EMA weights)", flush=True)

with torch.no_grad():
    # Classic5
    if CLASSIC5.exists():
        import cv2
        imgs = sorted(list(CLASSIC5.glob("*.bmp")) + list(CLASSIC5.glob("*.png")))
        for qf in (20, 30):
            psnrs, psnrbs = [], []
            for p in imgs:
                bgr = cv2.imread(str(p))
                if bgr is None: continue
                rgb = bgr[:, :, ::-1].copy()
                _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
                meta = parse_jpeg(enc.tobytes())
                cond = quant_table_to_condition(meta.quant_tables[0].values, 1.0).to(DEV)
                ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=DEV))
                inp = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
                _, _, h, w = inp.shape
                inp32 = F.pad(inp, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32)).contiguous(memory_format=torch.channels_last)
                o = model(inp32, ctx)[:, :, :h, :w].clamp(0, 1)
                o = (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                psnrs.append(psnr(rgb, o)); psnrbs.append(psnr_b(rgb, o))
            p_, pb = float(np.mean(psnrs)), float(np.mean(psnrbs))
            fb_p, fb_pb = FBCNN[qf]
            print(f"[eval] Classic5 QF{qf}: PSNR={p_:.2f} ({p_-fb_p:+.2f} vs FBCNN)  "
                  f"PSNR-B={pb:.2f} ({pb-fb_pb:+.2f} vs FBCNN)", flush=True)

    # contact sheet
    csrc = DegradedBatchSource(SHARDS, seed=7)
    rng = random.Random(7)
    pairs = []
    while len(pairs) < 8:
        s = csrc.draw(15, 35, rng)
        if not s["is_control"]:
            pairs.append(s)
    sheet = []
    for s in pairs:
        j = torch.from_numpy(s["jpeg"]).permute(2, 0, 1).unsqueeze(0).float().to(DEV).contiguous(memory_format=torch.channels_last)
        cond = sample_condition(s).to(DEV)
        ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=DEV))
        o = model(j, ctx).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        sheet.append((s["jpeg"] * 255).astype(np.uint8))
        sheet.append((o * 255).astype(np.uint8))
    contact_sheet(sheet, cols=2, thumb=320, path=str(OUT / "contact_sheet_p1.png"))
    print("[eval] contact sheet -> contact_sheet_p1.png", flush=True)
print("[eval] DONE", flush=True)
