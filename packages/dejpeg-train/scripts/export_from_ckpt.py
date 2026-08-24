"""Export a saved capacity-probe checkpoint to FP16 ONNX on CPU (GPU stays free
for concurrent training). Env: CKPT, C0, OUT."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg.export.onnx import export_onnx, fuse_for_export
from dejpeg.model.student import DeJPEGNetS
from dejpeg.paths import phase_dir

CKPT = os.environ["CKPT"]
C0 = int(os.environ.get("C0", "40"))
OUT = os.environ.get("OUT", str(phase_dir("phase07")))

m = DeJPEGNetS(cond_mode="none", c0=C0)
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
m.load_state_dict(ck["model"])
m.eval()
fused = fuse_for_export(m)
n = sum(p.numel() for p in fused.parameters())
tag = Path(CKPT).stem
dst = Path(OUT) / f"{tag}_fp16.onnx"
sample = (torch.rand(1, 3, 256, 256), torch.rand(1, 97))
export_onnx(fused, sample, str(dst), opset=17, simplify=True, fp16=True)
print(f"[export] {tag}: params={n:,} fp16_bytes~{n*2/1e6:.2f}MB -> {dst}")
