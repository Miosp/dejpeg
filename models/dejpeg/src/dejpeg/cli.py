"""Command-line image restoration: ``dejpeg restore in.jpg out.png [--sharpness 0.1]``."""
from __future__ import annotations

import argparse

import torch

from .infer import load_model, restore_image


def main() -> None:
    p = argparse.ArgumentParser(prog="dejpeg", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("restore", help="restore one image or every image in a folder")
    r.add_argument("input", type=str)
    r.add_argument("output", type=str, help="output file (or folder for --glob)")
    r.add_argument("--weights", default=None,
                   help="path to a .pt checkpoint (default: cached release weights)")
    r.add_argument("--sharpness", type=float, default=0.10,
                   help="post-restoration unsharp amount; 0 disables")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from pathlib import Path

    model = load_model(args.weights, args.device)
    src = Path(args.input)
    if src.is_dir():
        dst = Path(args.output)
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                restore_image(f, dst / f"{f.stem}.png", model=model,
                              sharpness=args.sharpness, device=args.device)
                print(f"restored {f.name}")
    else:
        restore_image(src, args.output, model=model,
                      sharpness=args.sharpness, device=args.device)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
