"""Phase 0.7.1 -- build the two-halves-different-QF eval set (spatially-varying gate).

Each LIVE1 image: left half compressed at QF=10, right half at QF=90 (independent
JPEG compress per half, recombined losslessly). Tests spatially-adaptive
restoration: a global-conditioned model must treat the two halves differently
purely from local evidence. Saved as PNG (no further JPEG); at eval time the
conditioning is the DROPPED path (validity 0) since a mixed/re-saved image's DQT
is exactly the unreliable case the spec worries about.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from dejpeg.paths import testsets_dir, eval_sets_dir

SRC = testsets_dir("LIVE1_color")
DST = eval_sets_dir() / "twohalves"
DST.mkdir(parents=True, exist_ok=True)

QF_LEFT, QF_RIGHT = 10, 90


def compress(img_bgr: np.ndarray, qf: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    assert ok
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    n = 0
    for p in sorted(SRC.glob("*.bmp")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        mid = w // 2
        left = compress(img[:, :mid], QF_LEFT)
        right = compress(img[:, mid:], QF_RIGHT)
        deg = np.concatenate([left, right], axis=1)
        cv2.imwrite(str(DST / f"{p.stem}_2h.png"), deg)
        n += 1
    print(f"[twohalves] wrote {n} images -> {DST} (left QF{QF_LEFT} / right QF{QF_RIGHT})")


if __name__ == "__main__":
    main()
