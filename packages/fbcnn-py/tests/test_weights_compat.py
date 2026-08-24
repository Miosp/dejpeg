"""Original weight compatibility — the port's strictness contract.

Skipped unless FBCNN_WEIGHTS_DIR points at a directory containing the original
.pth files from https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fbcnn import VARIANTS, build_model
from fbcnn.weights import load_pretrained

from conftest import skip_without_weights


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.id for v in VARIANTS])
def test_load_original_weights_strict(variant, weights_dir: Path | None):
    skip_without_weights(weights_dir)
    net = build_model(variant)
    # Must not raise — strict=True is enforced inside load_pretrained.
    load_pretrained(net, variant, weights_dir)
