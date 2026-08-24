"""Loading original FBCNN pretrained weights into the ported network.

Strict loading is the contract: zero missing keys, zero unexpected keys.
If we ever rename a module attribute, KEY_MAP is the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fbcnn.config import Variant
from fbcnn.network import FBCNN

__all__ = ["load_original_state_dict", "load_pretrained", "KEY_MAP"]

# Empty by default — our port preserves upstream module attribute names.
# If a future refactor renames anything, map {old_key: new_key} here.
KEY_MAP: dict[str, str] = {}


def load_original_state_dict(net: FBCNN, pth_path: Path) -> None:
    """Load an upstream FBCNN .pth into the ported network. Strict.

    Raises:
        RuntimeError: if torch.load fails or any keys are missing/unexpected.
    """
    pth_path = Path(pth_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"Weight file not found: {pth_path}")

    # Upstream .pth files contain pickled non-tensor objects; we trust these
    # specific vendored files, so opt out of torch>=2.6 weights_only default.
    raw = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected weight format in {pth_path}: {type(raw).__name__}")

    mapped = {KEY_MAP.get(k, k): v for k, v in raw.items()}
    net.load_state_dict(mapped, strict=True)


def load_pretrained(net: FBCNN, variant: Variant, weights_dir: Path) -> None:
    """Convenience: load_variant by ID using a directory of .pth files."""
    weights_dir = Path(weights_dir)
    pth_path = weights_dir / variant.weight_filename
    load_original_state_dict(net, pth_path)
