"""Modern PyTorch port of FBCNN."""

from fbcnn.blocks import (
    QFAttention,
    ResBlock,
    conv,
    downsample_strideconv,
    sequential,
    upsample_convtranspose,
)
from fbcnn.config import (
    COLOR_REAL,
    GRAY,
    GRAY_DOUBLE,
    VARIANTS,
    VARIANTS_BY_ID,
    Variant,
    build_model,
)
from fbcnn.inference import run
from fbcnn.network import FBCNN
from fbcnn.weights import KEY_MAP, load_original_state_dict, load_pretrained

__all__ = [
    "COLOR_REAL",
    "FBCNN",
    "GRAY",
    "GRAY_DOUBLE",
    "KEY_MAP",
    "QFAttention",
    "ResBlock",
    "VARIANTS",
    "VARIANTS_BY_ID",
    "Variant",
    "build_model",
    "conv",
    "downsample_strideconv",
    "load_original_state_dict",
    "load_pretrained",
    "run",
    "sequential",
    "upsample_convtranspose",
]
