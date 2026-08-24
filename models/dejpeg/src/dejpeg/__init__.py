"""DeJPEGNet -- compact perceptual JPEG artifact removal.

    from dejpeg import load_model, restore_image
    model = load_model()
    restore_image("input.jpg", "output.png", model)
"""
from .infer import DEFAULT_SHARPNESS, load_model, restore_array, restore_image
from .model import DeJPEGNet

__version__ = "1.0.0"
__all__ = ["DeJPEGNet", "DEFAULT_SHARPNESS", "load_model", "restore_array", "restore_image"]
