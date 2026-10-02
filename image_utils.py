"""Legacy conversion adapter; strict new engines use core.tensors directly."""
from .core.tensors import legacy_image_tensor


def enforce_image_format(image, force_rgb: bool = False):
    return legacy_image_tensor(image, force_rgb)
