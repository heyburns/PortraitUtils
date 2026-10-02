"""Exact pixel-margin crops for IMAGE and standard BHW MASK tensors."""

from .core.margins import ImageMarginEngine, MaskMarginEngine


class CropImageByMargins:
    """Crop an IMAGE tensor by pixel margins (left/top/right/bottom)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "top_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "right_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "bottom_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "snap_multiple": ("INT", {"default": 1, "min": 1, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "PortraitUtils/Transform"

    def crop(self, image, left_px, top_px, right_px, bottom_px, snap_multiple=1):
        return ImageMarginEngine().crop(
            image, left_px, top_px, right_px, bottom_px, snap_multiple,
        )

class CropMaskByMargins:
    """Same as above but for MASK input and output (single-channel)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "left_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "top_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "right_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "bottom_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "snap_multiple": ("INT", {"default": 1, "min": 1, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "crop"
    CATEGORY = "PortraitUtils/Transform"

    def crop(self, mask, left_px, top_px, right_px, bottom_px, snap_multiple=1):
        return MaskMarginEngine().crop(
            mask, left_px, top_px, right_px, bottom_px, snap_multiple,
        )
