"""Straighten scanned prints without quantizing or discarding source detail."""

from .core.scans import ScannedPhotoEngine


class ProcessScannedPhoto:
    """Automatically straightens and crops scanned photos in a single pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "straighten": ("BOOLEAN", {"default": True}),
                "crop_mode": (["Inner Photo Frame", "Scanner Bed Only", "None"], {"default": "Inner Photo Frame"}),
                "padding": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "PortraitUtils/Transform"

    def process(self, image, straighten, crop_mode, padding, threshold):
        return ScannedPhotoEngine().process(
            image, straighten, crop_mode, padding, threshold,
        )
