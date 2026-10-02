"""Pixel-preserving margin crops with one geometry rule for IMAGE and MASK."""

from copy import deepcopy

from .crop_utils import CropImageByMargins, CropMaskByMargins

from .core.margins import (_crop_box, ImageMarginEngine, MaskMarginEngine)


class PortraitCropImageMarginsPreview(ImageMarginEngine):
    @classmethod
    def INPUT_TYPES(cls):
        return deepcopy(CropImageByMargins.INPUT_TYPES())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "PortraitUtils/Transform"


class PortraitCropMaskMarginsPreview(MaskMarginEngine):
    @classmethod
    def INPUT_TYPES(cls):
        return deepcopy(CropMaskByMargins.INPUT_TYPES())

    RETURN_TYPES = ("MASK",)
    FUNCTION = "crop"
    CATEGORY = "PortraitUtils/Transform"


NODE_CLASS_MAPPINGS = {
    "PortraitCropImageMarginsPreview": PortraitCropImageMarginsPreview,
    "PortraitCropMaskMarginsPreview": PortraitCropMaskMarginsPreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitCropImageMarginsPreview": "Crop by Margins (Image)",
    "PortraitCropMaskMarginsPreview": "Crop by Margins (Mask)",
}
