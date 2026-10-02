"""Conservative, pixel-preserving border/footer removal for the 2.0 preview."""

from __future__ import annotations

from .core.contracts import CROP_CONFIG_TYPE
from .core.borders import (
    POLICIES,
    _LOG,
    _first_failure,
    _boundary_fraction,
    _border_depth,
    _text_like,
    _banner_depth,
    _supported_trims,
    _bounds,
    _validate_config,
    BorderCropEngine,
)


class PortraitAutoCropPreview(BorderCropEngine):
    CATEGORY = "PortraitUtils/Transform"
    DESCRIPTION = ("Remove solid hosting borders and text-bearing dark footers. "
                   "Uses the existing crop config. Crops only; no resize or color changes.")
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("image", "trim_left", "trim_top", "trim_right", "trim_bottom", "detected")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "crop_config": (CROP_CONFIG_TYPE, {"forceInput": True}),
        }, "optional": {
            "border_policy": (POLICIES, {"default": "Conservative", "tooltip":
                "Conservative: require 98.5% solid edge coverage and cap color tolerance at 0.12. "
                "Configured tolerances: use the config thresholds literally; inspect crops carefully. "
                "Both require a bounded border and a contrasting content boundary."}),
        }}


NODE_CLASS_MAPPINGS = {"PortraitAutoCropPreview": PortraitAutoCropPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitAutoCropPreview": "Intelligent AutoCrop"}
