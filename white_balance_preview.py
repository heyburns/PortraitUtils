"""Full-resolution, linear-light white balance and D65 Lab reference matching.

This preview deliberately has a new node ID. The legacy implementation and
saved workflows remain unchanged until the replacement has been evaluated.
"""

from __future__ import annotations

import torch.nn.functional as F

from .core.white_balance import (
    METHODS,
    MATCH_METHODS,
    _TILE_PIXELS,
    _HIST_BINS,
    _EPS,
    _LAB_STD_FLOOR,
    _RGB_XYZ,
    _XYZ_RGB,
    _D65,
    _constants,
    srgb_to_linear,
    linear_to_srgb,
    linear_rgb_to_lab,
    rgb_to_lab,
    lab_to_rgb,
    _work_dtype,
    _validate_image,
    _rgb_weights,
    _tiles,
    _analysis_image,
    _wb_gains,
    _lab_moments,
    WhiteBalanceEngine,
)


class PortraitWhiteBalancePreview(WhiteBalanceEngine):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "reference": ("IMAGE",),
            "method": (METHODS.copy(), {"default": "wb_highlight+reinhard"}),
            "percentile": ("FLOAT", {"default": 95.0, "min": 80.0, "max": 99.9, "step": 0.1}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "clip_gamut": ("BOOLEAN", {"default": True}),
            "force_size": ("BOOLEAN", {"default": False,
                                        "tooltip": "Limit statistics sampling only; output stays full resolution."}),
            "target_width": ("INT", {"default": 1440, "min": 16, "max": 8192, "step": 1}),
            "target_height": ("INT", {"default": 1080, "min": 16, "max": 8192, "step": 1}),
        }, "optional": {
            "preserve_luminance": ("BOOLEAN", {"default": True,
                "tooltip": "WB preserves the sampled linear-light luminance instead of forcing highlights to 95% white. "
                           "Reference matching can still change brightness."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "PortraitUtils/Analysis"


NODE_CLASS_MAPPINGS = {"PortraitWhiteBalancePreview": PortraitWhiteBalancePreview}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitWhiteBalancePreview": "Auto White-Balance + Color Match",
}
