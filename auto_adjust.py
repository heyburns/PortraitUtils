"""Photoshop-style automatic tone and color adjustments for ComfyUI images.

The implementation follows Adobe's documented automatic correction families:

* Auto Levels uses one monochromatic contrast mapping for all RGB channels.
* Auto Tone maximizes the range of each RGB channel independently.
* Auto Color estimates average dark/light colors and can neutralize a genuinely
  low-chroma midtone using per-channel gamma.

Adobe's exact heuristics are proprietary, so this is a conservative,
photography-oriented approximation rather than a bit-exact clone.
"""


from .core.adjustments import (
    _EPSILON,
    _MIN_TONAL_SPAN,
    _AUTO_COLOR_TAIL_FRACTION,
    _normalize_valid_mask,
    _mask_for_channel,
    _extreme,
    _percentiles_exact,
    _percentiles_hist,
    _percentile,
    _tonal_bounds,
    _linear_stretch,
    _linear_stretch_scalar,
    _luma,
    _apply_composite_gamma,
    _auto_levels,
    _auto_tone,
    _masked_average,
    _snap_neutral_midtones,
    _auto_color,
    _rgb_to_ycbcr,
    _ycbcr_to_rgb,
    _to_1d,
    _median_masked,
    AutoAdjustmentEngine,
)


class AutoAdjustNode(AutoAdjustmentEngine):
    """Internal adjustment implementation used by AutoAdjustV2, not registered."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "precision": (
                    ["Histogram (fast)", "Exact"],
                    {
                        "default": "Histogram (fast)",
                        "tooltip": "256-bin statistics are fast and Photoshop-like; Exact sorts every pixel.",
                    },
                ),
                "auto_levels": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Photoshop Auto Contrast-style: one contrast stretch shared by all channels.",
                    },
                ),
                "levels_shadow_clip_pct": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Dark-pixel clipping used by Auto Levels and Auto Color.",
                    },
                ),
                "levels_highlight_clip_pct": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Light-pixel clipping used by Auto Levels and Auto Color.",
                    },
                ),
                "levels_gamma_normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Optional Levels-style midtone gamma normalization; not part of Photoshop Auto Contrast.",
                    },
                ),
                "auto_tone": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Photoshop Auto Tone-style per-channel contrast. Usually use this instead of Auto Levels, not after it.",
                    },
                ),
                "tone_mode": (
                    ["Per-channel", "Monochromatic"],
                    {
                        "default": "Per-channel",
                        "tooltip": "Per-channel matches Auto Tone; Monochromatic behaves like Auto Contrast.",
                    },
                ),
                "tone_shadow_clip_pct": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "tone_highlight_clip_pct": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "auto_color": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Photoshop Auto Color-style dark/light color correction. Includes its own contrast adjustment.",
                    },
                ),
                "snap_neutral_midtones": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Neutralize only genuinely low-chroma midtones; saturated skin and scenery are excluded.",
                    },
                ),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "adjustment_mode": (
                    [
                        "Use switches (legacy)",
                        "Auto Levels / Contrast",
                        "Auto Tone",
                        "Auto Color",
                    ],
                    {
                        "default": "Use switches (legacy)",
                        "tooltip": "Choose one Photoshop-style command, or honor the existing boolean switches for workflow compatibility.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend the correction with the original, like adjustment-layer opacity.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "PortraitUtils/Adjustment"
    DESCRIPTION = (
        "Photoshop-style Auto Levels, Auto Tone, and Auto Color. The three "
        "corrections are normally alternatives; enabling several deliberately "
        "stacks them in Levels -> Tone -> Color order."
    )
