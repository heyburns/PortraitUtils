"""Typed, stage-scoped configuration buses for PortraitUtils workflows.

V2 deliberately keeps configuration objects separate from images and masks.
PortraitUtils processing wrappers consume the typed objects directly; small
readers expose primitive values only where a third-party node requires them.
Only the current configured interfaces are registered. Shared processing
implementations are internal and do not expose their retired node IDs.
"""

from __future__ import annotations

from .auto_adjust import AutoAdjustNode
from .intelligent_auto_crop import IntelligentAutoCrop
from .load_image_combined import LoadImageCombined
from .outpaint_config import OutpaintPaddingComputeNode
from .core.outpaint import compute_padding
from .core.borders import BorderCropEngine
from .core.composite import CompositeEngine
from .core.tensors import image_tensor
from .smart_photo_prepare import ASPECT_OPTIONS
from .stitch_image_by_mask import StitchByMask
from .seed_utils import (
    SEED_MAX,
    SEED_MIN,
    _clamp_seed,
    _new_random_seed,
    _resolve_seed,
)

from .core.validation import choice, integer
from .core.contracts import (
    validate_config,
    PADDING_FILLS,
    CROP_CONFIG_TYPE,
    EDIT_CONFIG_TYPE,
    OUTPAINT_CONFIG_TYPE,
    FINISH_CONFIG_TYPE,
    SOURCE_INFO_TYPE,
    CropConfigV2,
    EditConfigV2,
    OutpaintConfigV2,
    FinishConfigV2,
    SourceInfoV2,
    _expect,
)


class CropConfigNodeV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "autocrop_section": (
                    ["── AUTOCROP · SOLID BORDER REMOVAL ──"],
                    {"default": "── AUTOCROP · SOLID BORDER REMOVAL ──"},
                ),
                "autocrop_strip_bottom_banner": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Remove a dark text/banner strip touching the bottom edge before border detection.",
                    },
                ),
                "autocrop_detect_borders": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Remove uniform solid borders before subject framing.",
                    },
                ),
                "autocrop_fuzz_tolerance": (
                    "FLOAT",
                    {
                        "default": 0.07,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Color variation tolerated while identifying a solid border.",
                    },
                ),
                "autocrop_edge_uniformity": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Fraction of an edge row/column that must match the border color.",
                    },
                ),
                "autocrop_pad_px": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Pixels of detected border to retain on each cropped edge.",
                    },
                ),
                "smart_crop_section": (
                    ["── SMART CROP · SUBJECT / NATIVE SIZE ──"],
                    {"default": "── SMART CROP · SUBJECT / NATIVE SIZE ──"},
                ),
                "smart_crop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "OFF keeps the widest safe frame. ON frames more tightly around the subject mask.",
                    },
                ),
                "resolution_policy": (
                    [
                        "Auto (preserve detail)",
                        "Auto (closest scale)",
                        "Force 1 MP",
                        "Force 1.5 MP",
                        "Force 2K (~4 MP)",
                    ],
                    {
                        "default": "Auto (preserve detail)",
                        "tooltip": "Auto: FireRed uses ~1 MP workflow presets; Klein can use 1/1.5 MP; Qwen 2.1 can also use 2K (~4 MP). Force 1.5 MP is an experimental larger output for FireRed. Force 2K is Qwen 2.1 only.",
                    },
                ),
                "forced_aspect_ratio": (
                    list(ASPECT_OPTIONS),
                    {
                        "default": "Auto",
                        "tooltip": "Force an exact output aspect family, or use Auto for minimal cropping.",
                    },
                ),
                "subject_mask": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Subject mask prompt",
                    },
                ),
            },
            "optional": {
                "padding_fill": (
                    list(PADDING_FILLS),
                    {
                        "default": "Edge extension",
                        "tooltip": "Smart Photo Prepare only: fill added canvas with Black, White, or repeated edge pixels. Does not change crop/protection decisions or AutoCrop border removal.",
                    },
                ),
            },
        }

    RETURN_TYPES = (CROP_CONFIG_TYPE,)
    RETURN_NAMES = ("crop_config",)
    FUNCTION = "build"
    CATEGORY = "PortraitUtils/Config V2"

    def build(
        self,
        autocrop_section,
        autocrop_strip_bottom_banner,
        autocrop_detect_borders,
        autocrop_fuzz_tolerance,
        autocrop_edge_uniformity,
        autocrop_pad_px,
        smart_crop_section,
        smart_crop,
        resolution_policy,
        forced_aspect_ratio,
        subject_mask,
        padding_fill="Edge extension",
    ):
        return (
            validate_config(CropConfigV2(
                autocrop_strip_bottom_banner,
                autocrop_detect_borders,
                autocrop_fuzz_tolerance,
                autocrop_edge_uniformity,
                autocrop_pad_px,
                smart_crop,
                resolution_policy,
                forced_aspect_ratio,
                subject_mask,
                padding_fill,
            ), "crop_config"),
        )


class EditConfigNodeV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_opacity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "stitch_opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "inpaint_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Image-edit / inpaint prompt",
                    },
                ),
                "stitch_prompt_1": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Stitch 1 (e.g. hair)",
                    },
                ),
                "stitch_prompt_2": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Stitch 2 (e.g. background)",
                    },
                ),
                "stitch_prompt_3": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Stitch 3 (e.g. rocks)",
                    },
                ),
            }
        }

    RETURN_TYPES = (EDIT_CONFIG_TYPE,)
    RETURN_NAMES = ("edit_config",)
    FUNCTION = "build"
    CATEGORY = "PortraitUtils/Config V2"

    def build(self, blend_opacity, stitch_opacity, inpaint_prompt,
              stitch_prompt_1, stitch_prompt_2, stitch_prompt_3):
        return (
            validate_config(EditConfigV2(
                inpaint_prompt=inpaint_prompt,
                blend_opacity=blend_opacity,
                stitch_opacity=stitch_opacity,
                stitch_prompt_1=stitch_prompt_1,
                stitch_prompt_2=stitch_prompt_2,
                stitch_prompt_3=stitch_prompt_3,
            ), "edit_config"),
        )


class OutpaintConfigNodeV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Percent", "Pixels"], {"default": "Percent"}),
                "gravity": (
                    [
                        "center",
                        "left",
                        "right",
                        "top",
                        "bottom",
                        "top left",
                        "top right",
                        "bottom right",
                        "bottom left",
                    ],
                    {"default": "center"},
                ),
                "horizontal_percent": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 10000.0, "step": 0.1},
                ),
                "vertical_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 10000.0, "step": 0.1},
                ),
                "left_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "right_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "top_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "bottom_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "outpaint_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Outpaint prompt",
                    },
                ),
            }
        }

    RETURN_TYPES = (OUTPAINT_CONFIG_TYPE,)
    RETURN_NAMES = ("outpaint_config",)
    FUNCTION = "build"
    CATEGORY = "PortraitUtils/Config V2"

    def build(
        self,
        mode,
        gravity,
        horizontal_percent,
        vertical_percent,
        left_px,
        right_px,
        top_px,
        bottom_px,
        outpaint_prompt,
    ):
        return (
            validate_config(OutpaintConfigV2(
                mode,
                gravity,
                horizontal_percent,
                vertical_percent,
                left_px,
                right_px,
                top_px,
                bottom_px,
                outpaint_prompt
            ), "outpaint_config"),
        )


class FinishConfigNodeV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_upscale": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.1, "max": 100.0, "step": 0.1},
                ),
                "final_upscale": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1},
                ),
                "auto_levels": ("BOOLEAN", {"default": False}),
                "auto_tone": ("BOOLEAN", {"default": False}),
                "auto_color": ("BOOLEAN", {"default": False}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "seed": (
                    "INT",
                    {"default": 0, "min": SEED_MIN, "max": SEED_MAX},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (FINISH_CONFIG_TYPE,)
    RETURN_NAMES = ("finish_config",)
    FUNCTION = "build"
    CATEGORY = "PortraitUtils/Config V2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed = kwargs.get("seed", 0)
        if seed in (-1, -2, -3):
            return _new_random_seed()
        visible = tuple(
            (key, value)
            for key, value in kwargs.items()
            if key not in ("prompt", "extra_pnginfo", "unique_id")
        )
        return repr(visible)

    def build(
        self,
        initial_upscale,
        final_upscale,
        auto_levels,
        auto_tone,
        auto_color,
        flip_horizontal,
        seed,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        integer(seed, "Finish Config.seed", -3, SEED_MAX)
        seed_value = _clamp_seed(
            _resolve_seed(seed, prompt, extra_pnginfo, unique_id)
        )
        return (
            validate_config(FinishConfigV2(
                initial_upscale,
                final_upscale,
                auto_levels,
                auto_tone,
                auto_color,
                flip_horizontal,
                seed_value
            ), "finish_config"),
        )


class LoadImageCombinedV2(LoadImageCombined):
    RETURN_TYPES = ("IMAGE", SOURCE_INFO_TYPE)
    RETURN_NAMES = ("image", "source_info")
    FUNCTION = "load_image_v2"

    def load_image_v2(
        self,
        mode,
        input_dir,
        output_dir,
        pattern,
        strip_trailing_numbers,
        repeat_last,
        image,
    ):
        loaded, filename, resolved_output_dir, width, height = self.load_image(
            mode,
            input_dir,
            output_dir,
            pattern,
            strip_trailing_numbers,
            repeat_last,
            image,
        )
        return (
            loaded,
            SourceInfoV2(filename, resolved_output_dir, int(width), int(height)),
        )


class IntelligentAutoCropV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_config": (CROP_CONFIG_TYPE, {"forceInput": True}),
            }
        }

    RETURN_TYPES = IntelligentAutoCrop.RETURN_TYPES
    RETURN_NAMES = IntelligentAutoCrop.RETURN_NAMES
    FUNCTION = "run"
    CATEGORY = "PortraitUtils/Transform V2"

    def run(self, image, crop_config):
        config = _expect(crop_config, CropConfigV2, "crop_config")
        return BorderCropEngine().run(image, config, "Configured tolerances")




class OutpaintPaddingComputeV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "outpaint_config": (OUTPAINT_CONFIG_TYPE, {"forceInput": True}),
            }
        }

    RETURN_TYPES = OutpaintPaddingComputeNode.RETURN_TYPES
    RETURN_NAMES = OutpaintPaddingComputeNode.RETURN_NAMES
    FUNCTION = "compute"
    CATEGORY = "PortraitUtils/Config V2"

    def compute(self, image, outpaint_config):
        config = _expect(
            outpaint_config, OutpaintConfigV2, "outpaint_config"
        )
        tensor = image_tensor(image, "Outpaint Padding.image")
        return compute_padding(int(tensor.shape[2]), int(tensor.shape[1]), config)


class AutoAdjustV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "finish_config": (FINISH_CONFIG_TYPE, {"forceInput": True}),
                "precision": (
                    ["Histogram (fast)", "Exact"],
                    {
                        "default": "Histogram (fast)",
                        "tooltip": "256-bin statistics are efficient and close to Photoshop's histogram-based behavior.",
                    },
                ),
                "levels_shadow_clip_pct": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "levels_highlight_clip_pct": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "levels_gamma_normalize": ("BOOLEAN", {"default": False}),
                "tone_mode": (
                    ["Per-channel", "Monochromatic"],
                    {"default": "Per-channel"},
                ),
                "tone_shadow_clip_pct": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "tone_highlight_clip_pct": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "snap_neutral_midtones": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Used by Auto Color to neutralize genuinely low-chroma midtones.",
                    },
                ),
                "adjustment_mode": (
                    [
                        "Use Finish Config switches",
                        "Auto Levels / Contrast",
                        "Auto Tone",
                        "Auto Color",
                    ],
                    {
                        "default": "Use Finish Config switches",
                        "tooltip": "Select one Photoshop-style command or use the existing Finish Config booleans.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend the correction with the original image.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "PortraitUtils/Adjustment V2"

    def apply(
        self,
        image,
        finish_config,
        precision,
        levels_shadow_clip_pct,
        levels_highlight_clip_pct,
        levels_gamma_normalize,
        tone_mode,
        tone_shadow_clip_pct,
        tone_highlight_clip_pct,
        snap_neutral_midtones,
        adjustment_mode="Use Finish Config switches",
        strength=1.0,
    ):
        config = _expect(finish_config, FinishConfigV2, "finish_config")
        node_mode = (
            "Use switches (legacy)"
            if adjustment_mode == "Use Finish Config switches"
            else adjustment_mode
        )
        return AutoAdjustNode().apply(
            image,
            precision,
            config.auto_levels,
            levels_shadow_clip_pct,
            levels_highlight_clip_pct,
            levels_gamma_normalize,
            config.auto_tone,
            tone_mode,
            tone_shadow_clip_pct,
            tone_highlight_clip_pct,
            config.auto_color,
            snap_neutral_midtones,
            config.flip_horizontal,
            node_mode,
            strength,
        )


class StitchByMaskV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "edit_config": (EDIT_CONFIG_TYPE, {"forceInput": True}),
                "opacity_source": (
                    ["blend_opacity", "stitch_opacity"],
                    {"default": "stitch_opacity"},
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "bypass_mask": ("BOOLEAN", {"default": False}),
                "feather_radius": (
                    "INT",
                    {"default": 5, "min": 0, "max": 100, "step": 1},
                ),
                "force_size": ("BOOLEAN", {"default": False}),
                "target_width": (
                    "INT",
                    {"default": 1344, "min": 16, "max": 8192, "step": 1},
                ),
                "target_height": (
                    "INT",
                    {"default": 768, "min": 16, "max": 8192, "step": 1},
                ),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = StitchByMask.RETURN_TYPES
    RETURN_NAMES = StitchByMask.RETURN_NAMES
    FUNCTION = "blend"
    CATEGORY = "PortraitUtils/Composite V2"

    def blend(
        self,
        image_a,
        image_b,
        edit_config,
        opacity_source,
        invert_mask=False,
        bypass_mask=False,
        feather_radius=5,
        force_size=False,
        target_width=1344,
        target_height=768,
        mask=None,
    ):
        config = _expect(edit_config, EditConfigV2, "edit_config")
        return CompositeEngine().blend(
            image_a, image_b, config, opacity_source, invert_mask, bypass_mask,
            feather_radius, force_size, target_width, target_height, mask,
        )


class CropPromptV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"crop_config": (CROP_CONFIG_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subject_mask",)
    FUNCTION = "read"
    CATEGORY = "PortraitUtils/Config V2/Readers"

    def read(self, crop_config):
        return (_expect(crop_config, CropConfigV2, "crop_config").subject_mask,)


class EditPromptV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_config": (EDIT_CONFIG_TYPE, {"forceInput": True}),
                "prompt": (
                    ["inpaint_prompt", "stitch_prompt_1", "stitch_prompt_2", "stitch_prompt_3"],
                    {"default": "inpaint_prompt"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "read"
    CATEGORY = "PortraitUtils/Config V2/Readers"

    def read(self, edit_config, prompt):
        config = _expect(edit_config, EditConfigV2, "edit_config")
        choice(prompt, "prompt", ("inpaint_prompt", "stitch_prompt_1", "stitch_prompt_2", "stitch_prompt_3"))
        return (getattr(config, prompt),)


class OutpaintPromptV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outpaint_config": (OUTPAINT_CONFIG_TYPE, {"forceInput": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("outpaint_prompt",)
    FUNCTION = "read"
    CATEGORY = "PortraitUtils/Config V2/Readers"

    def read(self, outpaint_config):
        return (
            _expect(
                outpaint_config, OutpaintConfigV2, "outpaint_config"
            ).outpaint_prompt,
        )


class FinishSeedV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "finish_config": (FINISH_CONFIG_TYPE, {"forceInput": True})
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "read"
    CATEGORY = "PortraitUtils/Config V2/Readers"

    def read(self, finish_config):
        return (_expect(finish_config, FinishConfigV2, "finish_config").seed,)


class FinishScaleV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "finish_config": (FINISH_CONFIG_TYPE, {"forceInput": True}),
                "scale": (
                    ["initial_upscale", "final_upscale"],
                    {"default": "final_upscale"},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("scale",)
    FUNCTION = "read"
    CATEGORY = "PortraitUtils/Config V2/Readers"

    def read(self, finish_config, scale):
        config = _expect(finish_config, FinishConfigV2, "finish_config")
        choice(scale, "scale", ("initial_upscale", "final_upscale"))
        return (float(getattr(config, scale)),)


class SourceSaveInfoV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_info": (SOURCE_INFO_TYPE, {"forceInput": True})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filename", "output_dir")
    FUNCTION = "read"
    CATEGORY = "PortraitUtils/Config V2/Readers"

    def read(self, source_info):
        info = _expect(source_info, SourceInfoV2, "source_info")
        return info.filename_no_ext, info.output_dir


NODE_CLASS_MAPPINGS = {
    "PortraitCropConfigV2": CropConfigNodeV2,
    "PortraitEditConfigV2": EditConfigNodeV2,
    "PortraitOutpaintConfigV2": OutpaintConfigNodeV2,
    "PortraitFinishConfigV2": FinishConfigNodeV2,
    "OutpaintPaddingComputeV2": OutpaintPaddingComputeV2,
    "AutoAdjustV2": AutoAdjustV2,
    "PortraitCropPromptV2": CropPromptV2,
    "PortraitEditPromptV2": EditPromptV2,
    "PortraitOutpaintPromptV2": OutpaintPromptV2,
    "PortraitFinishSeedV2": FinishSeedV2,
    "PortraitFinishScaleV2": FinishScaleV2,
    "PortraitSourceSaveInfoV2": SourceSaveInfoV2,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitCropConfigV2": "Input & Crop Config",
    "PortraitEditConfigV2": "Edit & Composite Config",
    "PortraitOutpaintConfigV2": "Outpaint Config",
    "PortraitFinishConfigV2": "Finish & Run Config",
    "OutpaintPaddingComputeV2": "Outpaint Padding Compute",
    "AutoAdjustV2": "Auto Adjust (Photoshop-style)",
    "PortraitCropPromptV2": "Crop Prompt from Config",
    "PortraitEditPromptV2": "Edit Prompt from Config",
    "PortraitOutpaintPromptV2": "Outpaint Prompt from Config",
    "PortraitFinishSeedV2": "Seed from Config",
    "PortraitFinishScaleV2": "Scale from Config",
    "PortraitSourceSaveInfoV2": "Save Info from Source",
}
