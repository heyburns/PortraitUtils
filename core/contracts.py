"""Immutable, data-only workflow contracts; independent of ComfyUI."""
from dataclasses import dataclass
from typing import Any
from .validation import boolean, choice, integer, number, text
from .editor_profiles import ASPECT_OPTIONS

RESOLUTION_POLICIES = ("Auto (preserve detail)", "Auto (closest scale)", "Force 1 MP", "Force 1.5 MP", "Force 2K (~4 MP)")
PADDING_FILLS = ("Edge extension", "Black", "White")
GRAVITIES = ("center", "left", "right", "top", "bottom", "top left", "top right", "bottom left", "bottom right")

CROP_CONFIG_TYPE = "PORTRAIT_CROP_CONFIG_V2"

EDIT_CONFIG_TYPE = "PORTRAIT_EDIT_CONFIG_V2"

OUTPAINT_CONFIG_TYPE = "PORTRAIT_OUTPAINT_CONFIG_V2"

FINISH_CONFIG_TYPE = "PORTRAIT_FINISH_CONFIG_V2"

SOURCE_INFO_TYPE = "PORTRAIT_SOURCE_INFO_V2"

@dataclass(frozen=True)
class CropConfigV2:
    autocrop_strip_bottom_banner: bool
    autocrop_detect_borders: bool
    autocrop_fuzz_tolerance: float
    autocrop_edge_uniformity: float
    autocrop_pad_px: int
    smart_crop: bool
    resolution_policy: str
    forced_aspect_ratio: str
    subject_mask: str
    padding_fill: str = "Edge extension"

@dataclass(frozen=True)
class EditConfigV2:
    inpaint_prompt: str
    blend_opacity: float
    stitch_opacity: float
    stitch_prompt_1: str
    stitch_prompt_2: str = ""
    stitch_prompt_3: str = ""

@dataclass(frozen=True)
class OutpaintConfigV2:
    mode: str
    gravity: str
    horizontal_percent: float
    vertical_percent: float
    left_px: int
    right_px: int
    top_px: int
    bottom_px: int
    outpaint_prompt: str

@dataclass(frozen=True)
class FinishConfigV2:
    initial_upscale: float
    final_upscale: float
    auto_levels: bool
    auto_tone: bool
    auto_color: bool
    flip_horizontal: bool
    seed: int

@dataclass(frozen=True)
class SourceInfoV2:
    filename_no_ext: str
    output_dir: str
    width: int
    height: int

def _expect(value: Any, expected_type, label: str):
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{label} expected {expected_type.__name__}; received "
            f"{type(value).__name__}. Connect the matching PortraitUtils V2 config node."
        )
    return validate_config(value, label)


def validate_config(value, label="config"):
    """Validate at producer/consumer boundaries, not in dataclass construction."""
    def field(name):
        return getattr(value, name), f"{label}.{name}"

    if isinstance(value, CropConfigV2):
        for name in ("autocrop_strip_bottom_banner", "autocrop_detect_borders", "smart_crop"):
            boolean(*field(name))
        number(*field("autocrop_fuzz_tolerance"), 0, 0.5)
        number(*field("autocrop_edge_uniformity"), 0, 1)
        integer(*field("autocrop_pad_px"), 0, 256)
        choice(*field("resolution_policy"), RESOLUTION_POLICIES)
        choice(*field("forced_aspect_ratio"), ASPECT_OPTIONS)
        text(*field("subject_mask"))
        choice(*field("padding_fill"), PADDING_FILLS)
    elif isinstance(value, EditConfigV2):
        for name in ("blend_opacity", "stitch_opacity"):
            number(*field(name), 0, 1)
        for name in ("inpaint_prompt", "stitch_prompt_1", "stitch_prompt_2", "stitch_prompt_3"):
            text(*field(name))
    elif isinstance(value, OutpaintConfigV2):
        choice(*field("mode"), ("Percent", "Pixels"))
        choice(*field("gravity"), GRAVITIES)
        for name in ("horizontal_percent", "vertical_percent"):
            number(*field(name), 0, 10000)
        for name in ("left_px", "right_px", "top_px", "bottom_px"):
            integer(*field(name), 0, 1000000)
        text(*field("outpaint_prompt"))
    elif isinstance(value, FinishConfigV2):
        for name in ("initial_upscale", "final_upscale"):
            number(*field(name), 0.1, 100)
        for name in ("auto_levels", "auto_tone", "auto_color", "flip_horizontal"):
            boolean(*field(name))
        integer(*field("seed"), 0, 0xffffffff)
    elif isinstance(value, SourceInfoV2):
        for name in ("filename_no_ext", "output_dir"):
            text(*field(name))
        for name in ("width", "height"):
            integer(*field(name), 1, 2**31 - 1)
    else:
        raise TypeError(f"{label}: unsupported config type {type(value).__name__}.")
    return value
