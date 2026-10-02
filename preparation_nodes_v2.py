"""Native photo preparation and strongly typed, local metadata readers."""
import json

from .core.contracts import CROP_CONFIG_TYPE, CropConfigV2, _expect
from .core.preparation import PREPARATION_INFO_TYPE, PreparationInfo, PreparationOptions, prepare_photo
from .core.validation import choice
from .smart_photo_prepare import SmartPhotoPrepare, _lanczos_resampler, _LAZY_NOT_CONNECTED


def _info(value):
    if not isinstance(value, PreparationInfo):
        raise TypeError("prepare_info: connect the info output of Smart Photo Prepare. "
                        f"Expected {PREPARATION_INFO_TYPE}, received {type(value).__name__}.")
    value.plan.validate()
    return value


class PhotoPrepareStandardV2:
    """The single registered crop/pad/native-resolution preparation node."""

    @classmethod
    def INPUT_TYPES(cls):
        schema = {
            "required": {
                "image": ("IMAGE",),
                "editor_target": ("PORTRAIT_EDITOR_TARGET", {"forceInput": True}),
                "crop_config": (CROP_CONFIG_TYPE, {"forceInput": True}),
                "resize_method": (
                    ["bicubic", "lanczos", "area", "bilinear"],
                    {"default": "bicubic"},
                ),
                "invert_mask": (["auto", "false", "true"], {"default": "auto"}),
                "q_left": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.0, "max": 0.2, "step": 0.001},
                ),
                "q_right": (
                    "FLOAT",
                    {"default": 0.995, "min": 0.8, "max": 1.0, "step": 0.001},
                ),
                "q_top": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.0, "max": 0.2, "step": 0.001},
                ),
                "q_bottom": (
                    "FLOAT",
                    {"default": 0.995, "min": 0.8, "max": 1.0, "step": 0.001},
                ),
                "min_span_px": ("INT", {"default": 8, "min": 1, "max": 2048}),
                "headroom_ratio": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.0, "max": 0.45, "step": 0.01},
                ),
                "footroom_ratio": (
                    "FLOAT",
                    {"default": 0.06, "min": 0.0, "max": 0.45, "step": 0.01},
                ),
                "side_margin_ratio": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 0.45, "step": 0.01},
                ),
                "bottom_priority": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "horiz_gravity": (["center", "left", "right"], {"default": "center"}),
                "framing_tolerance_percent": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Allows similarly framed safe crops to compete on resize quality; fallback pixel counts take priority.",
                    },
                ),
            },
            "optional": {
                "crop_tolerance_px": SmartPhotoPrepare.INPUT_TYPES()["optional"]["crop_tolerance_px"],
                "mask": ("MASK",),
                "protected_region_mask": (
                    "MASK",
                    {
                        "lazy": True,
                        "tooltip": "Connected critical-region masks are protected in every crop mode, including Auto aspect with Smart Crop off.",
                    },
                ),
            },
        }
        schema["optional"]["profile"] = ("BOOLEAN", {"default": False,
            "tooltip": "Record host stage timings in info. No CUDA synchronization or model-memory changes."})
        return schema

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", PREPARATION_INFO_TYPE)
    RETURN_NAMES = ("prepared_image", "native_crop", "prepared_mask", "prepare_info")
    FUNCTION = "prepare"
    CATEGORY = "PortraitUtils/Photo Edit V2"
    DESCRIPTION = (
        "Prepare a photo for the selected editor with a native source crop and "
        "one typed diagnostics output."
    )

    def check_lazy_status(
        self, crop_config, protected_region_mask=_LAZY_NOT_CONNECTED, **kwargs
    ):
        _expect(crop_config, CropConfigV2, "crop_config")
        if protected_region_mask is None:
            return ["protected_region_mask"]
        return []

    def prepare(self, image, editor_target, crop_config, resize_method, invert_mask,
                q_left, q_right, q_top, q_bottom, min_span_px, headroom_ratio,
                footroom_ratio, side_margin_ratio, bottom_priority, horiz_gravity,
                framing_tolerance_percent, mask=None, protected_region_mask=None,
                crop_tolerance_px=4, profile=False):
        config = _expect(crop_config, CropConfigV2, "crop_config")
        options = PreparationOptions(
            smart_crop=config.smart_crop, resolution_policy=config.resolution_policy,
            forced_aspect_ratio=config.forced_aspect_ratio, resize_method=resize_method,
            invert_mask=invert_mask, q_left=q_left, q_right=q_right, q_top=q_top,
            q_bottom=q_bottom, min_span_px=min_span_px, headroom_ratio=headroom_ratio,
            footroom_ratio=footroom_ratio, side_margin_ratio=side_margin_ratio,
            bottom_priority=bottom_priority, horiz_gravity=horiz_gravity,
            framing_tolerance_percent=framing_tolerance_percent,
            crop_tolerance_px=crop_tolerance_px, padding_fill=config.padding_fill,
        )
        result = prepare_photo(image, editor_target, options, mask, protected_region_mask,
                               profile=profile, resampler=_lanczos_resampler)
        return result.image, result.native_crop, result.mask, result.info


class PrepareDimensionsV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prepare_info": (PREPARATION_INFO_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "read"
    CATEGORY = "PortraitUtils/Photo Edit V2/Readers"

    def read(self, prepare_info):
        p = _info(prepare_info).plan
        return p.target_width, p.target_height


class PrepareCropBoxV2(PrepareDimensionsV2):
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("crop_x", "crop_y", "crop_width", "crop_height")

    def read(self, prepare_info):
        c = _info(prepare_info).plan.crop
        return c.x, c.y, c.width, c.height


class PrepareScaleV2(PrepareDimensionsV2):
    FIELDS = ("scale_factor", "scale_x", "scale_y", "crop_loss_percent")

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"]["field"] = (list(cls.FIELDS), {"default": "scale_factor"})
        return schema

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)

    def read(self, prepare_info, field):
        choice(field, "Prepare Scale.field", self.FIELDS)
        p = _info(prepare_info).plan
        value = {"scale_factor": p.crop.scale, "scale_x": p.scale_xy[0],
                 "scale_y": p.scale_xy[1], "crop_loss_percent": p.crop.crop_loss_percent}[field]
        return (float(value),)


class PrepareSummaryV2(PrepareDimensionsV2):
    FIELDS = ("debug", "resolution", "selection_reason", "JSON")

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"]["field"] = (list(cls.FIELDS), {"default": "debug"})
        return schema

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    def read(self, prepare_info, field):
        choice(field, "Prepare Summary.field", self.FIELDS)
        info = _info(prepare_info)
        if field == "JSON":
            return (json.dumps(info.to_dict(), ensure_ascii=False, allow_nan=False, indent=2),)
        return ({"debug": info.plan.debug, "resolution": info.plan.resolution,
                 "selection_reason": info.plan.reason}[field],)


NODE_CLASS_MAPPINGS = {
    "PortraitPhotoPrepareV2": PhotoPrepareStandardV2,
    "PortraitPrepareDimensionsV2": PrepareDimensionsV2,
    "PortraitPrepareCropBoxV2": PrepareCropBoxV2,
    "PortraitPrepareScaleV2": PrepareScaleV2,
    "PortraitPrepareSummaryV2": PrepareSummaryV2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitPhotoPrepareV2": "Smart Photo Prepare",
    "PortraitPrepareDimensionsV2": "Dimensions from Prepare Info",
    "PortraitPrepareCropBoxV2": "Crop Box from Prepare Info",
    "PortraitPrepareScaleV2": "Scale from Prepare Info",
    "PortraitPrepareSummaryV2": "Summary from Prepare Info",
}
