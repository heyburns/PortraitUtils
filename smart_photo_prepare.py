"""Subject-aware preparation for photo-editing models.

The nodes in this module deliberately separate editor selection from image
preparation.  A tiny beacon can live in each mutually-exclusive editor group,
while one shared prepare node performs the expensive crop/resize operation.
"""

from __future__ import annotations

from .core.editor_profiles import (
    EDITOR_TARGET_TYPE,
    TARGET_FIRERED,
    TARGET_KLEIN,
    TARGET_QWEN21,
    TARGET_DIRECT,
    EDITOR_TARGETS,
    TIER_1MP,
    TIER_15MP,
    TIER_2K,
    ASPECT_AUTO,
    FORCED_ASPECTS,
    ASPECT_OPTIONS,
    _LAZY_NOT_CONNECTED,
    _Resolution,
    _unwrap_target,
    _aligned_resolution,
    _exact_aspect_resolution,
    _build_resolutions,
)
from .core.geometry import (
    _CropChoice,
    _full_frame_crop,
    _full_frame_forced_crop,
    _safe_component_bounds,
    _union_bounds,
    _foreground_bounds,
    _pad_bbox,
    _best_axis_window,
    _protected_axis_origin,
    _protected_safety_axis_interval,
    _margin_axis_interval,
    _tight_forced_aspect_size,
    _forced_aspect_crop,
    _smallest_ratio_size,
    _clamp_to_interval,
    _place_subject_crop,
    _subject_crop,
    _candidate_choices,
    _limited_crop_choices,
    _padded_choices,
    _forced_aspect_choices,
    _choose_resolution,
)

from .core.contracts import PADDING_FILLS
from .core.preparation import PreparationOptions, prepare_photo
from .core.resampling import lanczos_resize
from .core.tensors import (
    aligned_mask as _as_mask,
    pad_crop as _pad_crop,
    resize_mask as _resize_mask,
    resize_image,
)


def _lanczos_resampler(samples, width, height):
    return lanczos_resize(samples, width, height)


def _resize_image(image, width, height, method):
    return resize_image(image, width, height, method, _lanczos_resampler)


class EditorTargetBeacon:
    """A tiny editor selector intended to live inside a muter-controlled group."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": (
                    list(EDITOR_TARGETS),
                    {
                        "default": TARGET_QWEN21,
                        "tooltip": "Place one beacon in each editor group and set it to that group's editor.",
                    },
                )
            }
        }

    RETURN_TYPES = (EDITOR_TARGET_TYPE, "STRING")
    RETURN_NAMES = ("target", "target_name")
    FUNCTION = "emit"
    CATEGORY = "PortraitUtils/Photo Edit"
    DESCRIPTION = "Small editor-target token for use inside mutually-exclusive Fast Groups Muter groups."

    def emit(self, target):
        target = _unwrap_target(target)
        return target, target


class ActiveEditorTarget:
    """Resolve the one beacon that remains active after group muting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fallback_target": (
                    list(EDITOR_TARGETS),
                    {
                        "default": TARGET_DIRECT,
                        "tooltip": "Used only when every connected editor beacon is muted.",
                    },
                )
            },
            "optional": {
                "target_1": (EDITOR_TARGET_TYPE,),
                "target_2": (EDITOR_TARGET_TYPE,),
                "target_3": (EDITOR_TARGET_TYPE,),
                "target_4": (EDITOR_TARGET_TYPE,),
            },
        }

    RETURN_TYPES = (EDITOR_TARGET_TYPE, "STRING")
    RETURN_NAMES = ("active_target", "target_name")
    FUNCTION = "resolve"
    CATEGORY = "PortraitUtils/Photo Edit"
    DESCRIPTION = "Selects the single active group beacon; errors if more than one editor group is active."

    def resolve(self, fallback_target, target_1=None, target_2=None, target_3=None, target_4=None):
        active = [
            target
            for target in (
                _unwrap_target(target_1),
                _unwrap_target(target_2),
                _unwrap_target(target_3),
                _unwrap_target(target_4),
            )
            if target is not None
        ]
        if len(active) > 1:
            raise ValueError(
                "Active Editor Target received more than one live beacon. "
                "Configure Fast Groups Muter so exactly one editor group is active."
            )
        target = active[0] if active else _unwrap_target(fallback_target)
        return target, target



class EditorResultGate:
    """Forward only the selected editor result; never substitute a bypass image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "editor_target": (EDITOR_TARGET_TYPE, {"forceInput": True}),
                "allow_direct": ("BOOLEAN", {"default": False, "tooltip":
                    "Enable only when a deliberate Direct / No Editor run is intended."}),
            },
            "optional": {
                "firered_image": ("IMAGE", {"lazy": True}),
                "klein_image": ("IMAGE", {"lazy": True}),
                "qwen21_image": ("IMAGE", {"lazy": True}),
                "direct_image": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "route"
    CATEGORY = "PortraitUtils/Photo Edit"
    DESCRIPTION = ("Route the actual selected editor output to the upscaler. "
                   "Missing editor results fail instead of falling back to the source.")

    @staticmethod
    def _selected_socket(editor_target, allow_direct):
        if type(allow_direct) is not bool:
            raise ValueError("Editor Result Gate: allow_direct must be a Boolean.")
        target = _unwrap_target(editor_target)
        if target is None:
            raise ValueError("Editor Result Gate: connect Active Editor Target.")
        sockets = {
            TARGET_FIRERED: "firered_image",
            TARGET_KLEIN: "klein_image",
            TARGET_QWEN21: "qwen21_image",
            TARGET_DIRECT: "direct_image",
        }
        if target == TARGET_DIRECT and not allow_direct:
            raise ValueError("Editor Result Gate: no editor is active. Enable allow_direct "
                             "only for an intentional Direct / No Editor run; otherwise "
                             "enable exactly one editor group in Fast Groups Muter.")
        return target, sockets[target]

    def check_lazy_status(self, editor_target, allow_direct=False,
                          firered_image=_LAZY_NOT_CONNECTED, klein_image=_LAZY_NOT_CONNECTED,
                          qwen21_image=_LAZY_NOT_CONNECTED, direct_image=_LAZY_NOT_CONNECTED):
        _, socket = self._selected_socket(editor_target, allow_direct)
        values = dict(firered_image=firered_image, klein_image=klein_image,
                      qwen21_image=qwen21_image, direct_image=direct_image)
        return [socket] if values[socket] is None else []

    def route(self, editor_target, allow_direct=False, firered_image=None, klein_image=None,
              qwen21_image=None, direct_image=None):
        target, socket = self._selected_socket(editor_target, allow_direct)
        values = dict(firered_image=firered_image, klein_image=klein_image,
                      qwen21_image=qwen21_image, direct_image=direct_image)
        image = values[socket]
        if image is None:
            raise ValueError(f"Editor Result Gate: {target} was selected but {socket} has no "
                             "image. Connect that editor's decoded output, not its input "
                             "or a shared pre-editor switch.")
        from .core.tensors import image_tensor
        image_tensor(image, f"Editor Result Gate.{socket}", allow_unbatched=False)
        return (image,)

class SmartPhotoPrepare:
    """Internal crop/resize engine used by the production Photo Prepare node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "editor_target": (EDITOR_TARGET_TYPE, {"forceInput": True}),
                "smart_crop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "OFF preserves the full photo with the least possible aspect crop. ON frames tightly around the mask subject.",
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
                    {"default": "Auto (preserve detail)"},
                ),
                "resize_method": (
                    ["bicubic", "lanczos", "area", "bilinear"],
                    {"default": "bicubic"},
                ),
                "invert_mask": (["auto", "false", "true"], {"default": "auto"}),
                "q_left": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.2, "step": 0.001}),
                "q_right": ("FLOAT", {"default": 0.995, "min": 0.8, "max": 1.0, "step": 0.001}),
                "q_top": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.2, "step": 0.001}),
                "q_bottom": ("FLOAT", {"default": 0.995, "min": 0.8, "max": 1.0, "step": 0.001}),
                "min_span_px": ("INT", {"default": 8, "min": 1, "max": 2048}),
                "headroom_ratio": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.45, "step": 0.01}),
                "footroom_ratio": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.45, "step": 0.01}),
                "side_margin_ratio": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.45, "step": 0.01}),
                "bottom_priority": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
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
                "forced_aspect_ratio": (
                    list(ASPECT_OPTIONS),
                    {
                        "default": ASPECT_AUTO,
                        "tooltip": "Force an exact output aspect family. Auto lets the node choose the least-destructive native family.",
                    },
                ),
            },
            "optional": {
                "padding_fill": (list(PADDING_FILLS), {"default": "Edge extension"}),
                "crop_tolerance_px": (
                    "INT",
                    {
                        "default": 4, "min": 0, "max": 64, "step": 1,
                        "tooltip": "When no safe crop fits, allow at most this many source pixels per edge into available safety margins. 0 always preserves them. The protected-region mask stays intact.",
                    },
                ),
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

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "MASK",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "FLOAT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "prepared_image",
        "native_crop",
        "prepared_mask",
        "target_width",
        "target_height",
        "crop_x",
        "crop_y",
        "crop_width",
        "crop_height",
        "scale_factor",
        "crop_loss_percent",
        "resolution",
        "debug",
    )
    FUNCTION = "prepare"
    CATEGORY = "PortraitUtils/Photo Edit"
    DESCRIPTION = (
        "Shared real-photo cropper and editor-resolution normalizer for FireRed Image Edit 1.1, "
        "FLUX.2 Klein Base, Qwen Image 2.1, or a direct upscaler path."
    )

    def check_lazy_status(
        self,
        forced_aspect_ratio,
        protected_region_mask=_LAZY_NOT_CONNECTED,
        smart_crop=False,
        **kwargs,
    ):
        if protected_region_mask is None:
            return ["protected_region_mask"]
        return []

    def prepare(
        self,
        image,
        editor_target,
        smart_crop,
        resolution_policy,
        resize_method,
        invert_mask,
        q_left,
        q_right,
        q_top,
        q_bottom,
        min_span_px,
        headroom_ratio,
        footroom_ratio,
        side_margin_ratio,
        bottom_priority,
        horiz_gravity,
        framing_tolerance_percent,
        forced_aspect_ratio,
        mask=None,
        protected_region_mask=None,
        crop_tolerance_px=4,
        padding_fill="Edge extension",
    ):
        options = PreparationOptions(
            smart_crop=smart_crop,
            resolution_policy=resolution_policy,
            resize_method=resize_method,
            invert_mask=invert_mask,
            q_left=q_left,
            q_right=q_right,
            q_top=q_top,
            q_bottom=q_bottom,
            min_span_px=min_span_px,
            headroom_ratio=headroom_ratio,
            footroom_ratio=footroom_ratio,
            side_margin_ratio=side_margin_ratio,
            bottom_priority=bottom_priority,
            horiz_gravity=horiz_gravity,
            framing_tolerance_percent=framing_tolerance_percent,
            forced_aspect_ratio=forced_aspect_ratio,
            crop_tolerance_px=crop_tolerance_px,
            padding_fill=padding_fill,
        )
        return prepare_photo(image, editor_target, options, mask, protected_region_mask,
                             resampler=_lanczos_resampler).legacy_outputs()


NODE_CLASS_MAPPINGS = {
    "PortraitEditorTargetBeacon": EditorTargetBeacon,
    "PortraitActiveEditorTarget": ActiveEditorTarget,
    "PortraitEditorResultGate": EditorResultGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitEditorTargetBeacon": "Editor Target Beacon",
    "PortraitActiveEditorTarget": "Active Editor Target",
    "PortraitEditorResultGate": "Editor Result Gate",
}
