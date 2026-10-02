"""Analyze masks -> plan framing -> apply pixels, with no ComfyUI dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
import math
from time import perf_counter

import numpy as np
import torch

from .contracts import PADDING_FILLS, RESOLUTION_POLICIES
from .editor_profiles import (
    ASPECT_AUTO, ASPECT_OPTIONS, EDITOR_TARGETS, FORCED_ASPECTS, TARGET_DIRECT, TARGET_QWEN21,
    _Resolution, _build_resolutions, _unwrap_target,
)
from .geometry import (
    _CropChoice, _candidate_choices, _choose_resolution, _forced_aspect_choices,
    _forced_aspect_crop, _foreground_bounds, _limited_crop_choices, _pad_bbox,
    _padded_choices, _protected_safety_axis_interval, _safe_component_bounds, _subject_crop, _union_bounds,
)
from .masks import _auto_pick_foreground, _quantile_bounds
from .tensors import aligned_mask, image_tensor, normalized, pad_crop, resize_image, resize_mask
from .validation import boolean, choice, integer, number

_LOG = logging.getLogger("PortraitUtils.preparation")
PREPARATION_INFO_TYPE = "PORTRAIT_PREPARATION_INFO_V2"


@dataclass(frozen=True)
class PreparationOptions:
    smart_crop: bool = False
    resolution_policy: str = "Auto (preserve detail)"
    resize_method: str = "bicubic"
    invert_mask: str = "auto"
    q_left: float = 0.005
    q_right: float = 0.995
    q_top: float = 0.005
    q_bottom: float = 0.995
    min_span_px: int = 8
    headroom_ratio: float = 0.12
    footroom_ratio: float = 0.06
    side_margin_ratio: float = 0.08
    bottom_priority: float = 0.75
    horiz_gravity: str = "center"
    framing_tolerance_percent: float = 0.5
    forced_aspect_ratio: str = ASPECT_AUTO
    crop_tolerance_px: int = 4
    padding_fill: str = "Edge extension"

    def validate(self):
        boolean(self.smart_crop, "Smart Photo Prepare.smart_crop")
        choice(self.resolution_policy, "resolution_policy", RESOLUTION_POLICIES)
        choice(self.resize_method, "resize_method", ("bicubic", "lanczos", "area", "bilinear"))
        choice(self.padding_fill, "padding_fill", PADDING_FILLS)
        choice(self.invert_mask, "invert_mask", ("auto", "false", "true"))
        choice(self.horiz_gravity, "horiz_gravity", ("center", "left", "right"))
        try:
            choice(self.forced_aspect_ratio, "forced_aspect_ratio", ASPECT_OPTIONS)
        except ValueError as exc:
            raise ValueError(f"{exc} If an aspect was selected accidentally, set Forced Aspect Ratio to Auto.") from exc
        for name in ("q_left", "q_right", "q_top", "q_bottom", "bottom_priority"):
            number(getattr(self, name), name, 0, 1)
        if self.q_left >= self.q_right or self.q_top >= self.q_bottom:
            raise ValueError("Smart Photo Prepare: mask quantiles must satisfy q_left < q_right "
                             "and q_top < q_bottom. Restore the default quantiles.")
        for name in ("headroom_ratio", "footroom_ratio", "side_margin_ratio"):
            number(getattr(self, name), name, 0, 0.45)
        number(self.framing_tolerance_percent, "framing_tolerance_percent", 0, 10)
        integer(self.min_span_px, "min_span_px", 1, 2048)
        integer(self.crop_tolerance_px, "crop_tolerance_px", 0, 64)
        return self


@dataclass(frozen=True)
class MaskAnalysis:
    subject_bbox: tuple | None
    protected_bbox: tuple | None
    subject_foreground: np.ndarray | None
    note: str
    subject_status: str
    protected_status: str


def analyze_masks(mask, protected_region_mask, width, height, options):
    """Only analysis arrays cross to CPU; the original IMAGE is never sampled here."""
    options.validate()
    bbox, foreground = None, None
    subject_status = "not connected"
    note = "no subject mask connected; using source frame"
    if mask is not None:
        # BHW is explicit: no squeeze ambiguity for one-pixel-wide masks.
        array = mask[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        interpretation = "false" if options.invert_mask == "auto" and np.ptp(array) == 0 else options.invert_mask
        foreground, picked = _auto_pick_foreground(array, interpretation)
        if float(foreground.sum()) <= 1e-6:
            foreground = None
            subject_status = "empty"
            note = f"subject mask was empty after {picked}; using source frame"
        else:
            quantile_bbox = _quantile_bounds(foreground, options.q_left, options.q_right,
                                           options.q_top, options.q_bottom, options.min_span_px)
            bbox = _safe_component_bounds(foreground, quantile_bbox)
            subject_status = picked
            mode = "tight subject framing" if options.smart_crop else "loose subject framing"
            note = f"{mode} bbox={bbox}, quantile_bbox={quantile_bbox}, mask={picked}"
    protected_bbox = None
    protected_status = "not connected"
    if protected_region_mask is not None:
        array = protected_region_mask[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        protected, picked = _auto_pick_foreground(array, "false")
        protected_bbox = _foreground_bounds(protected)
        if protected_bbox is not None:
            protected_status = "active"
            protected_bbox = _pad_bbox(protected_bbox, max(2, round(0.01 * min(width, height))), width, height)
            note += f"; protected bbox={protected_bbox}, mask={picked}"
        else:
            protected_status = "empty"
            note += "; protected-region mask was empty"
            _LOG.warning("Protected-region mask is connected but empty: no critical region can be protected.")
    elif bbox is not None:
        note += "; no additional protected-region mask"
    return MaskAnalysis(bbox, protected_bbox, foreground, note, subject_status, protected_status)


@dataclass(frozen=True)
class PreparationPlan:
    """No pixel tensors or mutable config. All coordinates refer to the input IMAGE."""
    source_width: int
    source_height: int
    editor_target: str
    options: PreparationOptions
    crop: _CropChoice
    protected_bbox: tuple | None
    subject_bbox: tuple | None
    mask_note: str
    subject_status: str
    protected_status: str

    @property
    def target_width(self):
        return self.crop.resolution.width

    @property
    def target_height(self):
        return self.crop.resolution.height

    @property
    def scale_xy(self):
        left, right, top, bottom = self.crop.padding
        return (self.target_width / (self.crop.width + left + right),
                self.target_height / (self.crop.height + top + bottom))

    def source_to_output(self, x, y):
        """Map pixel-center coordinates; coordinates in padding need not have a source pixel."""
        sx, sy = self.scale_xy
        left, _, top, _ = self.crop.padding
        return ((x - self.crop.x + left + 0.5) * sx - 0.5,
                (y - self.crop.y + top + 0.5) * sy - 0.5)

    def output_to_source(self, x, y):
        sx, sy = self.scale_xy
        left, _, top, _ = self.crop.padding
        return ((x + 0.5) / sx - 0.5 + self.crop.x - left,
                (y + 0.5) / sy - 0.5 + self.crop.y - top)

    @property
    def resolution(self):
        c = self.crop
        if self.editor_target == TARGET_DIRECT:
            aspect = self.options.forced_aspect_ratio
            suffix = f" [{aspect} forced]" if aspect != ASPECT_AUTO else ""
            return f"{self.target_width}x{self.target_height} native{suffix}"
        return f"{self.target_width}x{self.target_height} [{c.resolution.tier}, {c.resolution.family}]"

    @property
    def reason(self):
        resize = "upscale" if self.crop.scale > 1.0005 else "downscale" if self.crop.scale < 0.9995 else "same size"
        fill_note = f"; padding_fill={self.options.padding_fill}" if any(self.crop.padding) else ""
        return (f"{self.options.resolution_policy}; selected {self.resolution}, {resize}; "
                f"{'tight subject' if self.options.smart_crop else 'widest safe'} framing; "
                f"{self.crop.margin_note}{fill_note}")

    @property
    def debug(self):
        c = self.crop
        if self.editor_target == TARGET_DIRECT:
            resize_note = "no resize"
        else:
            direction = "upscale" if c.scale > 1.0005 else "downscale" if c.scale < 0.9995 else "same-size"
            resize_note = f"{direction}={c.scale:.5f}x"
        area = self.source_width * self.source_height
        fill_note = f"; padding_fill={self.options.padding_fill}" if any(c.padding) else ""
        return (f"target={self.editor_target}; policy={self.options.resolution_policy}; {self.mask_note}; "
                f"source={self.source_width}x{self.source_height}; crop=({c.x},{c.y},{c.width},{c.height}) "
                f"loss={c.crop_loss_percent:.3f}%; output={self.resolution}; {resize_note}; "
                f"removed_px={area - c.width * c.height}; added_px={c.added_pixels}; "
                f"changed_px={c.changed_pixels(area)}; framing_cost={c.framing_cost:.3f}; {c.margin_note}{fill_note}")

    def validate(self):
        if not isinstance(self.options, PreparationOptions):
            raise TypeError("Preparation plan: options must be an immutable PreparationOptions object.")
        self.options.validate()
        choice(self.editor_target, "plan.editor_target", EDITOR_TARGETS)
        integer(self.source_width, "plan.source_width", 1, 2**31 - 1)
        integer(self.source_height, "plan.source_height", 1, 2**31 - 1)
        if not isinstance(self.crop, _CropChoice) or not isinstance(self.crop.resolution, _Resolution):
            raise TypeError("Preparation plan: expected an immutable crop/resolution choice.")
        c = self.crop
        integer(c.width, "plan.crop.width", 1, self.source_width)
        integer(c.height, "plan.crop.height", 1, self.source_height)
        integer(c.x, "plan.crop.x", 0, self.source_width - c.width)
        integer(c.y, "plan.crop.y", 0, self.source_height - c.height)
        integer(self.target_width, "plan.target_width", 1, 2**31 - 1)
        integer(self.target_height, "plan.target_height", 1, 2**31 - 1)
        if not isinstance(c.padding, tuple) or len(c.padding) != 4:
            raise ValueError("Preparation plan: padding must be an immutable (left, right, top, bottom) tuple.")
        for pad in c.padding:
            integer(pad, "plan.padding", 0, 2**31 - 1)
        number(c.framing_cost, "plan.framing_cost", 0, math.inf)
        number(c.crop_loss_percent, "plan.crop_loss_percent", 0, 100)
        integer(c.margin_intrusion_px, "plan.margin_intrusion_px", 0, self.options.crop_tolerance_px)
        for label, bounds in (("protected_bbox", self.protected_bbox), ("subject_bbox", self.subject_bbox)):
            if bounds is not None:
                if not isinstance(bounds, tuple) or len(bounds) != 4:
                    raise ValueError(f"Preparation plan: {label} must be an immutable (x, y, width, height) tuple.")
                x, y, width, height = bounds
                integer(width, f"plan.{label}.width", 1, self.source_width)
                integer(height, f"plan.{label}.height", 1, self.source_height)
                integer(x, f"plan.{label}.x", 0, self.source_width - width)
                integer(y, f"plan.{label}.y", 0, self.source_height - height)
        if self.protected_bbox is not None:
            x, y, width, height = self.protected_bbox
            if not (c.x <= x and c.y <= y and c.x + c.width >= x + width and c.y + c.height >= y + height):
                raise ValueError("Preparation plan would crop inside the protected region. "
                                 "Protection is mandatory even with Smart Crop disabled; this plan was rejected.")
            if not any(c.padding):
                ix = _protected_safety_axis_interval(self.source_width, c.width, x, width,
                                                     self.options.side_margin_ratio)
                iy = _protected_safety_axis_interval(self.source_height, c.height, y, height,
                                                     self.options.headroom_ratio)
                tolerance = c.margin_intrusion_px
                if not (ix[0] - tolerance <= c.x <= ix[1] + tolerance and
                        iy[0] - tolerance <= c.y <= iy[1] + tolerance):
                    raise ValueError("Preparation plan enters the protected safety margin beyond crop tolerance. "
                                     "Generate a new plan; the protected edge cannot be replaced by center gravity.")
        if self.editor_target == TARGET_DIRECT:
            left, right, top, bottom = c.padding
            if (self.target_width, self.target_height) != (c.width + left + right, c.height + top + bottom):
                raise ValueError("Direct preparation plan must not resize the source crop/canvas.")
        else:
            containment = _union_bounds(self.subject_bbox, self.protected_bbox)
            smart_ratio = None
            o = self.options
            if o.smart_crop and self.subject_bbox is not None:
                _, _, width, height = containment
                smart_ratio = (width / max(0.02, 1 - 2 * o.side_margin_ratio)) / max(
                    1.0, height / max(0.02, 1 - o.headroom_ratio - o.footroom_ratio))
            allowed = _build_resolutions(self.editor_target, self.source_width / self.source_height,
                                         smart_ratio, o.forced_aspect_ratio, o.resolution_policy)
            if c.resolution not in allowed:
                raise ValueError("Preparation plan output is not an eligible native editor resolution. "
                                 "Generate a new plan rather than changing width/height independently.")
            forced_tier = {"Force 1 MP": "1 MP", "Force 1.5 MP": "1.5 MP",
                           "Force 2K (~4 MP)": "2K (~4 MP)"}.get(o.resolution_policy)
            if forced_tier is not None and c.resolution.tier != forced_tier:
                raise ValueError(f"Preparation plan contradicts resolution_policy={o.resolution_policy!r}.")
        return self


@dataclass(frozen=True)
class PreparationInfo:
    plan: PreparationPlan
    timings_ms: tuple = ()

    def to_dict(self):
        # Fresh serialization only; never expose a mutable live config object.
        p = self.plan
        return {"schema_version": 1, "plan": asdict(p), "resolution": p.resolution,
                "selection_reason": p.reason, "debug": p.debug,
                "transform": {"scale_xy": p.scale_xy, "coordinate_convention": "pixel centers",
                              "crop_xy": (p.crop.x, p.crop.y), "padding_lrtb": p.crop.padding,
                              "padding_fill": p.options.padding_fill},
                "pixel_changes": {"removed": p.source_width * p.source_height - p.crop.width * p.crop.height,
                                  "added": p.crop.added_pixels,
                                  "margin_intrusion_px": p.crop.margin_intrusion_px},
                "warnings": tuple(message for condition, message in (
                    (p.subject_status == "empty", "Subject mask is empty; source-frame framing used."),
                    (p.protected_status == "empty", "Protected mask is empty; no critical region identified."),
                    (any(p.crop.padding), f"Protection required a padded canvas; fill: {p.options.padding_fill}."),
                    (p.crop.margin_intrusion_px > 0, "Configured crop tolerance entered a safety margin; protected pixels remain intact."),
                ) if condition),
                "timings_ms": dict(self.timings_ms)}


@dataclass(frozen=True)
class PreparationResult:
    image: torch.Tensor
    native_crop: torch.Tensor
    mask: torch.Tensor
    info: PreparationInfo

    def legacy_outputs(self):
        p, c = self.info.plan, self.info.plan.crop
        return (self.image, self.native_crop, self.mask, p.target_width, p.target_height,
                c.x, c.y, c.width, c.height, float(c.scale), float(c.crop_loss_percent),
                p.resolution, p.debug)


def plan_photo(width, height, editor_target, analysis, options):
    options.validate()
    integer(width, "source_width", 1, 2**31 - 1)
    integer(height, "source_height", 1, 2**31 - 1)
    target = _unwrap_target(editor_target)
    if target is None:
        raise ValueError("Smart Photo Prepare: no editor target. Connect Active Editor Target "
                         "or choose Direct / No Editor for the bypass path.")
    if options.resolution_policy == "Force 2K (~4 MP)" and target not in (TARGET_QWEN21, TARGET_DIRECT):
        raise ValueError("Smart Photo Prepare: Force 2K (~4 MP) is available only for "
                         "Qwen Image Edit 2.1. Select Auto or a 1/1.5 MP tier for this editor.")
    try:
        crop = _select_crop(width, height, target, analysis, options)
    except ValueError as exc:
        raise ValueError(f"Smart Photo Prepare: cannot fit {width}x{height} for {target} with "
                         f"forced_aspect_ratio={options.forced_aspect_ratio!r}, "
                         f"resolution_policy={options.resolution_policy!r}. {exc}. "
                         "Check the forced aspect (set Auto if accidental), safety margins, and mask alignment. "
                         "The protected region has not been cropped.") from exc
    return PreparationPlan(width, height, target, options, crop, analysis.protected_bbox,
                           analysis.subject_bbox, analysis.note, analysis.subject_status,
                           analysis.protected_status).validate()


@torch.no_grad()
def apply_plan(image, mask, plan, *, resampler=None):
    plan.validate()
    image = normalized(image_tensor(image), "Smart Photo Prepare.image")
    if tuple(image.shape[:3]) != (1, plan.source_height, plan.source_width):
        raise ValueError("Preparation plan source size does not match the input IMAGE. "
                         "Generate a new plan after border removal or any other geometry change.")
    mask = aligned_mask(mask, plan.source_height, plan.source_width, image.device) if mask is not None else None
    return _apply_validated_plan(image, mask, plan, resampler)


def _apply_validated_plan(image, mask, plan, resampler):
    """Internal fast path after boundary validation; no repeated pixel scans."""
    c = plan.crop
    native_crop = image[:, c.y:c.y + c.height, c.x:c.x + c.width, :]
    cropped_mask = mask[:, c.y:c.y + c.height, c.x:c.x + c.width] if mask is not None else None
    canvas, coverage = pad_crop(native_crop, cropped_mask, c.padding, plan.options.padding_fill)
    result = resize_image(canvas, plan.target_width, plan.target_height,
                          plan.options.resize_method, resampler)
    coverage = (resize_mask(coverage, plan.target_width, plan.target_height) if coverage is not None else
                torch.zeros((1, plan.target_height, plan.target_width), device=image.device, dtype=torch.float32))
    return result, native_crop, coverage


@torch.no_grad()
def prepare_photo(image, editor_target, options, mask=None, protected_region_mask=None,
                  *, profile=False, resampler=None):
    boolean(profile, "profile")
    options.validate()
    started = perf_counter() if profile else None
    tensor = normalized(image_tensor(image), "Smart Photo Prepare.image")
    if tensor.shape[0] != 1:
        raise ValueError("Smart Photo Prepare currently expects one image (B=1)")
    height, width = tensor.shape[1:3]
    aligned = aligned_mask(mask, height, width, tensor.device, "subject mask") if mask is not None else None
    protected = (aligned_mask(protected_region_mask, height, width, tensor.device, "protected-region mask")
                 if protected_region_mask is not None else None)
    analyzed = perf_counter() if profile else None
    analysis = analyze_masks(aligned if mask is not None else None, protected, width, height, options)
    planned = perf_counter() if profile else None
    plan = plan_photo(width, height, editor_target, analysis, options)
    applied = perf_counter() if profile else None
    prepared, native_crop, prepared_mask = _apply_validated_plan(tensor, aligned, plan, resampler)
    timings = ()
    if profile:
        finished = perf_counter()
        timings = tuple((name, 1000 * elapsed) for name, elapsed in (
            ("validate_align", analyzed - started), ("mask_analysis", planned - analyzed),
            ("planning", applied - planned), ("apply", finished - applied), ("total_host", finished - started)))
        _LOG.info("Preparation host timings (ms): %s", dict(timings))
    _LOG.debug(plan.debug)
    return PreparationResult(prepared, native_crop, prepared_mask, PreparationInfo(plan, timings))


def _select_crop(image_width, image_height, target, analysis, options):
    smart_crop = options.smart_crop
    resolution_policy = options.resolution_policy
    resize_method = options.resize_method
    invert_mask = options.invert_mask
    q_left = options.q_left
    q_right = options.q_right
    q_top = options.q_top
    q_bottom = options.q_bottom
    min_span_px = options.min_span_px
    headroom_ratio = options.headroom_ratio
    footroom_ratio = options.footroom_ratio
    side_margin_ratio = options.side_margin_ratio
    bottom_priority = options.bottom_priority
    horiz_gravity = options.horiz_gravity
    framing_tolerance_percent = options.framing_tolerance_percent
    forced_aspect_ratio = options.forced_aspect_ratio
    crop_tolerance_px = options.crop_tolerance_px
    bbox = analysis.subject_bbox
    protected_bbox = analysis.protected_bbox
    subject_foreground = analysis.subject_foreground
    forced_parts = FORCED_ASPECTS.get(forced_aspect_ratio)
    forced_ratio = forced_parts[0] / forced_parts[1] if forced_parts is not None else None
    containment = _union_bounds(bbox, protected_bbox)
    if target == TARGET_DIRECT and forced_ratio is None:
        if smart_crop and bbox is not None:
            fitted = _subject_crop(
                image_width, image_height, containment, None,
                headroom_ratio, footroom_ratio, side_margin_ratio,
                bottom_priority, horiz_gravity,
            )
            _, _, crop_width, crop_height, _, _ = fitted
            fitted = _forced_aspect_crop(
                image_width, image_height, crop_width / crop_height,
                bbox, protected_bbox, subject_foreground,
                headroom_ratio, footroom_ratio, side_margin_ratio,
                bottom_priority, horiz_gravity, tight_crop=True,
                crop_size=(crop_width, crop_height), crop_context="direct smart crop",
            )
            if fitted is None:
                x, y, crop_width, crop_height = 0, 0, image_width, image_height
                cost, note = 0.0, "full source frame; protected safety margin"
            else:
                x, y, crop_width, crop_height, cost, note = fitted
        else:
            x, y, crop_width, crop_height = 0, 0, image_width, image_height
            cost, note = 0.0, "full source frame"
        choice = _CropChoice(
            _Resolution(crop_width, crop_height, "native", ASPECT_AUTO),
            x, y, crop_width, crop_height, cost,
            100.0 * (1.0 - crop_width * crop_height / (image_width * image_height)),
            note,
        )
    else:
        smart_ratio = None
        if smart_crop and bbox is not None:
            _, _, bbox_width, bbox_height = containment
            ideal_width = bbox_width / max(0.02, 1.0 - 2.0 * side_margin_ratio)
            ideal_height = bbox_height / max(0.02, 1.0 - headroom_ratio - footroom_ratio)
            smart_ratio = ideal_width / max(1.0, ideal_height)
        if target == TARGET_DIRECT:
            resolutions = (_Resolution(*forced_parts, "native", forced_aspect_ratio),)
        else:
            resolutions = _build_resolutions(
                target, image_width / image_height, smart_ratio, forced_aspect_ratio,
                resolution_policy,
            )
        if forced_ratio is not None:
            forced_crop = _forced_aspect_crop(
                image_width, image_height, forced_ratio, bbox,
                protected_bbox, subject_foreground, headroom_ratio,
                footroom_ratio, side_margin_ratio, bottom_priority,
                horiz_gravity, tight_crop=bool(smart_crop),
            )
            if forced_crop is None and smart_crop:
                forced_crop = _forced_aspect_crop(
                    image_width, image_height, forced_ratio, bbox,
                    protected_bbox, subject_foreground, headroom_ratio,
                    footroom_ratio, side_margin_ratio, bottom_priority,
                    horiz_gravity, tight_crop=False,
                    crop_context="forced aspect; expanded for protection",
                )
            choices = () if forced_crop is None else _forced_aspect_choices(
                resolutions, image_width, image_height, forced_crop
            )
        else:
            choices = _candidate_choices(
                resolutions, image_width, image_height, bbox, protected_bbox,
                subject_foreground, headroom_ratio, footroom_ratio,
                side_margin_ratio, bottom_priority, horiz_gravity,
                tight_crop=bool(smart_crop),
            )
        choices += _limited_crop_choices(
            resolutions, image_width, image_height, bbox, protected_bbox,
            headroom_ratio, footroom_ratio, side_margin_ratio,
            bottom_priority, horiz_gravity, max(0, int(crop_tolerance_px)),
            forced_parts,
        )
        choices += _padded_choices(
            resolutions, image_width, image_height, bbox, protected_bbox,
            headroom_ratio, footroom_ratio, side_margin_ratio,
            bottom_priority, horiz_gravity, bool(smart_crop), forced_parts,
        )
        selection_policy = "Auto (closest scale)" if target == TARGET_DIRECT else resolution_policy
        choice = _choose_resolution(
            choices, selection_policy, framing_tolerance_percent, image_width * image_height
        )
        if target == TARGET_DIRECT:
            left, right, top, bottom = choice.padding
            choice = replace(choice, resolution=_Resolution(
                choice.width + left + right, choice.height + top + bottom,
                "native", forced_aspect_ratio,
            ))

    return choice
