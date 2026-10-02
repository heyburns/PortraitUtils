"""Pure crop planning in source-pixel coordinates. No tensor transforms."""
from __future__ import annotations
import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np
from .editor_profiles import (TIER_1MP, TIER_15MP, TIER_2K, FORCED_ASPECTS, _Resolution)

@dataclass(frozen=True)
class _CropChoice:
    resolution: _Resolution
    x: int
    y: int
    width: int
    height: int
    framing_cost: float
    crop_loss_percent: float
    margin_note: str = ""
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)  # left, right, top, bottom
    margin_intrusion_px: int = 0

    @property
    def added_pixels(self) -> int:
        left, right, top, bottom = self.padding
        return (self.width + left + right) * (self.height + top + bottom) - self.width * self.height

    def changed_pixels(self, source_pixels: int) -> int:
        return source_pixels - self.width * self.height + self.added_pixels

    @property
    def scale(self) -> float:
        canvas_area = self.width * self.height + self.added_pixels
        return math.sqrt(self.resolution.area / max(1, canvas_area))

def _full_frame_crop(width: int, height: int, ratio: float, gravity: str) -> Tuple[int, int, int, int]:
    """Largest integer crop at the requested ratio, allowing only rounding-level distortion."""
    if width / height > ratio:
        crop_h = height
        crop_w = min(width, max(1, round(height * ratio)))
    else:
        crop_w = width
        crop_h = min(height, max(1, round(width / ratio)))

    if gravity == "left":
        x = 0
    elif gravity == "right":
        x = width - crop_w
    else:
        x = (width - crop_w) // 2
    y = (height - crop_h) // 2
    return int(x), int(y), int(crop_w), int(crop_h)

def _full_frame_forced_crop(
    width: int, height: int, ratio: float, gravity: str
) -> Tuple[int, int, int, int]:
    """Largest integer crop at one of the configured mathematically exact ratios."""
    numerator, denominator = min(
        FORCED_ASPECTS.values(),
        key=lambda parts: abs(parts[0] / parts[1] - ratio),
    )
    scale = min(width // numerator, height // denominator)
    if scale < 1:
        raise ValueError("Source image is too small for the requested aspect ratio")
    crop_width = numerator * scale
    crop_height = denominator * scale
    if gravity == "left":
        x = 0
    elif gravity == "right":
        x = width - crop_width
    else:
        x = (width - crop_width) // 2
    y = (height - crop_height) // 2
    return x, y, crop_width, crop_height

def _safe_component_bounds(
    foreground: np.ndarray, quantile_bbox: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Keep every marked subject, including small disconnected components.

    Quantiles may include faint mask edges, but cannot exclude another subject
    or a thin extremity. Mask cleanup belongs upstream of this node.
    """
    return _union_bounds(quantile_bbox, _foreground_bounds(foreground))

def _union_bounds(first, second):
    if first is None:
        return second
    if second is None:
        return first
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x, y = min(ax, bx), min(ay, by)
    return x, y, max(ax + aw, bx + bw) - x, max(ay + ah, by + bh) - y

def _foreground_bounds(foreground: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return hard bounds for every meaningful pixel in a clean priority mask."""
    max_value = float(foreground.max()) if foreground.size else 0.0
    if max_value <= 0.0:
        return None
    threshold = max(1e-4, min(0.25, max_value * 0.1))
    ys, xs = np.nonzero(foreground >= threshold)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0

def _pad_bbox(
    bbox: Tuple[int, int, int, int],
    pad: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x, y, width, height = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_width, x + width + pad)
    y1 = min(image_height, y + height + pad)
    return x0, y0, x1 - x0, y1 - y0

def _best_axis_window(
    mass: np.ndarray,
    crop_length: int,
    low: int,
    high: int,
    desired: float,
    mass_first: bool = False,
) -> int:
    """Choose a crop origin inside a prevalidated safe interval."""
    low = max(0, int(math.ceil(low)))
    high = min(len(mass) - crop_length, int(math.floor(high)))
    if low > high:
        raise ValueError("Protected region cannot fit inside the requested aspect crop")
    origins = np.arange(low, high + 1, dtype=np.int64)
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(mass)))
    retained = prefix[origins + crop_length] - prefix[origins]
    distance = np.abs(origins.astype(np.float64) - desired)

    if mass_first:
        best_retained = float(retained.max())
        candidates = np.flatnonzero(
            np.isclose(retained, best_retained, rtol=1e-9, atol=1e-9)
        )
        candidate_distance = distance[candidates]
        return int(origins[candidates[int(np.argmin(candidate_distance))]])

    best_distance = float(distance.min())
    safest = np.flatnonzero(np.isclose(distance, best_distance, atol=1e-9))
    if safest.size == 1:
        return int(origins[safest[0]])
    tie_mass = retained[safest]
    return int(origins[safest[int(np.argmax(tie_mass))]])

def _protected_axis_origin(
    image_length: int,
    crop_length: int,
    protected_start: int,
    protected_length: int,
) -> float:
    """Return the preferred anchor nearest the protected region's source edge.

    The actual origin may move inward when that preserves more of the subject,
    but it can never move beyond the padded protected-region boundary.
    """
    max_origin = max(0, image_length - crop_length)
    near_margin = protected_start
    far_margin = image_length - (protected_start + protected_length)
    if near_margin < far_margin:
        return 0.0
    if far_margin < near_margin:
        return float(max_origin)
    return max_origin / 2.0

def _protected_safety_axis_interval(
    image_length: int,
    crop_length: int,
    protected_start: int,
    protected_length: int,
    margin_ratio: float,
) -> Tuple[int, int]:
    """Return crop origins that preserve outward safety around a priority mask.

    The source edge nearest the protected region is treated as its outward
    direction. This makes head protection work for upright, rotated, and
    intentionally inverted compositions. When the full requested margin will
    not fit, keep the greatest outward margin the source and crop can supply.
    """
    max_origin = max(0, image_length - crop_length)
    protected_end = protected_start + protected_length
    low = max(0, protected_end - crop_length)
    high = min(max_origin, protected_start)
    if low > high:
        raise ValueError("Protected region cannot fit inside the requested aspect crop")

    requested = max(0, math.ceil(float(margin_ratio) * crop_length))
    before = max(0, protected_start)
    after = max(0, image_length - protected_end)

    if before < after:
        safe_high = protected_start - requested
        if safe_high >= low:
            high = min(high, safe_high)
        else:
            high = low
    elif after < before:
        safe_low = protected_end + requested - crop_length
        if safe_low <= high:
            low = max(low, safe_low)
        else:
            low = high
    else:
        safe_low = protected_end + requested - crop_length
        safe_high = protected_start - requested
        narrowed_low = max(low, safe_low)
        narrowed_high = min(high, safe_high)
        if narrowed_low <= narrowed_high:
            low, high = narrowed_low, narrowed_high

    return int(low), int(high)

def _margin_axis_interval(
    image_length: int,
    crop_length: int,
    subject_start: int,
    subject_length: int,
    near_margin_ratio: float,
    far_margin_ratio: float,
) -> Optional[Tuple[int, int]]:
    """Return origins that preserve every margin available inside the source.

    A requested margin is a safety preference, but an edge-touching subject may
    make that margin physically impossible. Clamp only that side's requirement
    to the source pixels that actually exist; do not discard achievable margins
    on the other three sides.
    """
    subject_end = subject_start + subject_length
    near_available = max(0, subject_start)
    far_available = max(0, image_length - subject_end)
    near_required = min(
        near_available,
        max(0, math.ceil(float(near_margin_ratio) * crop_length)),
    )
    far_required = min(
        far_available,
        max(0, math.ceil(float(far_margin_ratio) * crop_length)),
    )
    low = max(0, subject_end + far_required - crop_length)
    high = min(image_length - crop_length, subject_start - near_required)
    if low > high:
        return None
    return int(low), int(high)

def _tight_forced_aspect_size(
    image_width: int,
    image_height: int,
    bbox: Tuple[int, int, int, int],
    numerator: int,
    denominator: int,
    headroom: float,
    footroom: float,
    side_margin: float,
) -> Optional[Tuple[int, int, str]]:
    """Find the smallest exact-aspect crop with all achievable safety margins."""
    bx, by, bw, bh = bbox
    min_scale = max(
        1,
        math.ceil(bw / numerator),
        math.ceil(bh / denominator),
    )
    max_scale = min(image_width // numerator, image_height // denominator)
    for scale in range(min_scale, max_scale + 1):
        crop_width = numerator * scale
        crop_height = denominator * scale
        x_interval = _margin_axis_interval(
            image_width,
            crop_width,
            bx,
            bw,
            side_margin,
            side_margin,
        )
        y_interval = _margin_axis_interval(
            image_height,
            crop_height,
            by,
            bh,
            headroom,
            footroom,
        )
        if x_interval is None or y_interval is None:
            continue

        relaxed_edges = []
        requested_left = side_margin * crop_width
        requested_right = side_margin * crop_width
        requested_top = headroom * crop_height
        requested_bottom = footroom * crop_height
        if bx + 1e-9 < requested_left:
            relaxed_edges.append("left")
        if image_width - (bx + bw) + 1e-9 < requested_right:
            relaxed_edges.append("right")
        if by + 1e-9 < requested_top:
            relaxed_edges.append("top")
        if image_height - (by + bh) + 1e-9 < requested_bottom:
            relaxed_edges.append("bottom")
        if relaxed_edges:
            note = "source-edge margin relaxed: " + ", ".join(relaxed_edges)
        else:
            note = "all configured margins retained"
        return crop_width, crop_height, note
    return None

def _forced_aspect_crop(
    image_width: int,
    image_height: int,
    ratio: float,
    subject_bbox: Optional[Tuple[int, int, int, int]],
    protected_bbox: Optional[Tuple[int, int, int, int]],
    subject_foreground: Optional[np.ndarray],
    headroom: float,
    footroom: float,
    side_margin: float,
    bottom_priority: float,
    gravity: str,
    tight_crop: bool = False,
    crop_size: Optional[Tuple[int, int]] = None,
    crop_context: str = "forced aspect",
) -> Optional[Tuple[int, int, int, int, float, str]]:
    """Place a crop containing all subjects; return None when it must expand."""
    containment = _union_bounds(subject_bbox, protected_bbox)
    if crop_size is None:
        numerator, denominator = min(
            FORCED_ASPECTS.values(), key=lambda parts: abs(parts[0] / parts[1] - ratio)
        )
        if image_width < numerator or image_height < denominator:
            return None
        _, _, crop_width, crop_height = _full_frame_forced_crop(
            image_width, image_height, ratio, gravity
        )
        if tight_crop and subject_bbox is not None:
            tight_size = _tight_forced_aspect_size(
                image_width, image_height, containment, numerator, denominator,
                headroom, footroom, side_margin,
            )
            if tight_size is not None:
                crop_width, crop_height, margin_note = tight_size
                crop_context = f"tight {crop_context}; {margin_note}"
    else:
        crop_width, crop_height = crop_size

    if containment is None:
        if gravity == "left":
            x = 0
        elif gravity == "right":
            x = image_width - crop_width
        else:
            x = (image_width - crop_width) // 2
        return x, (image_height - crop_height) // 2, crop_width, crop_height, 0.0, crop_context

    bx, by, bw, bh = containment
    if crop_width < bw or crop_height < bh:
        return None
    placement = _place_subject_crop(
        image_width, image_height, containment, crop_width, crop_height,
        headroom, footroom, side_margin, bottom_priority, gravity,
    )
    if placement is None:
        return None
    x, y, margin_loss, margin_note = placement
    if protected_bbox is not None:
        px, py, pw, ph = protected_bbox
        x_low, x_high = _margin_axis_interval(
            image_width, crop_width, bx, bw, side_margin, side_margin
        )
        y_low, y_high = _margin_axis_interval(
            image_height, crop_height, by, bh, headroom, footroom
        )
        safe_x = _protected_safety_axis_interval(
            image_width, crop_width, px, pw, side_margin
        )
        safe_y = _protected_safety_axis_interval(
            image_height, crop_height, py, ph, headroom
        )
        x_low, x_high = max(x_low, safe_x[0]), min(x_high, safe_x[1])
        y_low, y_high = max(y_low, safe_y[0]), min(y_high, safe_y[1])
        if x_low > x_high or y_low > y_high:
            return None
        if not tight_crop:
            x = _protected_axis_origin(image_width, crop_width, px, pw)
            y = _protected_axis_origin(image_height, crop_height, py, ph)
        if subject_foreground is None:
            column_mass, row_mass = np.zeros(image_width), np.zeros(image_height)
        else:
            column_mass = subject_foreground.sum(axis=0, dtype=np.float64)
            row_mass = subject_foreground.sum(axis=1, dtype=np.float64)
        x = _best_axis_window(column_mass, crop_width, x_low, x_high, x, mass_first=True)
        y = _best_axis_window(row_mass, crop_height, y_low, y_high, y, mass_first=True)
        margin_note += "; protected safety margin (best available)"
    cost = 400.0 * margin_loss / max(1, crop_width + crop_height)
    return x, y, crop_width, crop_height, cost, f"{crop_context}; full subject retained; {margin_note}"

def _smallest_ratio_size(min_width: float, min_height: float, ratio: float) -> Tuple[int, int]:
    candidates = []

    crop_h = max(1, math.ceil(max(min_height, min_width / ratio)))
    crop_w = max(1, math.ceil(crop_h * ratio))
    candidates.append((crop_w, crop_h))

    crop_w = max(1, math.ceil(max(min_width, min_height * ratio)))
    crop_h = max(1, math.ceil(crop_w / ratio))
    candidates.append((crop_w, crop_h))

    valid = [(w, h) for w, h in candidates if w >= min_width and h >= min_height]
    return min(valid, key=lambda size: (size[0] * size[1], size[0] + size[1]))

def _clamp_to_interval(value: float, low: float, high: float) -> float:
    if low > high:
        return low
    return max(low, min(high, value))

def _place_subject_crop(
    image_width: int,
    image_height: int,
    bbox: Tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
    headroom: float,
    footroom: float,
    side_margin: float,
    bottom_priority: float,
    gravity: str,
) -> Optional[Tuple[int, int, float, str]]:
    bx, by, bw, bh = bbox
    bx1, by1 = bx + bw, by + bh
    x_interval = _margin_axis_interval(
        image_width, crop_width, bx, bw, side_margin, side_margin
    )
    y_interval = _margin_axis_interval(
        image_height, crop_height, by, bh, headroom, footroom
    )
    if x_interval is None or y_interval is None:
        return None

    if gravity == "left":
        desired_x = 0.0
    elif gravity == "right":
        desired_x = float(image_width - crop_width)
    else:
        desired_x = bx + bw / 2.0 - crop_width / 2.0
    x = int(round(_clamp_to_interval(desired_x, *x_interval)))

    desired_top = by - headroom * crop_height
    desired_bottom = by1 + footroom * crop_height - crop_height
    desired_y = (1.0 - bottom_priority) * desired_top + bottom_priority * desired_bottom
    y = int(round(_clamp_to_interval(desired_y, *y_interval)))

    left_loss = max(0.0, side_margin * crop_width - (bx - x))
    right_loss = max(0.0, side_margin * crop_width - (x + crop_width - bx1))
    top_loss = max(0.0, headroom * crop_height - (by - y))
    bottom_loss = max(0.0, footroom * crop_height - (y + crop_height - by1))
    total_margin_loss = left_loss + right_loss + top_loss + bottom_loss
    note = (
        f"margin_shortfall_px L/R/T/B={left_loss:.1f}/{right_loss:.1f}/"
        f"{top_loss:.1f}/{bottom_loss:.1f}"
    )
    return x, y, float(total_margin_loss), note

def _subject_crop(
    image_width: int,
    image_height: int,
    bbox: Tuple[int, int, int, int],
    ratio: Optional[float],
    headroom: float,
    footroom: float,
    side_margin: float,
    bottom_priority: float,
    gravity: str,
) -> Optional[Tuple[int, int, int, int, float, str]]:
    bx, by, bw, bh = bbox
    ideal_width = bw / max(0.02, 1.0 - 2.0 * side_margin)
    ideal_height = bh / max(0.02, 1.0 - headroom - footroom)

    margins_relaxed = False
    if ratio is None:
        crop_width, crop_height = math.ceil(ideal_width), math.ceil(ideal_height)
    else:
        crop_width, crop_height = _smallest_ratio_size(ideal_width, ideal_height, ratio)

    if crop_width > image_width or crop_height > image_height:
        margins_relaxed = True
        if ratio is None:
            # Preserve every available margin independently.  Collapsing both
            # dimensions to the raw bbox made over-constrained crops needlessly
            # tight even when most of the surrounding photo was available.
            crop_width = min(image_width, max(bw, math.ceil(ideal_width)))
            crop_height = min(image_height, max(bh, math.ceil(ideal_height)))
        else:
            # The ideal padded crop does not fit. Use the largest crop of this
            # model aspect that the source can supply, provided it contains the
            # complete subject. This retains maximum safety/context instead of
            # falling all the way back to a tight raw-subject rectangle.
            _, _, crop_width, crop_height = _full_frame_crop(
                image_width, image_height, ratio, "center"
            )

    placement = None
    while crop_width <= image_width and crop_height <= image_height:
        x_interval = _margin_axis_interval(
            image_width, crop_width, bx, bw, side_margin, side_margin
        )
        y_interval = _margin_axis_interval(
            image_height, crop_height, by, bh, headroom, footroom
        )
        if x_interval is not None and y_interval is not None:
            placement = _place_subject_crop(
                image_width, image_height, bbox, crop_width, crop_height,
                headroom, footroom, side_margin, bottom_priority, gravity,
            )
            break
        min_width = crop_width + int(x_interval is None)
        min_height = crop_height + int(y_interval is None)
        if ratio is None:
            crop_width, crop_height = min_width, min_height
        else:
            crop_width, crop_height = _smallest_ratio_size(min_width, min_height, ratio)
    if placement is None:
        return None
    x, y, margin_loss, margin_note = placement
    ideal_area = max(1.0, ideal_width * ideal_height)
    overhead_percent = max(0.0, (crop_width * crop_height / ideal_area - 1.0) * 100.0)
    margin_loss_percent = 100.0 * margin_loss / max(1.0, crop_width + crop_height)
    # Missing a requested subject margin is materially worse than retaining a
    # little extra background. This particularly protects heads, hair, hands,
    # and feet when comparing nearby native aspect buckets.
    framing_cost = overhead_percent + 4.0 * margin_loss_percent
    if margins_relaxed:
        margin_note += "; margins relaxed to fit source frame"
    return x, y, crop_width, crop_height, framing_cost, margin_note

def _candidate_choices(
    resolutions: Iterable[_Resolution],
    image_width: int,
    image_height: int,
    bbox: Optional[Tuple[int, int, int, int]],
    protected_bbox: Optional[Tuple[int, int, int, int]],
    subject_foreground: Optional[np.ndarray],
    headroom: float,
    footroom: float,
    side_margin: float,
    bottom_priority: float,
    gravity: str,
    tight_crop: bool,
) -> Tuple[_CropChoice, ...]:
    choices = []
    image_area = image_width * image_height
    containment = _union_bounds(bbox, protected_bbox)
    for resolution in resolutions:
        framing_cost = 0.0
        if tight_crop and bbox is not None:
            fitted = _subject_crop(
                image_width, image_height, containment, resolution.ratio,
                headroom, footroom, side_margin, bottom_priority, gravity,
            )
            if fitted is None:
                continue
            _, _, crop_width, crop_height, framing_cost, _ = fitted
        else:
            _, _, crop_width, crop_height = _full_frame_crop(
                image_width, image_height, resolution.ratio, gravity
            )
        fitted = _forced_aspect_crop(
            image_width, image_height, resolution.ratio, bbox, protected_bbox,
            subject_foreground, headroom, footroom, side_margin,
            bottom_priority, gravity, tight_crop=tight_crop,
            crop_size=(crop_width, crop_height),
            crop_context=f"native aspect {resolution.family}",
        )
        if fitted is None and tight_crop:
            _, _, full_width, full_height = _full_frame_crop(
                image_width, image_height, resolution.ratio, gravity
            )
            fitted = _forced_aspect_crop(
                image_width, image_height, resolution.ratio, bbox, protected_bbox,
                subject_foreground, headroom, footroom, side_margin,
                bottom_priority, gravity, tight_crop=True,
                crop_size=(full_width, full_height),
                crop_context=f"native aspect {resolution.family}; expanded for protection",
            )
            framing_cost += 100.0 * (full_width * full_height / (crop_width * crop_height) - 1.0)
        if fitted is None:
            continue
        x, y, crop_width, crop_height, _, margin_note = fitted
        crop_loss = 100.0 * (1.0 - crop_width * crop_height / image_area)
        if not tight_crop or bbox is None:
            framing_cost = crop_loss
        choices.append(_CropChoice(
            resolution, x, y, crop_width, crop_height,
            max(0.0, framing_cost), max(0.0, crop_loss), margin_note,
        ))
    return tuple(choices)

def _limited_crop_choices(
    resolutions: Iterable[_Resolution],
    image_width: int,
    image_height: int,
    bbox: Optional[Tuple[int, int, int, int]],
    protected_bbox: Optional[Tuple[int, int, int, int]],
    headroom: float,
    footroom: float,
    side_margin: float,
    bottom_priority: float,
    gravity: str,
    tolerance_px: int,
    forced_parts: Optional[Tuple[int, int]] = None,
) -> Tuple[_CropChoice, ...]:
    """Offer minimal-loss crops with bounded intrusion into available margins."""
    containment = _union_bounds(bbox, protected_bbox)
    if tolerance_px <= 0 or containment is None:
        return ()
    bx, by, bw, bh = containment

    def place_axis(image_length, crop_length, start, length, near_ratio, far_ratio,
                   desired, protected):
        end = start + length
        near_required = min(start, math.ceil(near_ratio * crop_length))
        far_required = min(image_length - end, math.ceil(far_ratio * crop_length))
        safe_low = end + far_required - crop_length
        safe_high = start - near_required
        hard_low, hard_high = 0, image_length - crop_length
        if protected is not None:
            protected_start, protected_length, protected_margin = protected
            if protected_length > crop_length:
                return None
            protected_low, protected_high = _protected_safety_axis_interval(
                image_length, crop_length, protected_start, protected_length,
                protected_margin,
            )
            safe_low, safe_high = max(safe_low, protected_low), min(safe_high, protected_high)
            hard_low = max(hard_low, protected_start + protected_length - crop_length)
            hard_high = min(hard_high, protected_start)
        low = max(hard_low, safe_low - tolerance_px)
        high = min(hard_high, safe_high + tolerance_px)
        if low > high:
            return None
        origins = np.arange(low, high + 1, dtype=np.int64)
        near_loss = np.maximum(0, origins - safe_high)
        far_loss = np.maximum(0, safe_low - origins)
        order = np.lexsort((np.abs(origins - desired), near_loss + far_loss,
                            np.maximum(near_loss, far_loss)))
        index = int(order[0])
        return int(origins[index]), int(near_loss[index]), int(far_loss[index])

    choices = []
    for resolution in resolutions:
        if forced_parts is not None:
            numerator, denominator = forced_parts
            scale = min(image_width // numerator, image_height // denominator)
            if scale < 1:
                continue
            width, height = numerator * scale, denominator * scale
        else:
            _, _, width, height = _full_frame_crop(
                image_width, image_height, resolution.ratio, gravity
            )
        if gravity == "left":
            desired_x = 0
        elif gravity == "right":
            desired_x = image_width - width
        else:
            desired_x = bx + (bw - width) / 2
        desired_y = ((1.0 - bottom_priority) * (by - headroom * height)
                     + bottom_priority * (by + bh + footroom * height - height))
        protected_x = protected_y = None
        if protected_bbox is not None:
            px, py, pw, ph = protected_bbox
            protected_x, protected_y = (px, pw, side_margin), (py, ph, headroom)
        placed_x = place_axis(image_width, width, bx, bw, side_margin, side_margin,
                              desired_x, protected_x)
        placed_y = place_axis(image_height, height, by, bh, headroom, footroom,
                              desired_y, protected_y)
        if placed_x is None or placed_y is None:
            continue
        x, left_loss, right_loss = placed_x
        y, top_loss, bottom_loss = placed_y
        intrusion = max(left_loss, right_loss, top_loss, bottom_loss)
        if intrusion == 0:
            # Safe crops are already generated with the requested framing mode.
            continue
        note = (f"limited crop; margin_intrusion_px L/R/T/B="
                f"{left_loss}/{right_loss}/{top_loss}/{bottom_loss}")
        if protected_bbox is not None:
            note += "; protected region retained"
        loss = 100.0 * (1.0 - width * height / (image_width * image_height))
        choices.append(_CropChoice(
            resolution, x, y, width, height, loss, loss, note,
            margin_intrusion_px=intrusion,
        ))
    return tuple(choices)

def _padded_choices(
    resolutions: Iterable[_Resolution],
    image_width: int,
    image_height: int,
    bbox: Optional[Tuple[int, int, int, int]],
    protected_bbox: Optional[Tuple[int, int, int, int]],
    headroom: float,
    footroom: float,
    side_margin: float,
    bottom_priority: float,
    gravity: str,
    tight_crop: bool,
    forced_parts: Optional[Tuple[int, int]] = None,
) -> Tuple[_CropChoice, ...]:
    """Expand the subject frame to a native aspect, using source pixels first."""
    containment = _union_bounds(bbox, protected_bbox)
    if tight_crop and bbox is not None:
        fitted = _subject_crop(
            image_width, image_height, containment, None,
            headroom, footroom, side_margin, bottom_priority, gravity,
        )
        bx, by, bw, bh, _, _ = fitted
        if protected_bbox is not None:
            fitted = _forced_aspect_crop(
                image_width, image_height, bw / bh, bbox, protected_bbox, None,
                headroom, footroom, side_margin, bottom_priority, gravity,
                tight_crop=True, crop_size=(bw, bh),
            )
            if fitted is None:
                bx, by, bw, bh = 0, 0, image_width, image_height
            else:
                bx, by, bw, bh, _, _ = fitted
    else:
        bx, by, bw, bh = 0, 0, image_width, image_height

    def canvas_axis(start, length, canvas_length, source_length, alignment):
        if canvas_length >= source_length:
            low, high = source_length - canvas_length, 0
        else:
            low = max(0, start + length - canvas_length)
            high = min(source_length - canvas_length, start)
        if alignment == "left":
            origin = high if canvas_length >= source_length else low
        elif alignment == "right":
            origin = low if canvas_length >= source_length else high
        else:
            if canvas_length >= source_length:
                desired = (source_length - canvas_length) / 2
            else:
                desired = start + (length - canvas_length) / 2
            origin = int(round(_clamp_to_interval(desired, low, high)))
        source_start = max(0, origin)
        source_end = min(source_length, origin + canvas_length)
        return (
            source_start, source_end - source_start,
            max(0, -origin), max(0, origin + canvas_length - source_length),
        )

    choices = []
    for resolution in resolutions:
        if forced_parts is None:
            canvas_width, canvas_height = _smallest_ratio_size(bw, bh, resolution.ratio)
        else:
            numerator, denominator = forced_parts
            scale = max(math.ceil(bw / numerator), math.ceil(bh / denominator))
            canvas_width, canvas_height = numerator * scale, denominator * scale
        x, width, left, right = canvas_axis(bx, bw, canvas_width, image_width, gravity)
        y, height, top, bottom = canvas_axis(by, bh, canvas_height, image_height, "center")
        padding = left, right, top, bottom
        if not any(padding):
            continue
        padding_percent = 100.0 * (1.0 - width * height / (canvas_width * canvas_height))
        crop_loss = 100.0 * (1.0 - width * height / (image_width * image_height))
        note = f"edge padding L/R/T/B={left}/{right}/{top}/{bottom}; full subject retained"
        if protected_bbox is not None:
            note += "; protected region retained"
        choices.append(_CropChoice(
            resolution, x, y, width, height,
            padding_percent, crop_loss, note, padding,
        ))
    return tuple(choices)

def _forced_aspect_choices(
    resolutions: Iterable[_Resolution],
    image_width: int,
    image_height: int,
    crop: Tuple[int, int, int, int, float, str],
) -> Tuple[_CropChoice, ...]:
    x, y, crop_width, crop_height, framing_cost, note = crop
    image_area = max(1, image_width * image_height)
    crop_loss = 100.0 * (1.0 - crop_width * crop_height / image_area)
    return tuple(
        _CropChoice(
            resolution=resolution,
            x=x,
            y=y,
            width=crop_width,
            height=crop_height,
            framing_cost=max(0.0, float(framing_cost)),
            crop_loss_percent=max(0.0, min(100.0, float(crop_loss))),
            margin_note=note,
        )
        for resolution in resolutions
    )

def _choose_resolution(
    choices: Sequence[_CropChoice], policy: str, framing_tolerance_percent: float,
    source_pixels: int,
) -> _CropChoice:
    if "Force 1 MP" in policy:
        choices = tuple(choice for choice in choices if choice.resolution.tier == TIER_1MP)
    elif "Force 1.5 MP" in policy:
        choices = tuple(choice for choice in choices if choice.resolution.tier == TIER_15MP)
    elif "Force 2K (~4 MP)" in policy:
        choices = tuple(choice for choice in choices if choice.resolution.tier == TIER_2K)

    if not choices:
        raise ValueError("No compatible crop fits this image and editor target")

    safe_crops = tuple(
        choice for choice in choices
        if not any(choice.padding) and choice.margin_intrusion_px == 0
    )
    if safe_crops:
        choices = safe_crops
    else:
        # Only compare cropping against padding when no safe crop fits.
        least_change = min(choice.changed_pixels(source_pixels) for choice in choices)
        choices = tuple(
            choice for choice in choices if choice.changed_pixels(source_pixels) == least_change
        )
        unpadded = tuple(choice for choice in choices if not any(choice.padding))
        if unpadded:
            choices = unpadded

    best_framing = min(choice.framing_cost for choice in choices)
    pool = tuple(
        choice
        for choice in choices
        if choice.framing_cost <= best_framing + max(0.0, framing_tolerance_percent)
    )

    if policy == "Auto (preserve detail)":
        non_downscales = tuple(choice for choice in pool if choice.scale >= 1.0 - 1e-9)
        if non_downscales:
            return min(
                non_downscales,
                key=lambda choice: (choice.scale, choice.framing_cost, choice.resolution.area),
            )
        return max(
            pool,
            key=lambda choice: (choice.scale, -choice.framing_cost, choice.resolution.area),
        )

    # Closest-scale and forced-tier modes both minimize the actual resize.
    return min(
        pool,
        key=lambda choice: (
            abs(math.log(max(choice.scale, 1e-9))),
            0 if choice.scale >= 1.0 else 1,
            choice.framing_cost,
            -choice.resolution.area,
        ),
    )
