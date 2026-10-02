"""Conservative rectangular photo-panel detection; source pixels are never resized."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import re

import numpy as np
import torch

from .tensors import image_tensor, normalized
from .validation import choice, integer, number, text

DETECTION_MODES = ("Gutters only", "Gutters + strong seams", "Collage (partial seams)", "Manual splits")
SENSITIVITIES = ("Conservative", "Balanced", "Sensitive")
# Color tolerance, line uniformity, minimum two-sided boundary support.
_PROFILES = {"Conservative": (.025, .985, .80),
             "Balanced": (.04, .97, .65), "Sensitive": (.06, .94, .50)}
_LINE_SAMPLES = 256
# Minimum pixel jump, supported fraction, and local peak prominence. The
# collage mode accepts interrupted seams, but only when both ends remain
# supported and the seam is unusual relative to nearby image edges.
_COLLAGE_PROFILES = {"Conservative": (.12, .78, 3.0),
                     "Balanced": (.09, .64, 2.5), "Sensitive": (.07, .52, 2.0)}


@dataclass(frozen=True)
class PanelOptions:
    detection_mode: str = "Gutters only"
    sensitivity: str = "Balanced"
    min_panel_area_percent: float = 3.0
    max_panels: int = 16

    def validate(self):
        choice(self.detection_mode, "Photo Panels.detection_mode", DETECTION_MODES)
        choice(self.sensitivity, "Photo Panels.sensitivity", SENSITIVITIES)
        number(self.min_panel_area_percent, "Photo Panels.min_panel_area_percent", .1, 40)
        integer(self.max_panels, "Photo Panels.max_panels", 2, 64)
        return self


@dataclass(frozen=True)
class PanelBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self):
        return self.width * self.height


@dataclass(frozen=True)
class PanelSeparator:
    axis: str
    start: int
    end: int
    parent: PanelBox
    kind: str
    evidence_score: float


@dataclass(frozen=True)
class PanelResult:
    images: tuple
    boxes: tuple
    separators: tuple
    warnings: tuple
    source_size: tuple
    options: PanelOptions
    manual: bool

    def to_dict(self):
        return {"panel_count": len(self.images), "source_width": self.source_size[0],
                "source_height": self.source_size[1], "manual_override": self.manual,
                "detection_mode": self.options.detection_mode, "sensitivity": self.options.sensitivity,
                "panels": [{"index": i, "number": i + 1, "x": b.x, "y": b.y,
                            "width": b.width, "height": b.height} for i, b in enumerate(self.boxes)],
                "separators": [{"axis": s.axis, "start": s.start, "end": s.end,
                                "kind": s.kind, "evidence_score": round(s.evidence_score, 4),
                                "parent_xywh": [s.parent.x, s.parent.y, s.parent.width, s.parent.height]}
                               for s in self.separators], "warnings": list(self.warnings)}


def _analysis_rgb(samples):
    pixels = samples.detach().to(device="cpu", dtype=torch.float32).numpy()
    if pixels.shape[-1] == 1:
        pixels = np.repeat(pixels, 3, axis=-1)
    elif pixels.shape[-1] == 4:
        pixels = pixels[..., :3] * pixels[..., 3:4] + 1 - pixels[..., 3:4]
    return pixels


def _line_pixels(image, box, axis):
    # Keep EVERY position along the split axis, including one-pixel gutters.
    # Only sample along the perpendicular axis; no full-resolution CPU copy.
    if axis == "x":
        step = max(1, math.ceil(box.height / _LINE_SAMPLES))
        samples = image[0, box.y:box.y + box.height:step, box.x:box.x + box.width]
    else:
        step = max(1, math.ceil(box.width / _LINE_SAMPLES))
        samples = image[0, box.y:box.y + box.height, box.x:box.x + box.width:step].transpose(0, 1)
    return _analysis_rgb(samples)


def _runs(flags):
    transitions = np.diff(np.pad(flags.astype(np.int8), (1, 1)))
    return zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))


def _children(box, axis, start, end):
    if axis == "x":
        return (PanelBox(box.x, box.y, start, box.height),
                PanelBox(box.x + end, box.y, box.width - end, box.height))
    return (PanelBox(box.x, box.y, box.width, start),
            PanelBox(box.x, box.y + end, box.width, box.height - end))


def _collage_candidates(image, pixels, box, axis, sensitivity, min_area):
    """Find straight seams with localized contrast despite short interruptions."""
    length = pixels.shape[1]
    if length < 12:
        return []
    jump_floor, required_support, prominence = _COLLAGE_PROFILES[sensitivity]
    jumps = np.max(np.abs(pixels[:, 1:] - pixels[:, :-1]), axis=-1)
    strength = np.quantile(jumps, .60, axis=0)
    support = np.mean(jumps >= jump_floor, axis=0)
    plausible = (support >= required_support) & (strength >= jump_floor * 1.2)
    result = []
    for start, end in _runs(plausible):
        peak = int(start + np.argmax(strength[start:end]))
        boundary = peak + 1
        children = _children(box, axis, boundary, boundary)
        if any(min(child.width, child.height) < 4 or child.area < min_area for child in children):
            continue
        if any(_blank(image, child, sensitivity) for child in children):
            continue
        hits = jumps[:, peak] >= jump_floor
        end_span = max(1, len(hits) // 4)
        # A line confined to one panel must not split its whole parent box.
        if min(np.mean(hits[:end_span]), np.mean(hits[-end_span:])) < .25:
            continue
        before = strength[max(0, peak - 10):max(0, peak - 2)]
        after = strength[min(len(strength), peak + 3):min(len(strength), peak + 11)]
        surroundings = np.concatenate((before, after))
        if not surroundings.size or strength[peak] < max(
            jump_floor * 1.2, float(np.median(surroundings)) * prominence
        ):
            continue
        score = float(.55 * support[peak] + .30 * min(1, strength[peak] / .3)
                      + .05 * min(np.mean(hits[:end_span]), np.mean(hits[-end_span:])))
        offset = box.x if axis == "x" else box.y
        separator = PanelSeparator(axis, offset + boundary, offset + boundary,
                                   box, "partial seam", score)
        result.append((score, separator, children))
    return result


def _axis_candidates(image, box, axis, options, min_area):
    pixels = _line_pixels(image, box, axis)
    length = pixels.shape[1]
    tolerance, uniformity, boundary_support = _PROFILES[options.sensitivity]
    colors = np.median(pixels, axis=0)
    deviations = np.max(np.abs(pixels - colors[None]), axis=-1)
    uniform = np.mean(deviations <= tolerance, axis=0) >= uniformity
    result = []
    for start, end in _runs(uniform):
        start, end = int(start), int(end)
        if start == 0 or end == length or end - start > max(2, length * .12):
            continue
        children = _children(box, axis, start, end)
        if any(min(child.width, child.height) < 4 or child.area < min_area for child in children):
            continue
        color = np.median(colors[start:end], axis=0)
        context = max(2, min(8, round(length * .01)))
        before = np.median(pixels[:, max(0, start - context):start], axis=1)
        after = np.median(pixels[:, end:min(length, end + context)], axis=1)
        # Both photographs must have a corroborated boundary against the gutter.
        contrast = max(.045, tolerance * 1.5)
        left = float(np.mean(np.max(np.abs(before - color), axis=-1) > contrast))
        right = float(np.mean(np.max(np.abs(after - color), axis=-1) > contrast))
        if min(left, right) < boundary_support:
            continue
        score = min(left, right) + .05 * min(1, (end - start) / 8)
        offset = box.x if axis == "x" else box.y
        result.append((score, PanelSeparator(axis, offset + start, offset + end, box, "gutter", score), children))
    if options.detection_mode == "Gutters + strong seams" and length >= 12:
        jumps = np.max(np.abs(pixels[:, 1:] - pixels[:, :-1]), axis=-1)
        strength = np.median(jumps, axis=0)
        support = np.mean(jumps > .12, axis=0)
        strong = (strength > .18) & (support >= max(.94, uniformity))
        for start, end in _runs(strong):
            boundary = int(start + np.argmax(strength[start:end])) + 1
            children = _children(box, axis, boundary, boundary)
            if any(min(child.width, child.height) < 4 or child.area < min_area for child in children):
                continue
            surroundings = np.concatenate((strength[max(0, boundary - 6):boundary - 1],
                                           strength[boundary:min(length - 1, boundary + 5)]))
            if not surroundings.size or strength[boundary - 1] < max(.18, float(np.median(surroundings)) * 4):
                continue
            score = float(support[boundary - 1]) * .8
            offset = box.x if axis == "x" else box.y
            result.append((score, PanelSeparator(axis, offset + boundary, offset + boundary, box, "strong seam", score), children))
    if options.detection_mode == "Collage (partial seams)":
        result.extend(_collage_candidates(image, pixels, box, axis, options.sensitivity, min_area))
    return result


def _best_separator(image, box, options, min_area):
    if box.area < min_area * 2 or min(box.width, box.height) < 4:
        return None
    candidates = []
    for axis in ("x", "y"):
        candidates.extend(_axis_candidates(image, box, axis, options, min_area))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def _blank(image, box, sensitivity):
    crop = image[0, box.y:box.y + box.height:max(1, math.ceil(box.height / 48)),
                 box.x:box.x + box.width:max(1, math.ceil(box.width / 48))]
    pixels = _analysis_rgb(crop).reshape(-1, 3)
    deviation = np.max(np.abs(pixels - np.median(pixels, axis=0)), axis=-1)
    return bool(np.mean(deviation <= _PROFILES[sensitivity][0]) >= .995)


def _position(token, extent, label):
    try:
        percentage = token.endswith("%")
        value = float(token[:-1] if percentage else token)
    except ValueError as exc:
        raise ValueError(f"{label}: use source pixels or percentages, e.g. '360' or '38.9%'.") from exc
    if not math.isfinite(value) or not 0 <= value <= (100 if percentage else extent):
        raise ValueError(f"{label}: split positions must be inside the image (0–{extent} pixels or 0–100%).")
    if not percentage and not value.is_integer():
        raise ValueError(f"{label}: pixel positions must be whole numbers; use % for fractions.")
    return round(value * extent / 100) if percentage else int(value)


def _manual_intervals(spec, extent, label):
    text(spec, label)
    if len(spec) > 4096:
        raise ValueError(f"{label}: too many split positions (maximum 4096 characters).")
    cuts = []
    for token in re.split(r"[,;\s]+", spec.strip()):
        if not token:
            continue
        parts = token.split(":")
        if len(parts) == 1:
            start = end = _position(parts[0], extent, label)
            if not 0 < start < extent:
                raise ValueError(f"{label}: split lines must be strictly inside the image.")
        elif len(parts) == 2 and all(parts):
            start, end = (_position(part, extent, label) for part in parts)
            if start >= end:
                raise ValueError(f"{label}: excluded gutter ranges must have start < end, e.g. '357:364'.")
        else:
            raise ValueError(f"{label}: use comma-separated lines or gutter ranges, e.g. '357:364, 70%'.")
        cuts.append((start, end))
    cuts = sorted(set(cuts))
    cursor, intervals = 0, []
    for start, end in cuts:
        if start < cursor:
            raise ValueError(f"{label}: gutter ranges overlap; use non-overlapping positions.")
        if start > cursor:
            intervals.append((cursor, start))
        cursor = end
    if cursor < extent:
        intervals.append((cursor, extent))
    return intervals, cuts


@torch.no_grad()
def split_photo_panels(image, options=None, vertical_splits="", horizontal_splits="",
                       column_horizontal_splits=""):
    options = PanelOptions() if options is None else options
    if not isinstance(options, PanelOptions):
        raise TypeError("Photo Panels: options must be a PanelOptions configuration.")
    options.validate()
    image = normalized(image_tensor(image, "Photo Panels.image"), "Photo Panels.image")
    if image.shape[0] != 1:
        raise ValueError("Photo Panels: connect one composite image (B=1), not an existing image batch.")
    text(vertical_splits, "vertical_splits")
    text(horizontal_splits, "horizontal_splits")
    text(column_horizontal_splits, "column_horizontal_splits")
    if len(column_horizontal_splits) > 4096:
        raise ValueError("column_horizontal_splits: maximum 4096 characters.")
    height, width = image.shape[1:3]
    root = PanelBox(0, 0, width, height)
    manual = bool(vertical_splits.strip() or horizontal_splits.strip() or column_horizontal_splits.strip())
    warnings, separators = [], []
    if options.detection_mode == "Manual splits" and not manual:
        raise ValueError("Photo Panels: Manual splits requires vertical_splits or horizontal_splits. "
                         "Use '50%' for a line, '357:364' to exclude a gutter, or choose Gutters only.")
    if manual:
        xs, xcuts = _manual_intervals(vertical_splits, width, "vertical_splits")
        separators = [PanelSeparator("x", start, end, root, "manual", 1)
                      for start, end in xcuts]
        if column_horizontal_splits.strip():
            if not vertical_splits.strip():
                raise ValueError("Photo Panels: column_horizontal_splits requires vertical_splits.")
            if horizontal_splits.strip():
                raise ValueError("Photo Panels: choose horizontal_splits or column_horizontal_splits, not both.")
            specs = column_horizontal_splits.split("|")
            if len(specs) != len(xs):
                raise ValueError("Photo Panels: column_horizontal_splits needs one '|' section per "
                                 f"column ({len(xs)} columns); for example '258,520 | 205,581 | 258,520'.")
            boxes = []
            for column, ((x0, x1), spec) in enumerate(zip(xs, specs), 1):
                ys, ycuts = _manual_intervals(spec, height, f"column_horizontal_splits column {column}")
                parent = PanelBox(x0, 0, x1 - x0, height)
                boxes.extend(PanelBox(x0, y0, x1 - x0, y1 - y0) for y0, y1 in ys)
                separators.extend(PanelSeparator("y", start, end, parent, "manual", 1)
                                  for start, end in ycuts)
            boxes.sort(key=lambda box: (box.y, box.x))
        else:
            ys, ycuts = _manual_intervals(horizontal_splits, height, "horizontal_splits")
            boxes = [PanelBox(x0, y0, x1 - x0, y1 - y0) for y0, y1 in ys for x0, x1 in xs]
            separators.extend(PanelSeparator("y", start, end, root, "manual", 1)
                              for start, end in ycuts)
        count = len(boxes)
        if not count or count > options.max_panels:
            raise ValueError(f"Photo Panels: manual splits produce {count} panels; choose 1–{options.max_panels}. "
                             "Reduce split lines or increase max_panels. No panels have been discarded.")
    else:
        min_area = max(16, root.area * options.min_panel_area_percent / 100)
        leaves, pending, serial = {root}, [], 0
        def enqueue(box):
            nonlocal serial
            candidate = _best_separator(image, box, options, min_area)
            if candidate:
                score, separator, children = candidate
                heapq.heappush(pending, (-score, serial, box, separator, children))
                serial += 1
        enqueue(root)
        while pending and len(leaves) < options.max_panels:
            _, _, parent, separator, children = heapq.heappop(pending)
            leaves.remove(parent)
            leaves.update(children)
            separators.append(separator)
            for child in children:
                enqueue(child)
        if pending:
            warnings.append("max_panels reached: remaining regions are kept unsplit, not discarded.")
        boxes = sorted(leaves, key=lambda b: (b.y, b.x))
        if len(boxes) > 1:
            nonblank = [box for box in boxes if not _blank(image, box, options.sensitivity)]
            if nonblank:
                if len(nonblank) != len(boxes):
                    warnings.append("Uniform empty cells were omitted.")
                boxes = nonblank
            else:
                boxes = [root]
                separators = []
        if len(boxes) == 1 and boxes[0] == root:
            warnings.append("No supported panel separators detected; the whole input is retained. "
                            "Use manual splits for ambiguous, borderless, rotated, or overlapping layouts.")
        if options.detection_mode == "Gutters + strong seams":
            warnings.append("Strong-seam detection can mistake scene boundaries for panels; verify the thumbnails.")
        elif options.detection_mode == "Collage (partial seams)":
            warnings.append("Partial-seam detection can mistake straight scene edges for panels; verify the thumbnails.")
    crops = tuple(image if box == root else image[:, box.y:box.y + box.height, box.x:box.x + box.width, :]
                  for box in boxes)
    return PanelResult(crops, tuple(boxes), tuple(separators), tuple(warnings), (width, height), options, manual)
