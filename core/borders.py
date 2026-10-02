"""Conservative solid-border and hosting-banner removal engine."""
from __future__ import annotations
import logging
import math
import cv2
import numpy as np
import torch
from .contracts import CROP_CONFIG_TYPE, CropConfigV2, _expect, validate_config
from .tensors import image_tensor

POLICIES = ["Conservative", "Configured tolerances"]

_LOG = logging.getLogger(__name__)

def _first_failure(matches):
    failed = np.flatnonzero(~matches)
    # A run with no observed end is an ambiguous solid image, not a border.
    return int(failed[0]) if failed.size else 0

def _boundary_fraction(lines, depth, tolerance):
    """Require an actual seam, not an anchored-color cutoff in a smooth gradient."""
    if depth <= 0 or depth >= len(lines):
        return 0.0
    before = lines[depth - 1]
    # A cumulative eight-row difference mistakes a short smooth gradient for a
    # seam on the pass after its real frame is removed. An actual photo edge
    # changes abruptly in at least one adjacent row (possibly after a tiny
    # subject tip or a few JPEG-ringing pixels).
    after = lines[depth:min(len(lines), depth + 8)]
    adjacent = np.diff(np.concatenate((before[None], after)), axis=0)
    differences = np.max(np.abs(adjacent), axis=2)
    return float(np.max(np.mean(differences > max(0.035, tolerance * 0.8), axis=1)))

def _border_depth(lines, tolerance, uniformity, conservative):
    """Measure a contiguous, stable, edge-connected color plateau."""
    limit = (len(lines) - 1) // 2
    if limit < 1:
        return 0
    base = np.median(lines[0], axis=0)
    outer_distance = np.max(np.abs(lines[0] - base), axis=1)
    if float(np.mean(outer_distance <= tolerance + 1e-6)) < uniformity:
        return 0
    strong_limit = max(0.08, tolerance * 1.5)
    if conservative and float(np.max(outer_distance)) > strong_limit:
        return 0
    # Fixed edge color: never adapt the border color to the photograph or jump
    # over a nonmatching graphic/object in search of more matching pixels.
    region = lines[:min(len(lines), limit + 8)]
    distance = np.max(np.abs(region - base), axis=2)
    matches = np.mean(distance[:limit] <= tolerance + 1e-6, axis=1) >= uniformity
    if conservative:
        noise = float(np.mean(distance[0]))
        matches &= np.mean(distance[:limit], axis=1) <= max(0.006, tolerance * 0.3 + noise * 1.5)
        # A tiny bright head/hair/object tip must stop a solid-border scan even
        # when it occupies less than the allowed nonmatching pixel fraction.
        matches &= np.max(distance[:limit], axis=1) <= strong_limit + 1e-6
    depth = _first_failure(matches)
    if depth and _boundary_fraction(region, depth, tolerance) >= 0.20:
        return depth
    return 0

def _text_like(bright):
    """Small, aligned bright components support a footer hypothesis, not OCR."""
    height, width = bright.shape
    count, _, stats, centers = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(bright, dtype=np.uint8), connectivity=8)
    if count < 3:
        return False
    stats, centers = stats[1:], centers[1:]
    good = ((stats[:, cv2.CC_STAT_AREA] >= max(2, int(height * width * 0.00001))) &
            (stats[:, cv2.CC_STAT_HEIGHT] >= 2) &
            (stats[:, cv2.CC_STAT_HEIGHT] <= height * 0.85) &
            (stats[:, cv2.CC_STAT_WIDTH] <= width * 0.60))
    centers = centers[good]
    if len(centers) < 2 or np.ptp(centers[:, 0]) < max(4, width * 0.01):
        return False
    # Allows multiple text lines without treating scattered bright photo details
    # as text merely because they occur on a dark background.
    ordered_y = np.sort(centers[:, 1])
    return bool(np.any(np.diff(ordered_y) <= max(2, height * 0.20)))

def _banner_depth(rgb, tolerance, uniformity):
    """Remove a bounded dark footer with text-like, bright foreground evidence."""
    height, width = rgb.shape[:2]
    limit = min(height - 1, int(height * 0.20))
    if limit < 6 or width < 8:
        return 0
    lines = rgb[::-1]
    base = np.median(lines[0], axis=0)
    if float(base @ np.array([0.2126, 0.7152, 0.0722])) > 0.20:
        return 0
    region = lines[:min(height, limit + 8)]
    distance = np.max(np.abs(region - base), axis=2)
    luma = region @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    background = distance <= tolerance + 1e-6
    # Lettering may occupy a large fraction of an individual row. Test the
    # remaining dark background's purity rather than stopping inside a glyph.
    base_luma = float(base @ np.array([0.2126, 0.7152, 0.0722]))
    dark = luma <= base_luma + tolerance * 0.75 + 1e-6
    dark_count = np.sum(dark[:limit], axis=1)
    purity = np.sum(background[:limit] & dark[:limit], axis=1) / np.maximum(1, dark_count)
    matches = ((np.mean(background[:limit], axis=1) >= max(0.50, uniformity - 0.30)) &
               (purity >= max(0.95, uniformity)))
    depth = _first_failure(matches)
    if depth < 6 or _boundary_fraction(region, depth, tolerance) < 0.25:
        return 0
    band = rgb[-depth:]
    luminance = band @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    bright = (luminance >= 0.65) & (np.ptp(band, axis=2) <= 0.20)
    # A cut through text is not a valid footer boundary. The whole footer must
    # be background-dominated even if its lettering is dense on a few rows.
    if (np.any(bright[0]) or float(np.mean(background[:depth])) < max(0.80, uniformity) or
            float(np.mean(luminance)) > 0.23):
        return 0
    bright_fraction = float(np.mean(bright))
    if not 0.002 <= bright_fraction <= 0.20 or not _text_like(bright):
        return 0
    return depth

def _supported_trims(work, trims, tolerance):
    """Corroborate a rectangular photo boundary within the proposed inner frame.

    Looking only at the full canvas would reject heavily letterboxed photos.
    Looking only for any subject-colored pixels would crop uniform studio
    backgrounds as if they were frames. Require broad seams in the candidate
    photo rectangle instead of tightening around an isolated object.
    """
    left, top, right, bottom = trims
    height, width = work.shape[:2]
    rows = work[:, left:width - right]
    columns = work[top:height - bottom].swapaxes(0, 1)
    lines = (columns, rows, columns[::-1], rows[::-1])
    return tuple(depth if depth and _boundary_fraction(edge, depth, tolerance) >= 0.65 else 0
                 for depth, edge in zip(trims, lines))

def _bounds(rgb, config, policy):
    height, width = rgb.shape[:2]
    left = top = right = bottom = 0
    conservative = policy == "Conservative"
    tolerance = float(config.autocrop_fuzz_tolerance)
    uniformity = float(config.autocrop_edge_uniformity)
    if conservative:
        tolerance = min(tolerance, 0.12)
    solid_uniformity = max(uniformity, 0.985) if conservative else uniformity
    banner_used = False
    # Revisit after stripping an outer frame: a footer may initially be hidden
    # behind a white bottom border. At most one footer is stripped per image.
    for _ in range(4):
        work = rgb[top:height - bottom, left:width - right]
        if min(work.shape[:2]) < 3:
            break
        changed = False
        if config.autocrop_strip_bottom_banner and not banner_used:
            depth = _banner_depth(work, tolerance, uniformity)
            if depth:
                bottom += depth
                banner_used = changed = True
                work = rgb[top:height - bottom, left:width - right]
        if config.autocrop_detect_borders:
            columns = work.swapaxes(0, 1)
            trims = (
                _border_depth(columns, tolerance, solid_uniformity, conservative),
                _border_depth(work, tolerance, solid_uniformity, conservative),
                _border_depth(columns[::-1], tolerance, solid_uniformity, conservative),
                _border_depth(work[::-1], tolerance, solid_uniformity, conservative),
            )
            if conservative and any(trims):
                trims = _supported_trims(work, trims, tolerance)
            if any(trims):
                dl, dt, dr, db = trims
                left, top, right, bottom = left + dl, top + dt, right + dr, bottom + db
                changed = True
        if not changed:
            break
    return left, top, right, bottom

def _validate_config(config):
    return validate_config(config, "AutoCrop preview.crop_config")


class BorderCropEngine:
    def run(self, image, crop_config, border_policy="Conservative"):
        config = _expect(crop_config, CropConfigV2, "crop_config")
        if border_policy not in POLICIES:
            raise ValueError("AutoCrop preview: border_policy must be Conservative or Configured tolerances.")
        image = image_tensor(image, "AutoCrop preview.image", allow_unbatched=False)
        if not config.autocrop_detect_borders and not config.autocrop_strip_bottom_banner:
            return image, 0, 0, 0, 0, False
        proposals = []
        # Analyze CPU float32 data once per image. For the usual CPU loader
        # output this shares memory; CUDA inputs require one bounded transfer,
        # not repeated per-row GPU synchronizations. Output stays on its device.
        for frame in image:
            analysis = frame.detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(analysis).all() or np.min(analysis) < 0 or np.max(analysis) > 1:
                raise ValueError("AutoCrop preview: image values must be finite and normalized to [0, 1]. "
                                 "Use a ComfyUI photo loader; do not feed raw 0–255 pixels or HDR values.")
            if analysis.shape[-1] == 4 and np.any(analysis[..., 3] != 1):
                # Transparent edges are not photographic borders. Composite
                # them explicitly in Photo Loader before requesting cropping.
                proposals.append((0, 0, 0, 0))
                continue
            rgb = np.repeat(analysis, 3, axis=-1) if analysis.shape[-1] == 1 else analysis[..., :3]
            proposals.append(_bounds(rgb, config, border_policy))
        # Scalar trim sockets must describe EVERY batch member. A shared safe
        # rectangle avoids the old per-image black padding and first-image-only
        # metadata (which could misalign masks or expose artificial borders).
        padding = config.autocrop_pad_px
        left, top, right, bottom = (max(0, min(values) - padding) for values in zip(*proposals))
        detected = bool(left or top or right or bottom)
        if not detected:
            return image, 0, 0, 0, 0, False
        result = image[:, top:image.shape[1] - bottom, left:image.shape[2] - right, :]
        _LOG.info("AutoCrop preview: %s policy; trim L/T/R/B=%s; %dx%d -> %dx%d; shared batch=%d",
                  border_policy, (left, top, right, bottom), image.shape[2], image.shape[1],
                  result.shape[2], result.shape[1], image.shape[0])
        return result, left, top, right, bottom, True
