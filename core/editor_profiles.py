"""Exact editor size profiles; no pixels, model loading, or node state."""
from __future__ import annotations
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

EDITOR_TARGET_TYPE = "PORTRAIT_EDITOR_TARGET"

TARGET_FIRERED = "FireRed Image Edit 1.1"

TARGET_KLEIN = "Flux 2 Klein Base"

TARGET_QWEN21 = "Qwen Image Edit 2.1"

TARGET_DIRECT = "Direct / No Editor"

EDITOR_TARGETS = (TARGET_QWEN21, TARGET_FIRERED, TARGET_KLEIN, TARGET_DIRECT)

TIER_1MP = "1 MP"

TIER_15MP = "1.5 MP"

TIER_2K = "2K (~4 MP)"

ASPECT_AUTO = "Auto"

FORCED_ASPECTS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
}

ASPECT_OPTIONS = (ASPECT_AUTO, *FORCED_ASPECTS.keys())

_LAZY_NOT_CONNECTED = object()

@dataclass(frozen=True)
class _Resolution:
    width: int
    height: int
    tier: str
    family: str

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def ratio(self) -> float:
        return self.width / self.height

# The authors' FireRed 1.1 ComfyUI example uses FluxKontextImageScale.
# Keep its actual ~1 MP sizes, without importing a ComfyUI/model dependency.
# https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/blob/main/firered-image-edit-1.1.json
_FIRERED_1MP = (
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024),
    (1104, 944), (1184, 880), (1248, 832), (1328, 800),
    (1392, 752), (1456, 720), (1504, 688), (1568, 672),
)

def _unwrap_target(value) -> Optional[str]:
    """Return a normalized editor target from ComfyUI's value wrappers."""
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if value is None:
        return None
    value = str(value)
    if value == "Qwen Image Edit 2511":
        raise ValueError("Qwen Image Edit 2511 has been retired. Set the Editor Target Beacon "
                         "to FireRed Image Edit 1.1 or Qwen Image Edit 2.1 and connect "
                         "that editor to the matching Editor Result Gate input.")
    if value not in EDITOR_TARGETS:
        raise ValueError(f"Unknown editor target: {value!r}")
    return value

def _aligned_resolution(ratio: float, pixel_budget: int, alignment: int = 16) -> Tuple[int, int]:
    """Find an aligned size close to both an aspect ratio and pixel budget."""
    ratio = max(1e-6, float(ratio))
    ideal_w_units = math.sqrt(pixel_budget * ratio) / alignment
    ideal_h_units = math.sqrt(pixel_budget / ratio) / alignment
    best = None
    for wu in range(max(4, round(ideal_w_units) - 10), round(ideal_w_units) + 11):
        for hu in range(max(4, round(ideal_h_units) - 10), round(ideal_h_units) + 11):
            width = wu * alignment
            height = hu * alignment
            ratio_error = abs(math.log((width / height) / ratio))
            area_error = abs(math.log((width * height) / pixel_budget))
            key = (ratio_error * 8.0 + area_error, ratio_error, area_error, width * height)
            if best is None or key < best[0]:
                best = (key, width, height)
    return int(best[1]), int(best[2])

def _exact_aspect_resolution(
    numerator: int, denominator: int, pixel_budget: int, alignment: int = 16
) -> Tuple[int, int]:
    """Return an aligned resolution with a mathematically exact aspect ratio."""
    step = math.lcm(
        alignment // math.gcd(alignment, numerator),
        alignment // math.gcd(alignment, denominator),
    )
    ideal_scale = math.sqrt(pixel_budget / max(1, numerator * denominator))
    center = max(step, round(ideal_scale / step) * step)
    scales = {
        max(step, center + offset * step)
        for offset in range(-3, 4)
    }
    scale = min(
        scales,
        key=lambda value: (
            abs(math.log((numerator * denominator * value * value) / pixel_budget)),
            numerator * denominator * value * value,
        ),
    )
    return numerator * scale, denominator * scale

@lru_cache(maxsize=128)
def _build_resolutions(
    target: str,
    source_ratio: float,
    smart_ratio: Optional[float],
    forced_aspect: str = ASPECT_AUTO,
    resolution_policy: Optional[str] = None,
) -> Tuple[_Resolution, ...]:
    forced_parts = FORCED_ASPECTS.get(forced_aspect)

    if target == TARGET_FIRERED:
        # Only ~1 MP is established by the authors' v1.1 workflow. Larger
        # aligned output is explicit opt-in, never silently selected by Auto.
        tiers = ((TIER_1MP, 1024 * 1024),)
        if resolution_policy in (None, "Force 1.5 MP"):
            tiers += ((TIER_15MP, round(1.5 * 1024 * 1024)),)
        if forced_parts is not None:
            return tuple(
                _Resolution(*_exact_aspect_resolution(*forced_parts, budget),
                            tier, f"{forced_aspect} forced, 16-aligned"
                            + (" (experimental)" if tier == TIER_15MP else ""))
                for tier, budget in tiers
            )
        resolutions = [
            _Resolution(w, h, TIER_1MP, "FireRed workflow preset")
            for w, h in _FIRERED_1MP
        ]
        if len(tiers) > 1:
            resolutions.extend(
                _Resolution(*_aligned_resolution(w / h, tiers[1][1]),
                            TIER_15MP, "FireRed larger output (experimental)")
                for w, h in _FIRERED_1MP
            )
        return tuple(resolutions)

    if target == TARGET_QWEN21:
        # The ComfyUI edit canvas follows image_1 when its resolution control is 0.
        # Qwen 2.1 follows the input aspect: choose exact
        # 32-pixel-aligned output dimensions at each supported pixel budget.
        if forced_parts is not None:
            numerator, denominator = forced_parts
            resolutions = []
            for tier, budget in (
                (TIER_1MP, 1024 * 1024),
                (TIER_15MP, round(1.5 * 1024 * 1024)),
                (TIER_2K, 2048 * 2048),
            ):
                width, height = _exact_aspect_resolution(
                    numerator, denominator, budget, alignment=32
                )
                resolutions.append(
                    _Resolution(width, height, tier, f"{forced_aspect} forced")
                )
            return tuple(resolutions)
        ratio = smart_ratio if smart_ratio is not None else source_ratio
        resolutions = []
        for tier, budget in (
            (TIER_1MP, 1024 * 1024),
            (TIER_15MP, round(1.5 * 1024 * 1024)),
            (TIER_2K, 2048 * 2048),
        ):
            width, height = _aligned_resolution(ratio, budget, alignment=32)
            resolutions.append(_Resolution(width, height, tier, "dynamic 32-aligned"))
        return tuple(resolutions)

    if target == TARGET_KLEIN:
        if forced_parts is not None:
            numerator, denominator = forced_parts
            resolutions = []
            for tier, budget in (
                (TIER_1MP, 1024 * 1024),
                (TIER_15MP, round(1.5 * 1024 * 1024)),
            ):
                width, height = _exact_aspect_resolution(
                    numerator, denominator, budget
                )
                resolutions.append(
                    _Resolution(width, height, tier, f"{forced_aspect} forced")
                )
            return tuple(resolutions)
        ratio = smart_ratio if smart_ratio is not None else source_ratio
        resolutions = []
        for tier, budget in ((TIER_1MP, 1024 * 1024), (TIER_15MP, round(1.5 * 1024 * 1024))):
            width, height = _aligned_resolution(ratio, budget)
            resolutions.append(_Resolution(width, height, tier, "dynamic aligned"))
        return tuple(resolutions)

    return ()
