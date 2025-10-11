"""
Flux-friendly aspect cropper and resolution normalizer.

This node crops an image to the closest allowed aspect ratio while discarding
as little content as possible, optionally pre-upscaling low resolution inputs,
then resizes the crop to the nearest Flux-compatible resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class _TargetResolution:
    width: int
    height: int
    label: str
    ratio_label_override: Optional[str] = None

    @property
    def ratio(self) -> float:
        return self.width / self.height

    @property
    def ratio_label(self) -> str:
        if self.ratio_label_override:
            return self.ratio_label_override
        num, den = _normalize_ratio(self.width, self.height)
        return f"{num}:{den}"

    @property
    def area(self) -> int:
        return self.width * self.height


def _normalize_ratio(num: int, den: int) -> Tuple[int, int]:
    g = math.gcd(num, den)
    return num // g, den // g


def _swap_ratio_label(label: Optional[str]) -> Optional[str]:
    if not label or ":" not in label:
        return label
    left, right = label.split(":", 1)
    return f"{right}:{left}"


def _build_target_resolutions() -> Tuple[_TargetResolution, ...]:
    """Flux legal resolution list including landscape inverses."""
    base_specs = [
        (896, 1152, None),
        (832, 1216, None),
        (768, 1344, None),
        (640, 1536, "9:21"),
        (1024, 1024, None),
        (1152, 1728, None),
        (1216, 1664, None),
        (1088, 1920, None),
        (960, 2176, "9:21"),
        (1408, 1408, None),
    ]

    targets: List[_TargetResolution] = []
    seen = set()
    for width, height, ratio_label in base_specs:
        for w, h, lbl in (
            (width, height, ratio_label),
            (height, width, _swap_ratio_label(ratio_label)),
        ):
            key = (w, h)
            if key in seen:
                continue
            seen.add(key)
            targets.append(_TargetResolution(w, h, f"{w}x{h}", lbl))
    return tuple(targets)


_TARGET_RESOLUTIONS: Tuple[_TargetResolution, ...] = _build_target_resolutions()


def _ensure_bhwc(image) -> torch.Tensor:
    tensor = image
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(tensor.shape)}")
    if tensor.shape[-1] not in (3, 4):
        raise ValueError("IMAGE tensor must have 3 (RGB) or 4 (RGBA) channels")
    return tensor.to(torch.float32).clamp(0.0, 1.0)


def _resize_sample(sample: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if sample.shape[0] == height and sample.shape[1] == width:
        return sample
    # sample: [H, W, C]
    tensor = sample.permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    return tensor.squeeze(0).permute(1, 2, 0)


def _compute_crop_dims(width: int, height: int, ratio: float) -> Tuple[int, int]:
    if ratio <= 0.0:
        return width, height

    current_ratio = width / height
    if current_ratio > ratio:
        target_height = height
        target_width = max(1, min(width, int(round(target_height * ratio))))
    else:
        target_width = width
        target_height = max(1, min(height, int(round(target_width / ratio))))

    target_width = min(target_width, width)
    target_height = min(target_height, height)
    return target_width, target_height


def _center_crop(sample: torch.Tensor, width: int, height: int) -> Tuple[torch.Tensor, int, int]:
    src_h, src_w = sample.shape[:2]
    left = max(0, (src_w - width) // 2)
    top = max(0, (src_h - height) // 2)
    right = min(src_w, left + width)
    bottom = min(src_h, top + height)
    # Re-align if rounding caused overflow.
    left = max(0, right - width)
    top = max(0, bottom - height)
    cropped = sample[top:bottom, left:right, :]
    return cropped, left, top


def _select_target_combo(width: int, height: int):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid crop dimensions encountered")

    original_area = width * height
    original_ratio = width / height
    best = None
    best_key = None

    for target in _TARGET_RESOLUTIONS:
        ratio = target.ratio
        crop_w, crop_h = _compute_crop_dims(width, height, ratio)
        if crop_w <= 0 or crop_h <= 0:
            continue

        crop_area = crop_w * crop_h
        area_loss = 1.0 - (crop_area / max(original_area, 1))
        area_loss = max(0.0, min(1.0, area_loss))

        cropped_ratio = crop_w / crop_h
        ratio_delta = abs(cropped_ratio - ratio)
        origin_delta = abs(cropped_ratio - original_ratio)

        scale_w = target.width / max(crop_w, 1)
        scale_h = target.height / max(crop_h, 1)
        scale_dev = max(abs(scale_w - 1.0), abs(scale_h - 1.0))
        scale_mean = (abs(scale_w - 1.0) + abs(scale_h - 1.0)) * 0.5

        key = (scale_dev, area_loss, scale_mean, ratio_delta, origin_delta, target.area)
        if best_key is None or key < best_key:
            best = (target, crop_w, crop_h, area_loss)
            best_key = key

    if best is None:
        raise RuntimeError("Failed to choose Flux target resolution")
    return best


def _maybe_pre_upscale(sample: torch.Tensor, min_megapixels: float) -> Tuple[torch.Tensor, float]:
    height, width = sample.shape[:2]
    area = height * width
    min_pixels = max(0.0, min_megapixels) * 1_000_000.0
    if min_pixels <= 0.0 or area >= min_pixels:
        return sample, 1.0

    scale = math.sqrt(min_pixels / max(area, 1.0))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    if new_height == height and new_width == width:
        return sample, 1.0
    upscaled = _resize_sample(sample, new_height, new_width)
    return upscaled, new_width / max(width, 1.0)


class FluxResolutionPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_megapixels": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 64.0, "step": 0.05}),
                "enable_pre_upscale": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "crop_width": ("INT", {"default": -1}),
                "crop_height": ("INT", {"default": -1}),
                "crop_x": ("INT", {"default": 0}),
                "crop_y": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "ratio", "target_width", "target_height", "area_loss_percent", "pre_scale_factor")
    FUNCTION = "apply"
    CATEGORY = "image/transform"

    def apply(
        self,
        image,
        min_megapixels=0.95,
        enable_pre_upscale=True,
        crop_width=-1,
        crop_height=-1,
        crop_x=0,
        crop_y=0,
    ):
        tensor = _ensure_bhwc(image)
        if tensor.shape[0] != 1:
            raise ValueError("FluxResolutionPrepare expects an unbatched IMAGE tensor (B=1)")

        sample = tensor[0]
        orig_height, orig_width = sample.shape[:2]
        pre_scale = 1.0
        if enable_pre_upscale:
            sample, pre_scale = _maybe_pre_upscale(sample, min_megapixels)

        sample_height, sample_width = sample.shape[:2]
        scale_x = sample_width / max(1, orig_width)
        scale_y = sample_height / max(1, orig_height)

        orig_area = float(sample_height * sample_width)

        work = sample
        if isinstance(crop_width, (int, float)) and isinstance(crop_height, (int, float)):
            if crop_width > 0 and crop_height > 0:
                sx = int(round(max(0.0, crop_x) * scale_x))
                sy = int(round(max(0.0, crop_y) * scale_y))
                sw = int(round(crop_width * scale_x))
                sh = int(round(crop_height * scale_y))
                if sw > 0 and sh > 0:
                    sx = max(0, min(sx, sample_width - 1))
                    sy = max(0, min(sy, sample_height - 1))
                    sw = max(1, min(sw, sample_width - sx))
                    sh = max(1, min(sh, sample_height - sy))
                    work = sample[sy : sy + sh, sx : sx + sw, :]

        work_height, work_width = work.shape[:2]
        target, crop_w, crop_h, _ = _select_target_combo(work_width, work_height)
        cropped, _, _ = _center_crop(work, crop_w, crop_h)

        resized = _resize_sample(cropped, target.height, target.width).clamp(0.0, 1.0)
        output = resized.unsqueeze(0)

        final_area = float(crop_w * crop_h)
        total_area_loss = max(0.0, 1.0 - (final_area / max(orig_area, 1.0)))
        area_loss_percent = float(total_area_loss * 100.0)
        return (
            output,
            target.ratio_label,
            int(target.width),
            int(target.height),
            area_loss_percent,
            float(pre_scale),
        )


NODE_CLASS_MAPPINGS = {
    "FluxResolutionPrepare": FluxResolutionPrepare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxResolutionPrepare": "Flux Resolution Prepare",
}
