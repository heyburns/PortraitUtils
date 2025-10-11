import math
from typing import Tuple

import numpy as np
import torch
from PIL import Image


class ImageMegapixelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "divisible_by": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Force output width/height to be divisible by this value.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "target_megapixels")
    FUNCTION = "select"
    CATEGORY = "image/analysis"

    @staticmethod
    def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")
        if image.shape[-1] not in (3, 4):
            raise ValueError("ImageMegapixelSelector expects RGB(A) images.")
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image.to(torch.float32).clamp(0.0, 1.0)

    @staticmethod
    def _choose_megapixels(total_pixels: float) -> float:
        megapixels = total_pixels / 1_000_000.0
        if megapixels <= 1.0:
            return 1.0
        if megapixels >= 3.0:
            return 3.0
        # Between 1 and 3: round to nearest legal breakpoint {1, 2, 3} with ties rounding up.
        candidates = [1.0, 2.0, 3.0]
        distances = [abs(megapixels - c) for c in candidates]
        min_distance = min(distances)
        # Pick the smallest candidate achieving the min distance, but prefer larger on ties.
        best = max(c for c, d in zip(candidates, distances) if abs(d - min_distance) < 1e-9)
        return best

    @staticmethod
    def _round_to_multiple(value: int, modulus: int) -> int:
        if modulus <= 1:
            return max(1, value)
        return max(modulus, int(round(value / modulus)) * modulus)

    @staticmethod
    def _enforce_mod_dims(
        raw_width: int,
        raw_height: int,
        aspect: float,
        modulus: int,
    ) -> Tuple[int, int]:
        if modulus <= 1:
            return max(1, raw_width), max(1, raw_height)

        raw_pixels = max(1, raw_width * raw_height)
        raw_ratio = aspect
        best = None

        width_candidates = set()
        rounded_base = ImageMegapixelSelector._round_to_multiple(raw_width, modulus)
        width_candidates.add(rounded_base)
        for offset in range(-4, 5):
            candidate = raw_width + offset * modulus
            if candidate < modulus:
                continue
            width_candidates.add(ImageMegapixelSelector._round_to_multiple(candidate, modulus))

        for w in sorted(width_candidates):
            if w < modulus:
                continue
            ratio_based_height = w / raw_ratio
            height_values = set()
            for base in (
                ratio_based_height,
                math.floor(ratio_based_height),
                math.ceil(ratio_based_height),
                raw_height,
            ):
                if base <= 0:
                    continue
                height_values.add(ImageMegapixelSelector._round_to_multiple(int(round(base)), modulus))

            for h in height_values:
                if h < modulus:
                    continue
                ratio = w / h
                ratio_diff = abs(ratio - raw_ratio)
                pixel_diff = abs((w * h) - raw_pixels)
                dim_diff = abs(w - raw_width) + abs(h - raw_height)
                key = (ratio_diff, pixel_diff, dim_diff)
                if best is None or key < best[0]:
                    best = (key, w, h)

        if best is not None:
            _, w_opt, h_opt = best
            return max(1, w_opt), max(1, h_opt)

        # Fallback: simple rounding
        return (
            ImageMegapixelSelector._round_to_multiple(raw_width, modulus),
            ImageMegapixelSelector._round_to_multiple(raw_height, modulus),
        )

    @staticmethod
    def _resize_lanczos(sample: torch.Tensor, width: int, height: int) -> torch.Tensor:
        device = sample.device
        dtype = sample.dtype
        mode = "RGB"

        np_img = sample.detach().cpu().numpy()
        np_img = np.clip(np_img, 0.0, 1.0)
        np_img = (np_img * 255.0).round().astype(np.uint8)

        pil_image = Image.fromarray(np_img, mode=mode)
        resized = pil_image.resize((int(width), int(height)), Image.LANCZOS)
        arr = np.asarray(resized).astype(np.float32) / 255.0

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]

        tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
        return tensor.clamp(0.0, 1.0)

    def select(self, image: torch.Tensor, divisible_by: int = 8):
        tensor = self._to_bhwc(image)
        if tensor.shape[0] != 1:
            raise ValueError("ImageMegapixelSelector currently supports unbatched images (B=1).")

        sample = tensor[0]
        height, width = sample.shape[:2]
        total_pixels = float(height * width)
        target_mp = self._choose_megapixels(total_pixels)
        target_pixels = max(1.0, target_mp * 1_000_000.0)

        scale = math.sqrt(target_pixels / max(total_pixels, 1.0))
        raw_width = max(1, int(round(width * scale)))
        raw_height = max(1, int(round(height * scale)))
        modulus = max(1, int(divisible_by))

        aspect = width / max(1.0, float(height))
        target_width, target_height = self._enforce_mod_dims(raw_width, raw_height, aspect, modulus)

        if target_width == width and target_height == height:
            resized = sample
        else:
            resized = self._resize_lanczos(sample, target_width, target_height)

        actual_mp = float(target_width * target_height) / 1_000_000.0
        output = resized.unsqueeze(0).to(image.dtype if isinstance(image, torch.Tensor) else torch.float32)

        return (output, actual_mp)


NODE_CLASS_MAPPINGS = {
    "ImageMegapixelSelector": ImageMegapixelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMegapixelSelector": "Image Megapixel Selector",
}
