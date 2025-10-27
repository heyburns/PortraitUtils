import torch


def _ensure_bhwc_rgb(image: torch.Tensor) -> torch.Tensor:
    """Normalize ComfyUI IMAGE tensors to float32 [B,H,W,3]."""
    t = image
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    t = t.to(torch.float32)

    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        raise ValueError(f"Expected tensor with 4 dims (B,H,W,C), got {tuple(t.shape)}")

    ch = t.shape[-1]
    if ch == 3:
        return t.clamp(0.0, 1.0)
    if ch == 1:
        return t.repeat(1, 1, 1, 3).clamp(0.0, 1.0)
    return t[..., :3].clamp(0.0, 1.0)


def _rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype)
    return torch.tensordot(rgb, weights, dims=([-1], [0]))


class StripBottomBanner:
    """Crop a dark metadata banner anchored to the bottom edge of an image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_dark_threshold": (
                    "FLOAT",
                    {"default": 0.22, "min": 0.01, "max": 0.6, "step": 0.01},
                ),
                "dark_fraction_threshold": (
                    "FLOAT",
                    {"default": 0.78, "min": 0.4, "max": 0.98, "step": 0.01},
                ),
                "bright_pixel_threshold": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.4, "max": 1.0, "step": 0.01},
                ),
                "bright_fraction_threshold": (
                    "FLOAT",
                    {"default": 0.015, "min": 0.0, "max": 0.5, "step": 0.001},
                ),
                "min_band_percent": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.1},
                ),
                "max_scan_percent": (
                    "FLOAT",
                    {"default": 35.0, "min": 5.0, "max": 80.0, "step": 1.0},
                ),
                "extra_trim_px": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "require_bright_rows": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN")
    RETURN_NAMES = ("image", "trimmed_rows", "detected")
    FUNCTION = "strip"
    CATEGORY = "Image/Transform"

    def strip(
        self,
        image,
        pixel_dark_threshold=0.22,
        dark_fraction_threshold=0.78,
        bright_pixel_threshold=0.7,
        bright_fraction_threshold=0.015,
        min_band_percent=3.0,
        max_scan_percent=35.0,
        extra_trim_px=2,
        require_bright_rows=True,
    ):
        img = _ensure_bhwc_rgb(image)
        B, H, W, _ = img.shape
        if H <= 4:
            return (img, 0, False)

        gray = _rgb_to_luma(img)

        scan_rows = max(1, int(H * (max_scan_percent / 100.0)))
        scan_rows = min(scan_rows, H)
        start_row = H - scan_rows
        region = gray[:, start_row:, :]

        row_mean = region.mean(dim=-1)
        dark_fraction = (region <= pixel_dark_threshold).float().mean(dim=-1)
        bright_fraction = (region >= bright_pixel_threshold).float().mean(dim=-1)

        min_band_px = max(1, int(H * (min_band_percent / 100.0)))
        trimmed = torch.zeros(B, dtype=torch.int64, device=img.device)
        detected = torch.zeros(B, dtype=torch.bool, device=img.device)

        row_mask = (row_mean <= pixel_dark_threshold) & (
            dark_fraction >= dark_fraction_threshold
        )

        row_mask_cpu = row_mask.cpu()
        bright_fraction_cpu = bright_fraction.cpu()

        for b in range(B):
            rows = row_mask_cpu[b]
            brights = bright_fraction_cpu[b]
            run = 0
            seen_bright = False
            for idx in range(rows.shape[0] - 1, -1, -1):
                if rows[idx]:
                    run += 1
                    if brights[idx] >= bright_fraction_threshold:
                        seen_bright = True
                else:
                    break

            if run == 0:
                continue

            if run < min_band_px:
                continue

            if require_bright_rows and not seen_bright:
                continue

            total_trim = min(H, run + extra_trim_px)
            trimmed[b] = total_trim
            detected[b] = True

        if not detected.any():
            return (img, 0, False)

        if not detected.all():
            return (img, 0, False)

        trim_rows = int(trimmed.min().item())
        if trim_rows <= 0:
            return (img, 0, False)

        new_height = max(1, H - trim_rows)
        cropped = img[:, :new_height, :, :].contiguous()
        return (cropped, trim_rows, True)


NODE_CLASS_MAPPINGS = {
    "StripBottomBanner": StripBottomBanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StripBottomBanner": "Strip Bottom Banner",
}
