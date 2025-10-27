import torch


def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
    """Normalize ComfyUI IMAGE tensors to float32 [B,H,W,3]."""
    t = image
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    t = t.to(torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        raise ValueError(f"Expected tensor with 4 dims (B,H,W,C), got {tuple(t.shape)}")
    if t.shape[-1] == 3:
        return t.clamp(0.0, 1.0)
    if t.shape[-1] == 1:
        return t.repeat(1, 1, 1, 3).clamp(0.0, 1.0)
    return t[..., :3].clamp(0.0, 1.0)


def _line_stats(line: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return mean colour vector and scalar variance for a row/column."""
    mean = line.mean(dim=0)
    variance = ((line - mean) ** 2).mean().item()
    return mean, variance


def _scan_uniform_band(
    img: torch.Tensor,
    axis: int,
    reverse: bool,
    max_scan: int,
    variance_threshold: float,
    colour_delta_threshold: float,
    min_band_px: int,
    mean_follow: float,
    min_uniform_fraction: float,
    dominant_bins: int,
    retry_allowance: int,
) -> int:
    """Return how many rows/columns from the edge look like a flat colour band."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (rows) or 1 (cols)")

    H, W, _ = img.shape
    length = H if axis == 0 else W
    limit = min(max_scan, length)
    if limit <= 0:
        return 0

    trimmed = 0
    base_mean = None
    remaining_retries = max(0, int(retry_allowance))
    for step in range(limit):
        idx = length - 1 - step if reverse else step
        if axis == 0:
            line = img[idx, :, :]
        else:
            line = img[:, idx, :]

        mean, var = _line_stats(line)
        if base_mean is None:
            base_mean = mean
            trimmed += 1
            continue

        diff_mean = torch.abs(mean - base_mean).max().item()
        diff_pixels = torch.abs(line - base_mean.unsqueeze(0))
        within_fraction = (
            (diff_pixels <= colour_delta_threshold)
            .all(dim=-1)
            .float()
            .mean()
            .item()
        )

        dominant_share = within_fraction
        if dominant_share < min_uniform_fraction and dominant_bins > 1:
            quant = torch.clamp(
                (line * (dominant_bins - 1)).round().to(torch.int64),
                min=0,
                max=dominant_bins - 1,
            )
            scale_g = dominant_bins * dominant_bins
            scale_b = dominant_bins
            hashes = quant[:, 0] * scale_g + quant[:, 1] * scale_b + quant[:, 2]
            uniq, counts = torch.unique(hashes, return_counts=True)
            dominant_share = counts.max().item() / max(1, hashes.shape[0])

        should_stop_on_diff = (
            within_fraction < min_uniform_fraction
            and dominant_share < min_uniform_fraction
            and diff_mean > colour_delta_threshold
        )
        should_stop_on_var = (
            var > variance_threshold
            and within_fraction < min_uniform_fraction
            and dominant_share < min_uniform_fraction
        )

        if should_stop_on_diff or should_stop_on_var:
            if remaining_retries > 0:
                remaining_retries -= 1
            else:
                break

        base_mean = base_mean * (1.0 - mean_follow) + mean * mean_follow

        trimmed += 1

    return trimmed if trimmed >= min_band_px else 0


def _strip_single(
    img: torch.Tensor,
    max_border_percent: float,
    variance_threshold: float,
    colour_delta_threshold: float,
    min_band_px: int,
    passes: int,
    mean_follow: float,
    min_uniform_fraction: float,
    dominant_bins: int,
    retry_allowance: int,
) -> tuple[torch.Tensor, dict[str, int], bool]:
    """Strip flat colour borders from a single image."""
    h_total = 0
    b_total = 0
    l_total = 0
    r_total = 0
    trimmed_any = False

    dom_bins = max(1, int(dominant_bins))
    retry_limit = max(0, int(retry_allowance))

    for _ in range(max(1, passes)):
        H, W, _ = img.shape
        max_rows = max(1, int(H * (max_border_percent / 100.0)))
        max_cols = max(1, int(W * (max_border_percent / 100.0)))

        top = _scan_uniform_band(
            img,
            axis=0,
            reverse=False,
            max_scan=max_rows,
            variance_threshold=variance_threshold,
            colour_delta_threshold=colour_delta_threshold,
            min_band_px=min_band_px,
            mean_follow=mean_follow,
            min_uniform_fraction=min_uniform_fraction,
            dominant_bins=dom_bins,
            retry_allowance=retry_limit,
        )
        if top > 0:
            img = img[top:, :, :]
            h_total += top
            trimmed_any = True

        bottom = _scan_uniform_band(
            img,
            axis=0,
            reverse=True,
            max_scan=max_rows,
            variance_threshold=variance_threshold,
            colour_delta_threshold=colour_delta_threshold,
            min_band_px=min_band_px,
            mean_follow=mean_follow,
            min_uniform_fraction=min_uniform_fraction,
            dominant_bins=dom_bins,
            retry_allowance=retry_limit,
        )
        if bottom > 0:
            img = img[:-bottom, :, :]
            b_total += bottom
            trimmed_any = True

        left = _scan_uniform_band(
            img,
            axis=1,
            reverse=False,
            max_scan=max_cols,
            variance_threshold=variance_threshold,
            colour_delta_threshold=colour_delta_threshold,
            min_band_px=min_band_px,
            mean_follow=mean_follow,
            min_uniform_fraction=min_uniform_fraction,
            dominant_bins=dom_bins,
            retry_allowance=retry_limit,
        )
        if left > 0:
            img = img[:, left:, :]
            l_total += left
            trimmed_any = True

        right = _scan_uniform_band(
            img,
            axis=1,
            reverse=True,
            max_scan=max_cols,
            variance_threshold=variance_threshold,
            colour_delta_threshold=colour_delta_threshold,
            min_band_px=min_band_px,
            mean_follow=mean_follow,
            min_uniform_fraction=min_uniform_fraction,
            dominant_bins=dom_bins,
            retry_allowance=retry_limit,
        )
        if right > 0:
            img = img[:, :-right, :]
            r_total += right
            trimmed_any = True

    totals = {
        "top": h_total,
        "bottom": b_total,
        "left": l_total,
        "right": r_total,
    }
    return img, totals, trimmed_any


class StripSolidBorders:
    """Iteratively trim large flat-colour borders around archive images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_border_percent": (
                    "FLOAT",
                    {"default": 65.0, "min": 5.0, "max": 95.0, "step": 1.0},
                ),
                "variance_threshold": (
                    "FLOAT",
                    {"default": 0.0025, "min": 1e-6, "max": 0.1, "step": 1e-4},
                ),
                "colour_delta_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.001, "max": 0.5, "step": 0.001},
                ),
                "min_band_px": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "passes": ("INT", {"default": 2, "min": 1, "max": 6, "step": 1}),
                "mean_follow": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "min_uniform_fraction": (
                    "FLOAT",
                    {"default": 0.68, "min": 0.4, "max": 0.99, "step": 0.01},
                ),
                "dominant_bins": (
                    "INT",
                    {"default": 8, "min": 2, "max": 64, "step": 1},
                ),
                "edge_retry_allowance": (
                    "INT",
                    {"default": 1, "min": 0, "max": 4, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "image",
        "trim_left",
        "trim_top",
        "trim_right",
        "trim_bottom",
        "detected",
    )
    FUNCTION = "strip"
    CATEGORY = "Image/Transform"

    def strip(
        self,
        image,
        max_border_percent=65.0,
        variance_threshold=0.0025,
        colour_delta_threshold=0.08,
        min_band_px=2,
        passes=2,
        mean_follow=0.2,
        min_uniform_fraction=0.68,
        dominant_bins=8,
        edge_retry_allowance=1,
    ):
        img = _ensure_bhwc(image)

        if img.shape[0] != 1:
            raise ValueError("StripSolidBorders currently expects a single image batch.")

        work = img[0]
        cropped, totals, trimmed_any = _strip_single(
            work,
            max_border_percent=max_border_percent,
            variance_threshold=variance_threshold,
            colour_delta_threshold=colour_delta_threshold,
            min_band_px=min_band_px,
            passes=passes,
            mean_follow=mean_follow,
            min_uniform_fraction=min_uniform_fraction,
            dominant_bins=dominant_bins,
            retry_allowance=edge_retry_allowance,
        )

        cropped = cropped.unsqueeze(0).clamp(0.0, 1.0)

        return (
            cropped,
            int(totals["left"]),
            int(totals["top"]),
            int(totals["right"]),
            int(totals["bottom"]),
            bool(trimmed_any),
        )


NODE_CLASS_MAPPINGS = {
    "StripSolidBorders": StripSolidBorders,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StripSolidBorders": "Strip Solid Borders",
}
