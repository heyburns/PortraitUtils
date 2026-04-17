# auto_adjust.py (extracted from ImageUtils.py)
# Minimal standalone module containing AutoAdjustNode and AutoColorConfigNode
import torch


def _auto_color(rgb, use_color, snap_midtones):
    if not use_color:
        return rgb

    ycbcr = _rgb_to_ycbcr(rgb)
    Y, Cb, Cr = ycbcr[..., 0:1], ycbcr[..., 1:2], ycbcr[..., 2:3]

    if snap_midtones:
        mask = (Y > 0.2) & (Y < 0.8)
        cb_mid = _median_masked(Cb, mask)
        cr_mid = _median_masked(Cr, mask)
        Cb = Cb - cb_mid
        Cr = Cr - cr_mid
    else:
        cb_mean = torch.mean(Cb, dim=(1, 2), keepdim=True)
        cr_mean = torch.mean(Cr, dim=(1, 2), keepdim=True)
        Cb = Cb - cb_mean
        Cr = Cr - cr_mean

    ycbcr = torch.cat([Y, Cb, Cr], dim=-1)
    out = _ycbcr_to_rgb(ycbcr)
    return _clamp01(out)


def _median_masked(ch, msk):
    """Per-image median of ch where msk is True; falls back to all pixels if mask is empty.

    ch, msk: [B, H, W, 1] — result: [B, 1, 1, 1].
    Replaces the original one-liner which was broken for B > 1 because
    boolean indexing collapses the batch dimension into a flat 1-D tensor.
    """
    B = ch.shape[0]
    result = torch.zeros(B, 1, 1, 1, device=ch.device, dtype=ch.dtype)
    for b in range(B):
        ch_b = ch[b]      # [H, W, 1]
        msk_b = msk[b]    # [H, W, 1]
        selected = ch_b[msk_b] if msk_b.any() else ch_b.reshape(-1)
        result[b] = torch.quantile(selected.float(), 0.5)
    return result


def _auto_levels(rgb, use_levels, shadow_pct, highlight_pct, gamma_norm, precision_mode):
    if not use_levels:
        return rgb

    Y = _luma(rgb)
    if shadow_pct == 0.0 and highlight_pct == 0.0:
        low, high = torch.min(Y, dim=(1, 2), keepdim=True).values, torch.max(Y, dim=(1, 2), keepdim=True).values
    else:
        if shadow_pct + highlight_pct > 0:
            if shadow_pct > 0:
                low = (_percentiles_exact(Y, shadow_pct / 100.0) if precision_mode == "Exact" else _percentiles_hist(Y, shadow_pct / 100.0))
            else:
                low = torch.min(Y, dim=(1, 2), keepdim=True).values
            if highlight_pct > 0:
                high = (_percentiles_exact(Y, 1.0 - (highlight_pct / 100.0)) if precision_mode == "Exact" else _percentiles_hist(Y, 1.0 - (highlight_pct / 100.0)))
            else:
                high = torch.max(Y, dim=(1, 2), keepdim=True).values
        else:
            low = _percentiles_exact(Y, shadow_pct / 100.0)
            high = _percentiles_exact(Y, 1.0 - (highlight_pct / 100.0))

    stretched = _linear_stretch_scalar(rgb, low, high)

    if gamma_norm:
        # Apply gamma per image so each image gets its own correction factor,
        # rather than computing one aggregate median across the whole batch.
        for b in range(stretched.shape[0]):
            Ys_b = _luma(stretched[b:b + 1])
            mid_mask = (Ys_b > 0.25) & (Ys_b < 0.75)
            if mid_mask.any():
                flat = Ys_b[mid_mask]
                median = torch.quantile(flat.float(), 0.5).item()
                if 0.35 < median < 0.65:
                    gamma = 0.0
                    if median != 0.0 and median != 1.0:
                        gamma = torch.log(torch.tensor(0.5)) / torch.log(torch.tensor(median))
                        gamma = float(torch.clamp(gamma, 0.85, 1.15))
                    if gamma != 0.0:
                        stretched[b:b + 1] = torch.clamp(stretched[b:b + 1], 1e-6, 1.0) ** gamma

    return _clamp01(stretched)


def _auto_tone(rgb, use_tone, mode, shadow_pct, highlight_pct, precision_mode):
    if not use_tone:
        return rgb

    if mode == "Per-channel":
        if shadow_pct == 0.0 and highlight_pct == 0.0:
            low = torch.min(rgb, dim=(1, 2), keepdim=True).values
            high = torch.max(rgb, dim=(1, 2), keepdim=True).values
        else:
            def pct(ch, p):
                if p == 0.0:
                    return torch.min(ch, dim=(1, 2), keepdim=True).values
                return (_percentiles_exact(ch, p / 100.0) if precision_mode == "Exact" else _percentiles_hist(ch, p / 100.0))
            low = torch.cat([pct(rgb[..., i:i+1], shadow_pct) for i in range(3)], dim=-1)
            def pct_hi(ch, p):
                if p == 0.0:
                    return torch.max(ch, dim=(1, 2), keepdim=True).values
                return (_percentiles_exact(ch, 1.0 - (p / 100.0)) if precision_mode == "Exact" else _percentiles_hist(ch, 1.0 - (p / 100.0)))
            high = torch.cat([pct_hi(rgb[..., i:i+1], highlight_pct) for i in range(3)], dim=-1)
        stretched = _linear_stretch(rgb, low, high)
        return _clamp01(stretched)

    # Monochromatic: stretch only luma
    ycbcr = _rgb_to_ycbcr(rgb)
    Y = ycbcr[..., 0:1]
    if shadow_pct == 0.0 and highlight_pct == 0.0:
        low = torch.min(Y, dim=(1, 2), keepdim=True).values
        high = torch.max(Y, dim=(1, 2), keepdim=True).values
    else:
        low = (_percentiles_exact(Y, shadow_pct / 100.0) if precision_mode == "Exact" else _percentiles_hist(Y, shadow_pct / 100.0)) if shadow_pct > 0.0 else torch.min(Y, dim=(1, 2), keepdim=True).values
        high = (_percentiles_exact(Y, 1.0 - (highlight_pct / 100.0)) if precision_mode == "Exact" else _percentiles_hist(Y, 1.0 - (highlight_pct / 100.0))) if highlight_pct > 0.0 else torch.max(Y, dim=(1, 2), keepdim=True).values

    Ys = _linear_stretch_scalar(Y, low, high)
    ycbcr = torch.cat([Ys, ycbcr[..., 1:2], ycbcr[..., 2:3]], dim=-1)
    out = _ycbcr_to_rgb(ycbcr)
    return _clamp01(out)


def _clamp01(x):
    return torch.clamp(x, 0.0, 1.0)


def _linear_stretch(x, low, high):
    eps = 1e-6
    return torch.clamp((x - low) / torch.clamp(high - low, min=eps), 0.0, 1.0)


def _linear_stretch_scalar(x, low, high):
    eps = 1e-6
    return torch.clamp((x - low) / torch.clamp(high - low, min=eps), 0.0, 1.0)


def _luma(rgb):
    return (0.2126 * rgb[..., 0:1]) + (0.7152 * rgb[..., 1:2]) + (0.0722 * rgb[..., 2:3])


def _percentiles_exact(x, q):
    flat = _to_1d(x)
    return torch.quantile(flat, q, dim=1, keepdim=True).view(x.shape[0], 1, 1, x.shape[-1])


def _percentiles_hist(x, q):
    """Histogram-approximate percentile using torch.histc — O(B*C*bins) memory, not O(B*H*W*bins)."""
    B, H, W, C = x.shape
    flat = x.view(B, -1, C)
    bins = 256
    edges = torch.linspace(0, 1, steps=bins + 1, device=x.device, dtype=x.dtype)
    centers = (edges[:-1] + edges[1:]) / 2.0
    result = torch.zeros(B, 1, 1, C, device=x.device, dtype=x.dtype)
    for b in range(B):
        for c in range(C):
            ch = flat[b, :, c].float()
            # torch.histc is O(N) time, O(bins) memory — no one-hot expansion
            counts = torch.histc(ch, bins=bins, min=0.0, max=1.0)
            cdf = torch.cumsum(counts, dim=0)
            cdf = cdf / cdf[-1].clamp(min=1.0)
            idx = int(torch.argmax((cdf >= q).to(torch.int64)).item())
            result[b, 0, 0, c] = centers[idx]
    return result


def _rgb_to_ycbcr(rgb):
    # Expect rgb in [0,1]
    r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 0.5643 * (b - Y)
    Cr = 0.7132 * (r - Y)
    return torch.cat([Y, Cb, Cr], dim=-1)


def _to_1d(x):
    B, H, W, C = x.shape
    return x.view(B, -1, C)


def _ycbcr_to_rgb(ycbcr):
    Y, Cb, Cr = ycbcr[..., 0:1], ycbcr[..., 1:2], ycbcr[..., 2:3]
    r = Y + 1.4020 * Cr
    g = Y - 0.3441 * Cb - 0.7141 * Cr
    b = Y + 1.7720 * Cb
    return torch.cat([r, g, b], dim=-1)


class AutoAdjustNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "precision": (["Histogram (fast)", "Exact"], {"default": "Exact"}),

                "auto_levels": ("BOOLEAN", {"default": True}),
                "levels_shadow_clip_pct": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "levels_highlight_clip_pct": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "levels_gamma_normalize": ("BOOLEAN", {"default": False}),

                "auto_tone": ("BOOLEAN", {"default": True}),
                "tone_mode": (["Per-channel", "Monochromatic"], {"default": "Per-channel"}),
                "tone_shadow_clip_pct": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "tone_highlight_clip_pct": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01}),

                "auto_color": ("BOOLEAN", {"default": True}),
                "snap_neutral_midtones": ("BOOLEAN", {"default": False}),

                "flip_horizontal": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/adjustment"

    def apply(
        self,
        image,
        precision,

        auto_levels, levels_shadow_clip_pct, levels_highlight_clip_pct, levels_gamma_normalize,
        auto_tone, tone_mode, tone_shadow_clip_pct, tone_highlight_clip_pct,
        auto_color, snap_neutral_midtones,

        flip_horizontal,
    ):
        with torch.no_grad():
            # Validate input — use explicit check rather than assert (which can be
            # disabled by python -O and gives a less informative error message).
            if not (image.ndim == 4 and image.shape[-1] in (3, 4)):
                raise ValueError(
                    f"Expected IMAGE tensor [B,H,W,3 or 4], got {tuple(image.shape)}"
                )
            rgb = image[..., :3].to(dtype=torch.float32)
            rgb = _clamp01(rgb)

            if precision not in ("Histogram (fast)", "Exact"):
                precision = "Histogram (fast)"

            rgb = _auto_levels(rgb, auto_levels, levels_shadow_clip_pct, levels_highlight_clip_pct, levels_gamma_normalize, precision)
            rgb = _auto_tone(rgb, auto_tone, tone_mode, tone_shadow_clip_pct, tone_highlight_clip_pct, precision)
            rgb = _auto_color(rgb, auto_color, snap_neutral_midtones)

            if flip_horizontal:
                rgb = torch.flip(rgb, dims=[2])

            out = _clamp01(rgb)
            return (out,)

# ============================================================
# AutoColor Config (as before)
# ============================================================

class AutoColorConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_levels": ("BOOLEAN", {"default": False}),
                "auto_tone": ("BOOLEAN", {"default": False}),
                "auto_color": ("BOOLEAN", {"default": False}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("auto_levels", "auto_tone", "auto_color", "flip_horizontal")
    FUNCTION = "apply"
    CATEGORY = "config"

    def apply(self, auto_levels, auto_tone, auto_color, flip_horizontal):
        return (auto_levels, auto_tone, auto_color, flip_horizontal)


NODE_CLASS_MAPPINGS = {
    "AutoAdjustNode": AutoAdjustNode,
    "AutoColorConfigNode": AutoColorConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoAdjustNode": "Auto Adjust (Levels/Tone/Color)",
    "AutoColorConfigNode": "AutoColor Config",
}
