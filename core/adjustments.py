"""Photoshop-style automatic corrections, independent of node interfaces."""
import math
import torch
from .tensors import image_tensor, normalized, work_dtype
from .validation import choice, number, boolean

_EPSILON = 1e-6

_MIN_TONAL_SPAN = 1.0 / 255.0

_AUTO_COLOR_TAIL_FRACTION = 0.005

def _normalize_valid_mask(valid_mask, reference):
    if valid_mask is None:
        return None
    mask = valid_mask
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=reference.device)
    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 4 or mask.shape[:3] != reference.shape[:3]:
        raise ValueError(
            "valid_mask must be [B,H,W] or [B,H,W,1] and match the image"
        )
    if mask.shape[-1] not in (1, reference.shape[-1]):
        raise ValueError("valid_mask must have one channel or match the image channels")
    return mask

def _mask_for_channel(valid_mask, batch_index, channel_index):
    if valid_mask is None:
        return None
    channel = 0 if valid_mask.shape[-1] == 1 else channel_index
    return valid_mask[batch_index, ..., channel]

def _extreme(x, minimum, valid_mask=None):
    valid_mask = _normalize_valid_mask(valid_mask, x)
    if valid_mask is None:
        operation = torch.amin if minimum else torch.amax
        return operation(x, dim=(1, 2), keepdim=True)

    batch, _, _, channels = x.shape
    result = torch.empty(batch, 1, 1, channels, device=x.device, dtype=x.dtype)
    for b in range(batch):
        for c in range(channels):
            values = x[b, ..., c]
            mask = _mask_for_channel(valid_mask, b, c)
            selected = values[mask]
            if selected.numel() == 0:
                selected = values.reshape(-1)
            result[b, 0, 0, c] = (
                torch.amin(selected) if minimum else torch.amax(selected)
            )
    return result

def _percentiles_exact(x, q, valid_mask=None):
    valid_mask = _normalize_valid_mask(valid_mask, x)
    if valid_mask is None:
        flat = x.reshape(x.shape[0], -1, x.shape[-1])
        return torch.quantile(flat.to(dtype=work_dtype(flat)), q, dim=1, keepdim=True).to(
            dtype=x.dtype
        ).view(x.shape[0], 1, 1, x.shape[-1])

    batch, _, _, channels = x.shape
    result = torch.empty(batch, 1, 1, channels, device=x.device, dtype=x.dtype)
    for b in range(batch):
        for c in range(channels):
            values = x[b, ..., c]
            mask = _mask_for_channel(valid_mask, b, c)
            selected = values[mask]
            if selected.numel() == 0:
                selected = values.reshape(-1)
            result[b, 0, 0, c] = torch.quantile(selected.to(dtype=work_dtype(selected)), q).to(
                dtype=x.dtype
            )
    return result

def _percentiles_hist(x, q, valid_mask=None):
    """Approximate percentiles with a Photoshop-like 256-bin histogram."""
    valid_mask = _normalize_valid_mask(valid_mask, x)
    batch, _, _, channels = x.shape
    flat = x.reshape(batch, -1, channels)
    bins = 256
    centers = torch.linspace(
        0.5 / bins,
        1.0 - 0.5 / bins,
        steps=bins,
        device=x.device,
        dtype=x.dtype,
    )
    result = torch.empty(batch, 1, 1, channels, device=x.device, dtype=x.dtype)
    for b in range(batch):
        for c in range(channels):
            values = flat[b, :, c]
            mask = _mask_for_channel(valid_mask, b, c)
            if mask is not None:
                selected = values[mask.reshape(-1)]
                if selected.numel() != 0:
                    values = selected
            counts = torch.histc(values.float(), bins=bins, min=0.0, max=1.0)
            cdf = torch.cumsum(counts, dim=0)
            cdf = cdf / cdf[-1].clamp(min=1.0)
            index = int(torch.argmax((cdf >= q).to(torch.int64)).item())
            result[b, 0, 0, c] = centers[index]
    return result

def _percentile(x, q, precision_mode, valid_mask=None):
    q = max(0.0, min(1.0, float(q)))
    if precision_mode == "Exact":
        return _percentiles_exact(x, q, valid_mask)
    return _percentiles_hist(x, q, valid_mask)

def _tonal_bounds(x, shadow_pct, highlight_pct, precision_mode, valid_mask=None):
    shadow = max(0.0, min(100.0, float(shadow_pct))) / 100.0
    highlight = max(0.0, min(100.0, float(highlight_pct))) / 100.0
    low = (
        _extreme(x, True, valid_mask)
        if shadow == 0.0
        else _percentile(x, shadow, precision_mode, valid_mask)
    )
    high = (
        _extreme(x, False, valid_mask)
        if highlight == 0.0
        else _percentile(x, 1.0 - highlight, precision_mode, valid_mask)
    )
    return low, high

def _linear_stretch(x, low, high):
    span = high - low
    mapped = torch.clamp((x - low) / span.clamp(min=_EPSILON), 0.0, 1.0)
    # Preserve flat or nearly-flat channels rather than manufacturing contrast.
    return torch.where(span >= _MIN_TONAL_SPAN, mapped, x)

def _linear_stretch_scalar(x, low, high):
    return _linear_stretch(x, low, high)

def _luma(rgb):
    return (
        0.2126 * rgb[..., 0:1]
        + 0.7152 * rgb[..., 1:2]
        + 0.0722 * rgb[..., 2:3]
    )

def _apply_composite_gamma(rgb, valid_mask=None):
    """Gently place the image median near 50% using a Levels-style gamma."""
    output = rgb.clone()
    valid_mask = _normalize_valid_mask(valid_mask, rgb)
    for b in range(rgb.shape[0]):
        luminance = _luma(output[b : b + 1])
        mask = (luminance > 0.01) & (luminance < 0.99)
        if valid_mask is not None:
            mask &= valid_mask[b : b + 1, ..., :1]
        values = luminance[mask]
        if values.numel() == 0:
            continue
        median = float(torch.quantile(values.float(), 0.5).item())
        if not 0.02 < median < 0.98:
            continue
        gamma = math.log(0.5) / math.log(median)
        gamma = max(0.75, min(1.33, gamma))
        positive = output[b : b + 1] > 0.0
        corrected = output[b : b + 1].clamp(min=_EPSILON).pow(gamma)
        output[b : b + 1] = torch.where(
            positive, corrected, torch.zeros_like(corrected)
        )
    return output

def _auto_levels(
    rgb,
    use_levels,
    shadow_pct,
    highlight_pct,
    gamma_norm,
    precision_mode,
    valid_mask=None,
):
    """Photoshop Auto Contrast / monochromatic-contrast approximation."""
    if not use_levels:
        return rgb
    luminance = _luma(rgb)
    low, high = _tonal_bounds(
        luminance, shadow_pct, highlight_pct, precision_mode, valid_mask
    )
    stretched = _linear_stretch_scalar(rgb, low, high)
    if gamma_norm:
        stretched = _apply_composite_gamma(stretched, valid_mask)
    return stretched.clamp(0.0, 1.0)

def _auto_tone(
    rgb,
    use_tone,
    mode,
    shadow_pct,
    highlight_pct,
    precision_mode,
    valid_mask=None,
):
    """Photoshop Auto Tone (per-channel) or Auto Contrast approximation."""
    if not use_tone:
        return rgb
    if mode == "Monochromatic":
        return _auto_levels(
            rgb,
            True,
            shadow_pct,
            highlight_pct,
            False,
            precision_mode,
            valid_mask,
        )
    low, high = _tonal_bounds(
        rgb, shadow_pct, highlight_pct, precision_mode, valid_mask
    )
    return _linear_stretch(rgb, low, high).clamp(0.0, 1.0)

def _masked_average(image, mask):
    weights = mask.to(dtype=image.dtype)
    count = weights.sum().clamp(min=1.0)
    return (image * weights).sum(dim=(1, 2), keepdim=True) / count

def _snap_neutral_midtones(rgb, valid_mask=None, reference_rgb=None):
    """Neutralize an actually low-chroma midtone using per-channel gamma.

    This deliberately refuses to infer a gray reference from saturated content.
    It is safer to skip a correction than to turn skin, foliage, or sunset light
    gray merely because that color dominates the frame.
    """
    output = rgb.clone()
    reference_rgb = rgb if reference_rgb is None else reference_rgb
    if reference_rgb.shape != rgb.shape:
        raise ValueError("reference_rgb must match the adjusted RGB image")
    valid_mask = _normalize_valid_mask(valid_mask, rgb)
    for b in range(rgb.shape[0]):
        image = output[b : b + 1]
        reference = reference_rgb[b : b + 1]
        luminance = _luma(reference)
        maximum = torch.amax(reference, dim=-1, keepdim=True)
        minimum = torch.amin(reference, dim=-1, keepdim=True)
        saturation = (maximum - minimum) / maximum.clamp(min=_EPSILON)
        midtone_mask = (luminance > 0.2) & (luminance < 0.8)
        if valid_mask is not None:
            midtone_mask &= valid_mask[b : b + 1, ..., :1]
        midtone_saturation = saturation[midtone_mask]
        if midtone_saturation.numel() == 0:
            continue

        low_chroma_quantile = float(
            torch.quantile(midtone_saturation.float(), 0.10).item()
        )
        neutral_cutoff = min(0.25, max(0.025, low_chroma_quantile + 0.02))
        neutral_mask = midtone_mask & (saturation <= neutral_cutoff)
        candidate_count = int(neutral_mask.sum().item())
        minimum_candidates = max(8, int(midtone_mask.sum().item()) // 10000)
        if candidate_count < minimum_candidates:
            continue

        chroma_weight = (
            1.0 - saturation / max(neutral_cutoff, _EPSILON)
        ).clamp(0.0, 1.0)
        tone_weight = (1.0 - 2.0 * (luminance - 0.5).abs()).clamp(0.05, 1.0)
        weights = neutral_mask.to(image.dtype) * chroma_weight.square() * tone_weight
        weight_sum = weights.sum().clamp(min=_EPSILON)
        # The candidate locations come from the unadjusted image, but gamma is
        # solved from those same pixels after dark/light endpoint correction.
        neutral = (image * weights).sum(dim=(1, 2), keepdim=True) / weight_sum
        neutral = neutral.clamp(0.02, 0.98)
        gamma = math.log(0.5) / torch.log(neutral)
        gamma = gamma.clamp(0.5, 2.0)
        positive = image > 0.0
        corrected = image.clamp(min=_EPSILON).pow(gamma)
        output[b : b + 1] = torch.where(
            positive, corrected, torch.zeros_like(corrected)
        )
    return output.clamp(0.0, 1.0)

def _auto_color(
    rgb,
    use_color,
    snap_midtones,
    shadow_pct=0.1,
    highlight_pct=0.1,
    precision_mode="Histogram (fast)",
    valid_mask=None,
):
    """Approximate Photoshop's Find Dark & Light Colors algorithm."""
    if not use_color:
        return rgb

    valid_mask = _normalize_valid_mask(valid_mask, rgb)
    output = rgb.clone()
    luminance = _luma(rgb)
    shadow_tail = max(
        _AUTO_COLOR_TAIL_FRACTION, max(0.0, float(shadow_pct)) / 100.0
    )
    highlight_tail = max(
        _AUTO_COLOR_TAIL_FRACTION, max(0.0, float(highlight_pct)) / 100.0
    )
    dark_threshold = _percentile(luminance, shadow_tail, precision_mode, valid_mask)
    light_threshold = _percentile(
        luminance, 1.0 - highlight_tail, precision_mode, valid_mask
    )
    fallback_low, fallback_high = _tonal_bounds(
        rgb, shadow_pct, highlight_pct, precision_mode, valid_mask
    )

    for b in range(rgb.shape[0]):
        valid = torch.ones_like(luminance[b : b + 1], dtype=torch.bool)
        if valid_mask is not None:
            valid &= valid_mask[b : b + 1, ..., :1]
        if not bool(valid.any()):
            continue
        dark_mask = valid & (luminance[b : b + 1] <= dark_threshold[b : b + 1])
        light_mask = valid & (luminance[b : b + 1] >= light_threshold[b : b + 1])
        if not bool(dark_mask.any()) or not bool(light_mask.any()):
            continue

        dark_color = _masked_average(rgb[b : b + 1], dark_mask)
        light_color = _masked_average(rgb[b : b + 1], light_mask)
        color_span = light_color - dark_color
        low = torch.where(
            color_span >= _MIN_TONAL_SPAN,
            dark_color,
            fallback_low[b : b + 1],
        )
        high = torch.where(
            color_span >= _MIN_TONAL_SPAN,
            light_color,
            fallback_high[b : b + 1],
        )
        output[b : b + 1] = _linear_stretch(rgb[b : b + 1], low, high)

    if snap_midtones:
        output = _snap_neutral_midtones(output, valid_mask, reference_rgb=rgb)
    return output.clamp(0.0, 1.0)

def _rgb_to_ycbcr(rgb):
    """Reversible full-range Rec.709 YCbCr helper retained for compatibility."""
    red, green, blue = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
    luminance = _luma(rgb)
    cb = (blue - luminance) / 1.8556
    cr = (red - luminance) / 1.5748
    return torch.cat([luminance, cb, cr], dim=-1)

def _ycbcr_to_rgb(ycbcr):
    luminance, cb, cr = ycbcr[..., 0:1], ycbcr[..., 1:2], ycbcr[..., 2:3]
    red = luminance + 1.5748 * cr
    green = luminance - 0.1873 * cb - 0.4681 * cr
    blue = luminance + 1.8556 * cb
    return torch.cat([red, green, blue], dim=-1)

def _to_1d(x):
    return x.reshape(x.shape[0], -1, x.shape[-1])

def _median_masked(channel, mask):
    """Per-image masked median retained for compatibility with older callers."""
    result = torch.zeros(
        channel.shape[0], 1, 1, 1, device=channel.device, dtype=channel.dtype
    )
    for b in range(channel.shape[0]):
        selected = (
            channel[b][mask[b]] if bool(mask[b].any()) else channel[b].reshape(-1)
        )
        result[b] = torch.quantile(selected.to(dtype=work_dtype(selected)), 0.5).to(channel.dtype)
    return result


class AutoAdjustmentEngine:
    def apply(
        self,
        image,
        precision,
        auto_levels,
        levels_shadow_clip_pct,
        levels_highlight_clip_pct,
        levels_gamma_normalize,
        auto_tone,
        tone_mode,
        tone_shadow_clip_pct,
        tone_highlight_clip_pct,
        auto_color,
        snap_neutral_midtones,
        flip_horizontal,
        adjustment_mode="Use switches (legacy)",
        strength=1.0,
    ):
        with torch.no_grad():
            choice(precision, "Auto Adjust.precision", ("Histogram (fast)", "Exact"))
            choice(tone_mode, "Auto Adjust.tone_mode", ("Per-channel", "Monochromatic"))
            choice(adjustment_mode, "Auto Adjust.adjustment_mode", ("Use switches (legacy)",
                   "Auto Levels / Contrast", "Auto Tone", "Auto Color"))
            for name, value in (("levels_shadow_clip_pct", levels_shadow_clip_pct),
                                ("levels_highlight_clip_pct", levels_highlight_clip_pct),
                                ("tone_shadow_clip_pct", tone_shadow_clip_pct),
                                ("tone_highlight_clip_pct", tone_highlight_clip_pct)):
                number(value, f"Auto Adjust.{name}", 0, 5)
            for name, value in (("auto_levels", auto_levels), ("auto_tone", auto_tone),
                                ("auto_color", auto_color), ("levels_gamma_normalize", levels_gamma_normalize),
                                ("snap_neutral_midtones", snap_neutral_midtones), ("flip_horizontal", flip_horizontal)):
                boolean(value, f"Auto Adjust.{name}")
            number(strength, "Auto Adjust.strength", 0, 1)
            image = normalized(image_tensor(image), "Auto Adjust.image")
            if adjustment_mode == "Auto Levels / Contrast":
                auto_levels, auto_tone, auto_color = True, False, False
            elif adjustment_mode == "Auto Tone":
                auto_levels, auto_tone, auto_color = False, True, False
            elif adjustment_mode == "Auto Color":
                auto_levels, auto_tone, auto_color = False, False, True
            if strength == 0 or not (auto_levels or auto_tone or auto_color):
                return (torch.flip(image, dims=[2]) if flip_horizontal else image,)
            has_alpha = image.shape[-1] == 4
            dtype = work_dtype(image)
            if image.shape[-1] == 1:
                rgb = image.expand(*image.shape[:-1], 3).to(dtype=dtype)
            else:
                rgb = image[..., :3].to(dtype=dtype)
            original_rgb = rgb

            alpha = None
            valid_mask = None
            if has_alpha:
                alpha = image[..., 3:4].to(dtype=dtype)
                valid_mask = alpha > _EPSILON

            rgb = _auto_levels(
                rgb,
                auto_levels,
                levels_shadow_clip_pct,
                levels_highlight_clip_pct,
                levels_gamma_normalize,
                precision,
                valid_mask,
            )
            rgb = _auto_tone(
                rgb,
                auto_tone,
                tone_mode,
                tone_shadow_clip_pct,
                tone_highlight_clip_pct,
                precision,
                valid_mask,
            )
            rgb = _auto_color(
                rgb,
                auto_color,
                snap_neutral_midtones,
                levels_shadow_clip_pct,
                levels_highlight_clip_pct,
                precision,
                valid_mask,
            )

            blend = float(strength)
            if blend < 1.0:
                rgb = torch.lerp(original_rgb, rgb, blend)

            if flip_horizontal:
                rgb = torch.flip(rgb, dims=[2])
                if alpha is not None:
                    alpha = torch.flip(alpha, dims=[2])
            output = rgb.clamp(0.0, 1.0)
            if alpha is not None:
                output = torch.cat([output, alpha], dim=-1)
            return (output.to(dtype=image.dtype),)
