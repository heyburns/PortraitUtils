"""Precision-preserving color engines; sampled statistics, tiled full-resolution application."""
from __future__ import annotations
import math
from functools import lru_cache
import torch
import torch.nn.functional as F
from .tensors import image_tensor, normalized, work_dtype
from .validation import number, boolean

METHODS = ["wb_grayworld", "wb_highlight", "reinhard_lab", "lab_l_only",
           "wb_highlight+reinhard"]

MATCH_METHODS = {"reinhard_lab", "lab_l_only", "wb_highlight+reinhard"}

_TILE_PIXELS = 262144

_HIST_BINS = 2048

_EPS = 1e-8

_LAB_STD_FLOOR = 0.05

# sRGB/D65 matrices and transfer curves: W3C CSS Color 4 sample conversions.
# Lab here uses D65 directly, not the D50 adaptation used by CSS lab().
_RGB_XYZ = ((506752 / 1228815, 87881 / 245763, 12673 / 70218),
            (87098 / 409605, 175762 / 245763, 12673 / 175545),
            (7918 / 409605, 87881 / 737289, 1001167 / 1053270))

_XYZ_RGB = ((12831 / 3959, -329 / 214, -1974 / 3959),
            (-851781 / 878810, 1648619 / 878810, 36519 / 878810),
            (705 / 12673, -2585 / 12673, 705 / 667))

_D65 = (0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290)

@lru_cache(maxsize=16)
def _constants(device, dtype):
    return (torch.tensor(_RGB_XYZ, device=device, dtype=dtype),
            torch.tensor(_XYZ_RGB, device=device, dtype=dtype),
            torch.tensor(_D65, device=device, dtype=dtype))

def srgb_to_linear(rgb):
    """Sign-preserving extension also supports intermediate out-of-gamut RGB."""
    absolute = rgb.abs()
    return torch.where(absolute <= 0.04045, rgb / 12.92,
                       rgb.sign() * ((absolute + 0.055) / 1.055).pow(2.4))

def linear_to_srgb(linear):
    absolute = linear.abs()
    return torch.where(absolute <= 0.0031308, linear * 12.92,
                       linear.sign() * (1.055 * absolute.pow(1 / 2.4) - 0.055))

def linear_rgb_to_lab(linear):
    matrix, _, white = _constants(linear.device, linear.dtype)
    # Matrices contain output-channel rows; BHWC pixels are row vectors.
    xyz = (linear @ matrix.T) / white
    f = torch.where(xyz > 216 / 24389, xyz.clamp_min(0).pow(1 / 3),
                    (xyz * (24389 / 27) + 16) / 116)
    x, y, z = f.unbind(-1)
    return torch.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), -1)

def rgb_to_lab(rgb):
    return linear_rgb_to_lab(srgb_to_linear(rgb))

def lab_to_rgb(lab):
    _, matrix, white = _constants(lab.device, lab.dtype)
    lightness, a, b = lab.unbind(-1)
    y = (lightness + 16) / 116
    f = torch.stack((y + a / 500, y, y - b / 200), -1)
    cube = f.pow(3)
    xyz = torch.where(cube > 216 / 24389, cube,
                      (116 * f - 16) / (24389 / 27)) * white
    return linear_to_srgb(xyz @ matrix.T)

def _work_dtype(image):
    return work_dtype(image)

def _validate_image(image, name):
    label = f"White Balance preview: {name}"
    return normalized(image_tensor(image, label, allow_unbatched=False), label)

def _rgb_weights(tile, dtype):
    tile = tile.to(dtype=dtype)
    rgb = tile[..., :3] if tile.shape[-1] != 1 else tile.expand(*tile.shape[:-1], 3)
    weights = tile[..., 3:4] if tile.shape[-1] == 4 else torch.ones_like(rgb[..., :1])
    return rgb, weights

def _tiles(frame):
    rows = max(1, _TILE_PIXELS // frame.shape[1])
    for start in range(0, frame.shape[0], rows):
        yield start, frame[start:start + rows]

def _analysis_image(image, force_size, width, height):
    """Resize a statistics sample only, retaining aspect and never enlarging."""
    if not force_size:
        return image
    scale = min(1.0, width / image.shape[2], height / image.shape[1])
    if scale == 1.0:
        return image
    size = (max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[2] * scale)))
    sample = image.to(dtype=_work_dtype(image))
    if sample.shape[-1] == 4:
        sample = torch.cat((sample[..., :3] * sample[..., 3:4], sample[..., 3:4]), -1)
    sample = F.interpolate(sample.permute(0, 3, 1, 2), size=size,
                           mode="bilinear", align_corners=False,
                           antialias=True).permute(0, 2, 3, 1)
    if sample.shape[-1] == 4:
        alpha = sample[..., 3:4].clamp(0, 1)
        rgb = torch.where(alpha > _EPS, sample[..., :3] / alpha.clamp_min(_EPS), 0)
        sample = torch.cat((rgb.clamp(0, 1), alpha), -1)
    return sample

def _wb_gains(image, highlight, percentile, preserve_luminance):
    dtype = _work_dtype(image)
    luminance_weights = _constants(image.device, dtype)[0][1]
    results = []
    for frame in image:
        threshold = None
        if highlight:
            histogram = torch.zeros(_HIST_BINS, device=image.device, dtype=dtype)
            for _, tile in _tiles(frame):
                rgb, alpha = _rgb_weights(tile, dtype)
                luminance = srgb_to_linear(rgb) @ luminance_weights
                eligible = (rgb.amax(-1) < 0.995) & (luminance > _EPS)
                weights = alpha[..., 0] * eligible
                indices = (luminance.clamp(0, 1) * (_HIST_BINS - 1)).long()
                histogram += torch.bincount(indices.reshape(-1), weights=weights.reshape(-1),
                                            minlength=_HIST_BINS)
            cumulative = histogram.cumsum(0)
            threshold = torch.searchsorted(cumulative, cumulative[-1] * (percentile / 100))
            threshold = threshold.to(dtype) / (_HIST_BINS - 1)

        channel_sum = torch.zeros(3, device=image.device, dtype=dtype)
        count = torch.zeros((), device=image.device, dtype=dtype)
        for _, tile in _tiles(frame):
            rgb, weights = _rgb_weights(tile, dtype)
            linear = srgb_to_linear(rgb)
            if highlight:
                luminance = linear @ luminance_weights
                selected = ((rgb.amax(-1) < 0.995) & (luminance > _EPS)
                            & (luminance >= threshold))
                weights = weights * selected[..., None]
            channel_sum += (linear * weights).sum((0, 1))
            count += weights.sum()
        mean = channel_sum / count.clamp_min(_EPS)
        if preserve_luminance:
            neutral = mean @ luminance_weights
        elif highlight:
            neutral = srgb_to_linear(mean.new_tensor(0.95))
        else:
            neutral = mean.mean()
        gains = (neutral / mean.clamp_min(_EPS)).clamp(0.25, 4.0)
        reliable = (count > _EPS) & (mean.amin() > _EPS)
        results.append(torch.where(reliable, gains, torch.ones_like(gains)))
    return torch.stack(results)

def _lab_moments(image, gains=None):
    """Weighted, streaming population moments; hidden RGB does not contribute."""
    dtype = _work_dtype(image)
    means, deviations, totals = [], [], []
    for index, frame in enumerate(image):
        total = torch.zeros((), device=image.device, dtype=dtype)
        mean = torch.zeros(3, device=image.device, dtype=dtype)
        m2 = torch.zeros_like(mean)
        for _, tile in _tiles(frame):
            rgb, weights = _rgb_weights(tile, dtype)
            linear = srgb_to_linear(rgb)
            if gains is not None:
                linear = linear * gains[index]
            lab = linear_rgb_to_lab(linear)
            count = weights.sum()
            tile_mean = (lab * weights).sum((0, 1)) / count.clamp_min(_EPS)
            tile_m2 = ((lab - tile_mean).square() * weights).sum((0, 1))
            new_total = total + count
            delta = tile_mean - mean
            m2 = m2 + tile_m2 + delta.square() * total * count / new_total.clamp_min(_EPS)
            mean = mean + delta * count / new_total.clamp_min(_EPS)
            total = new_total
        means.append(mean)
        deviations.append((m2 / total.clamp_min(_EPS)).clamp_min(0).sqrt())
        totals.append(total)
    return torch.stack(means), torch.stack(deviations), torch.stack(totals)


class WhiteBalanceEngine:
    @torch.no_grad()
    def run(self, image, reference, method="wb_highlight+reinhard", percentile=95.0,
            strength=1.0, clip_gamut=True, force_size=False, target_width=1440,
            target_height=1080, preserve_luminance=True):
        _validate_image(image, "image")
        for name, value in (("clip_gamut", clip_gamut), ("force_size", force_size),
                            ("preserve_luminance", preserve_luminance)):
            boolean(value, f"White Balance preview.{name}")
        if method not in METHODS:
            raise ValueError(f"White Balance preview: unknown method {method!r}; choose one of {METHODS}.")
        number(strength, "White Balance preview: strength", 0, 1)
        number(percentile, "White Balance preview: percentile", 80, 99.9)
        if force_size and any(isinstance(v, bool) or not isinstance(v, int) or not 16 <= v <= 8192
                              for v in (target_width, target_height)):
            raise ValueError("White Balance preview: statistics target dimensions must be integers from 16 to 8192.")
        if strength == 0:
            return (image,)

        matching = method in MATCH_METHODS
        if matching:
            _validate_image(reference, "reference")
            if reference.shape[0] not in (1, image.shape[0]):
                raise ValueError("White Balance preview: reference batch must contain one image or "
                                 "the same number of images as the source batch.")
            if reference is image and method in {"reinhard_lab", "lab_l_only"}:
                return (image,)

        sample = _analysis_image(image, force_size, target_width, target_height)
        gains = None
        if method in {"wb_grayworld", "wb_highlight", "wb_highlight+reinhard"}:
            gains = _wb_gains(sample, method != "wb_grayworld", float(percentile), preserve_luminance)
        slope = bias = None
        if matching:
            reference_sample = _analysis_image(reference, force_size, target_width, target_height)
            src_mean, src_std, _ = _lab_moments(sample, gains)
            ref_mean, ref_std, ref_weight = _lab_moments(reference_sample)
            if bool((ref_weight <= _EPS).any()):
                raise ValueError("White Balance preview: reference has no visible pixels; "
                                 "connect a nontransparent reference image.")
            ref_mean = ref_mean.to(device=image.device, dtype=src_mean.dtype)
            ref_std = ref_std.to(device=image.device, dtype=src_std.dtype)
            reliable = (src_std >= _LAB_STD_FLOOR) & (ref_std >= _LAB_STD_FLOOR)
            slope = torch.where(reliable, (ref_std / src_std.clamp_min(_LAB_STD_FLOOR)).clamp(0.125, 8), 1)
            bias = ref_mean - slope * src_mean
            if method == "lab_l_only":
                slope[:, 1:] = 1
                bias[:, 1:] = 0

        channels = 4 if image.shape[-1] == 4 else 3
        output = torch.empty((*image.shape[:-1], channels), device=image.device, dtype=image.dtype)
        dtype = _work_dtype(image)
        for index, frame in enumerate(image):
            for start, tile in _tiles(frame):
                rgb, _ = _rgb_weights(tile, dtype)
                linear = srgb_to_linear(rgb)
                if gains is not None:
                    linear = linear * gains[index]
                if matching:
                    corrected = lab_to_rgb(linear_rgb_to_lab(linear) * slope[index] + bias[index])
                else:
                    corrected = linear_to_srgb(linear)
                corrected = rgb + float(strength) * (corrected - rgb)
                if clip_gamut:
                    corrected = corrected.clamp(0, 1)
                if channels == 4:
                    corrected = torch.where(tile[..., 3:4] > 0, corrected, rgb)
                    corrected = torch.cat((corrected, tile[..., 3:4]), -1)
                output[index, start:start + tile.shape[0]] = corrected.to(dtype=image.dtype)
        return (output,)
