"""Strict layout helpers for preview nodes; never silently quantize pixels."""

import torch
import torch.nn.functional as F
from .contracts import PADDING_FILLS
from .validation import choice, integer, number as finite_float
from .resampling import lanczos_resize


def image_tensor(value, label="image", *, allow_unbatched=True):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{label}: expected a floating-point IMAGE tensor. "
                         "Connect the loader's image output or another IMAGE, not a MASK/config bundle.")
    if value.ndim == 3 and allow_unbatched:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] not in (1, 3, 4) or min(value.shape) < 1:
        raise ValueError(f"{label}: expected [batch, height, width, 1/3/4] IMAGE layout (shape). "
                         "Connect the loader's image output or another IMAGE, not a MASK/config bundle.")
    return value


def legacy_image_tensor(image, force_rgb=False):
    """Explicit compatibility conversion for legacy nodes, not the new engine contract."""
    tensor = image if isinstance(image, torch.Tensor) else torch.tensor(image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(tensor.shape)}")
    tensor = tensor.to(dtype=torch.float32)
    if force_rgb:
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 1, 3)
        elif tensor.shape[-1] > 3:
            tensor = tensor[..., :3]
    return tensor.clamp(0, 1)


def mask_tensor(value, label="mask"):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{label}: expected a floating-point MASK tensor.")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    # Every 3D MASK is BHW, including batches and one-pixel-wide images.
    if value.ndim != 3 or min(value.shape) < 1:
        raise ValueError(f"{label}: expected [batch, height, width] MASK layout "
                         "(also accepts HW or BHW1). For HWC1, add a batch dimension.")
    return value


def normalized(value, label):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label}: image/mask contains NaN or infinite pixels.")
    if bool((value.amin() < 0) | (value.amax() > 1)):
        raise ValueError(f"{label}: expected normalized 0–1 pixels; convert HDR/out-of-gamut data first.")
    return value


def work_dtype(image):
    return torch.float64 if image.dtype == torch.float64 else torch.float32


def aligned_mask(value, height, width, device, label="mask", *, resize=False):
    """Explicit alignment, never infer HWC from a three-dimensional MASK."""
    if value is None:
        return torch.zeros((1, height, width), dtype=torch.float32, device=device)
    mask = normalized(mask_tensor(value, label), label).to(device=device)
    if mask.shape[0] != 1:
        raise ValueError(f"{label}: preparation requires a single MASK (B=1).")
    if mask.shape[-2:] != (height, width):
        if not resize:
            raise ValueError(f"{label}: mask size {tuple(mask.shape[-2:])} does not match "
                             f"image {(height, width)}. Align both before preparation.")
        mask = resize_mask(mask, width, height)
    return mask


def pad_crop(image, mask, padding, padding_fill="Edge extension"):
    choice(padding_fill, "padding_fill", PADDING_FILLS)
    if not any(padding):
        return image, mask
    # Only the canvas is synthesized; native_crop remains an exact source slice.
    if padding_fill == "Edge extension":
        padded = F.pad(image.movedim(-1, 1), padding, mode="replicate").movedim(1, -1)
    else:
        left, right, top, bottom = padding
        batch, height, width, channels = image.shape
        padded = image.new_full((batch, height + top + bottom, width + left + right, channels),
                                0 if padding_fill == "Black" else 1)
        if channels == 4:
            # Solid borders are opaque, including black on an RGBA source.
            padded[..., 3] = 1
        padded[:, top:top + height, left:left + width, :] = image
    return padded, F.pad(mask, padding, mode="constant", value=0) if mask is not None else None


def resize_image(image, width, height, method="bicubic", resampler=None):
    """One final resize; retain dtype/device/channels and avoid RGBA edge halos."""
    if image.shape[1:3] == (height, width):
        return image
    dtype = image.dtype
    samples = image.to(dtype=work_dtype(image))
    alpha = samples.shape[-1] == 4
    if alpha:
        samples = torch.cat((samples[..., :3] * samples[..., 3:4], samples[..., 3:4]), -1)
    samples = samples.movedim(-1, 1)
    if method == "lanczos":
        samples = (resampler or lanczos_resize)(samples, width, height)
    elif method in ("bicubic", "bilinear"):
        samples = F.interpolate(samples, size=(height, width), mode=method,
                                align_corners=False, antialias=True)
    elif method == "area":
        samples = F.interpolate(samples, size=(height, width), mode="area")
    else:
        raise ValueError(f"resize_method: unsupported method {method!r}.")
    result = samples.movedim(1, -1)
    if alpha:
        raw_coverage = result[..., 3:4]
        rgb = torch.where(raw_coverage > 1e-8, result[..., :3] / raw_coverage.clamp_min(1e-8), 0)
        result = torch.cat((rgb, raw_coverage.clamp(0, 1)), -1)
    # Antialiased reconstruction may overshoot. Never clamp literal crops.
    return result.clamp(0, 1).to(dtype=dtype)


def resize_mask(mask, width, height):
    if mask.shape[-2:] == (height, width):
        return mask
    dtype = mask.dtype
    work = mask.to(dtype=work_dtype(mask))
    return F.interpolate(work[:, None], size=(height, width), mode="bilinear",
                         align_corners=False)[:, 0].clamp(0, 1).to(dtype=dtype)
