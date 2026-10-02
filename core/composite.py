"""Alpha-aware, tensor-preserving masked composition."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .tensors import image_tensor, mask_tensor, normalized, finite_float, integer
from .contracts import EditConfigV2, _expect
from .tensors import resize_image
from .validation import boolean

def _resize(image, height, width):
    return resize_image(image, width, height, "bilinear")

def _feather(mask, radius):
    if not radius:
        return mask
    sigma = max(0.5, radius / 2)
    if mask.device.type == "cpu":
        # OpenCV's separable implementation avoids large 2D disk convolutions.
        blurred = np.stack([cv2.GaussianBlur(frame.detach().numpy(), (2 * radius + 1,) * 2,
                            sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
                            for frame in mask])
        halo = torch.from_numpy(blurred).to(dtype=mask.dtype)
    else:
        axis = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
        kernel = torch.exp(-axis.square() / (2 * sigma * sigma))
        kernel /= kernel.sum()
        halo = F.conv2d(F.pad(mask[:, None], (radius, radius, 0, 0), mode="replicate"),
                        kernel.view(1, 1, 1, -1))
        halo = F.conv2d(F.pad(halo, (0, 0, radius, radius), mode="replicate"),
                        kernel.view(1, 1, -1, 1))[:, 0]
    return torch.maximum(mask, halo).clamp(0, 1)

def _channels(image, count):
    if image.shape[-1] == 1 and count != 1:
        image = image.expand(*image.shape[:-1], 3)
    if image.shape[-1] == 3 and count == 4:
        image = torch.cat((image, torch.ones_like(image[..., :1])), -1)
    return image


class CompositeEngine:
    @torch.no_grad()
    def blend(self, image_a, image_b, edit_config, opacity_source="stitch_opacity",
              invert_mask=False, bypass_mask=False, feather_radius=5, force_size=False,
              target_width=1344, target_height=768, mask=None):
        config = _expect(edit_config, EditConfigV2, "edit_config")
        for name, value in (("invert_mask", invert_mask), ("bypass_mask", bypass_mask), ("force_size", force_size)):
            boolean(value, f"Stitch preview.{name}")
        if opacity_source not in ("blend_opacity", "stitch_opacity"):
            raise ValueError("Stitch preview: opacity_source must be blend_opacity or stitch_opacity.")
        opacity = finite_float(getattr(config, opacity_source), "Stitch preview: opacity", 0, 1)
        integer(feather_radius, "Stitch preview: feather_radius", 0, 100)
        a = normalized(image_tensor(image_a, "Stitch preview: image_a"), "Stitch preview: image_a")
        b = normalized(image_tensor(image_b, "Stitch preview: image_b"), "Stitch preview: image_b")
        if b.shape[0] not in (1, a.shape[0]):
            raise ValueError("Stitch preview: image_b batch must contain one image or match image_a's batch.")
        dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
        original_dtype = a.dtype
        a = a.to(dtype=dtype)
        b = b.to(device=a.device, dtype=dtype)
        if force_size:
            integer(target_width, "Stitch preview: target_width", 16, 8192)
            integer(target_height, "Stitch preview: target_height", 16, 8192)
            a, b = _resize(a, target_height, target_width), _resize(b, target_height, target_width)
        elif a.shape[1:3] != b.shape[1:3]:
            raise ValueError(f"Stitch preview: size mismatch A={tuple(a.shape[1:3])}, "
                             f"B={tuple(b.shape[1:3])}; align inputs or enable force_size.")
        if bypass_mask:
            effective = torch.full(a.shape[:3], opacity, device=a.device, dtype=dtype)
        else:
            if mask is None:
                raise ValueError("Stitch preview: connect mask or enable bypass_mask.")
            coverage = normalized(mask_tensor(mask, "Stitch preview: mask"), "Stitch preview: mask")
            if coverage.shape[0] not in (1, a.shape[0]):
                raise ValueError("Stitch preview: mask batch must contain one mask or match image_a's batch.")
            coverage = coverage.to(device=a.device, dtype=dtype)
            if coverage.shape[1:3] != a.shape[1:3]:
                if not force_size:
                    raise ValueError("Stitch preview: mask size mismatch; align inputs or enable force_size.")
                coverage = F.interpolate(coverage[:, None], size=a.shape[1:3], mode="nearest")[:, 0]
            if invert_mask:
                coverage = 1 - coverage
            effective = _feather(coverage, feather_radius) * opacity
            effective = effective.expand(a.shape[:3])
        count = max(a.shape[-1], b.shape[-1])
        a, b = _channels(a, count), _channels(b, count)
        blend = effective[..., None]
        if count == 4:
            alpha = a[..., 3:4] * (1 - blend) + b[..., 3:4] * blend
            rgb = (a[..., :3] * a[..., 3:4] * (1 - blend)
                   + b[..., :3] * b[..., 3:4] * blend) / alpha.clamp_min(1e-8)
            hidden = a[..., :3] + blend * (b[..., :3] - a[..., :3])
            result = torch.cat((torch.where(alpha > 1e-8, rgb, hidden), alpha), -1)
        else:
            result = a + blend * (b - a)
        # Endpoints are exact even for RGBA: alpha unpremultiplication must not
        # alter a fully selected image's hidden pixels or float rounding.
        result = torch.where(blend == 0, a, torch.where(blend == 1, b, result))
        return result.clamp(0, 1).to(original_dtype), effective.to(original_dtype)
