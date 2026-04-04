import torch
import torch.nn.functional as _iutils_F

_iutils_torch = torch


def _iutils_ensure_bhwc_rgb(image):
    t = image
    if not isinstance(t, _iutils_torch.Tensor):
        t = _iutils_torch.tensor(t)
    t = t.to(_iutils_torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    # Expect [B,H,W,C]
    if t.shape[-1] == 3:
        return t.clamp(0.0, 1.0)
    elif t.shape[-1] > 3:
        return t[..., :3].clamp(0.0, 1.0)
    elif t.shape[-1] == 1:
        return t.repeat(1, 1, 1, 3).clamp(0.0, 1.0)
    else:
        return t.clamp(0.0, 1.0)


def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    # Accept [H,W,C] or [B,H,W,C]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected [B,H,W,C] or [H,W,C], got {tuple(x.shape)}")
    return x


def _crop_bhwc(img: torch.Tensor, left: int, top: int, right: int, bottom: int, multiple: int) -> torch.Tensor:
    B, H, W, C = img.shape

    # Clamp margins to safe range
    left = max(0, min(left, W - 1))
    right = max(0, min(right, W - 1))
    top = max(0, min(top, H - 1))
    bottom = max(0, min(bottom, H - 1))

    # Compute crop box
    x0 = left
    y0 = top
    x1 = max(x0 + 1, W - right)  # ensure at least 1 px
    y1 = max(y0 + 1, H - bottom)

    # Optionally snap to a multiple (e.g., 64)
    if multiple > 1:
        # Round down width/height to a multiple, keeping the top-left fixed
        new_w = ((x1 - x0) // multiple) * multiple
        new_h = ((y1 - y0) // multiple) * multiple
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        x1 = min(W, x0 + new_w)
        y1 = min(H, y0 + new_h)

    return img[:, y0:y1, x0:x1, :]


class CropImageByMargins:
    """Crop an IMAGE tensor by pixel margins (left/top/right/bottom)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "top_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "right_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "bottom_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "snap_multiple": ("INT", {"default": 1, "min": 1, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "Image/Transform"

    def crop(self, image, left_px, top_px, right_px, bottom_px, snap_multiple=1):
        x = _to_bhwc(image)
        y = _crop_bhwc(x, left_px, top_px, right_px, bottom_px, snap_multiple)
        return (y.clamp(0.0, 1.0),)


class CropMaskByMargins:
    """Same as above but for MASK input and output (single-channel)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "left_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "top_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "right_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "bottom_px": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "snap_multiple": ("INT", {"default": 1, "min": 1, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "crop"
    CATEGORY = "Mask/Transform"

    def crop(self, mask, left_px, top_px, right_px, bottom_px, snap_multiple=1):
        # Normalize mask to [B,H,W,1]
        m = mask
        if m.dim() == 2:  # [H,W]
            m = m.unsqueeze(0).unsqueeze(-1)
        elif m.dim() == 3:
            if m.shape[0] == 1:  # [1,H,W] -> [1,H,W,1]
                m = m.permute(1, 2, 0).unsqueeze(0)
            elif m.shape[-1] == 1:  # [H,W,1] -> [1,H,W,1]
                m = m.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected MASK shape {tuple(m.shape)}")
        elif m.dim() == 4 and m.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected MASK shape {tuple(m.shape)}")

        y = _crop_bhwc(m, left_px, top_px, right_px, bottom_px, snap_multiple)
        return (y.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "CropImageByMargins": CropImageByMargins,
    "CropMaskByMargins": CropMaskByMargins,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropImageByMargins": "Crop by Margins (Image)",
    "CropMaskByMargins": "Crop by Margins (Mask)",
}
