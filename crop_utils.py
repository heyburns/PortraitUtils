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


def _iutils_dilate(mask):
    k = _iutils_torch.ones((1, 1, 3, 3), device=mask.device)
    return (_iutils_F.conv2d(mask, k, padding=1) > 0).float()


def _iutils_rgb2y(img_bhwc):
    r, g, b = img_bhwc[..., 0], img_bhwc[..., 1], img_bhwc[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(-1)


def _iutils_region_grow(img, seed_mask, border_rgb, thr_ch, max_iter=4096):
    B, H, W, Cp = img.shape
    dist = (img - border_rgb).abs().sum(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    thr_l1 = thr_ch.sum(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    eligible = (dist <= thr_l1).float()
    cur = seed_mask.clone()
    for _ in range(max_iter):
        grown = _iutils_dilate(cur) * eligible
        if _iutils_torch.equal(grown, cur):
            break
        cur = grown
    return cur


def _iutils_bbox_from_mask(nonborder, pad):
    B, _, H, W = nonborder.shape
    boxes = []
    for b in range(B):
        m = nonborder[b, 0]
        ys = _iutils_torch.where(m.any(dim=1))[0]
        xs = _iutils_torch.where(m.any(dim=0))[0]
        if len(ys) == 0 or len(xs) == 0:
            l, t, w, h = 0, 0, W, H
        else:
            top, bottom = int(ys[0].item()), int(ys[-1].item())
            left, right = int(xs[0].item()), int(xs[-1].item())
            l = max(0, left - pad)
            t = max(0, top - pad)
            r = min(W, right + 1 + pad)
            btm = min(H, bottom + 1 + pad)
            w = max(1, r - l)
            h = max(1, btm - t)
        boxes.append((l, t, w, h))
    return boxes


class AutoCropBorders:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fuzz_mode": (["percent", "adaptive"], {"default": "adaptive"}),
                "fuzz_percent": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "adaptive_k": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "edge_margin_px": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "pad_px": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "use_luma_only": ("BOOLEAN", {"default": False}),
                "max_growth_iter": ("INT", {"default": 4096, "min": 64, "max": 16384, "step": 64}),
                "return_border_mask": ("BOOLEAN", {"default": False}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "MASK")
    RETURN_NAMES = ("image", "left", "top", "width", "height", "border_mask")
    FUNCTION = "run"
    CATEGORY = "image/transform"

    def run(
        self,
        image,
        fuzz_mode="adaptive",
        fuzz_percent=5.0,
        adaptive_k=2.0,
        edge_margin_px=4,
        pad_px=0,
        use_luma_only=False,
        max_growth_iter=4096,
        return_border_mask=False,
        use_gpu=True,
    ):

        img = _iutils_ensure_bhwc_rgb(image)
        target_dev = (
            _iutils_torch.device("cuda")
            if (use_gpu and _iutils_torch.cuda.is_available())
            else img.device
        )
        if img.device != target_dev:
            img = img.to(target_dev, non_blocking=True)

        B, H, W, C = img.shape
        work = _iutils_rgb2y(img) if use_luma_only else img
        Cp = work.shape[-1]

        m = int(edge_margin_px)
        seed = _iutils_torch.zeros(
            (B, 1, H, W), device=img.device, dtype=_iutils_torch.float32
        )
        seed[:, :, :m, :] = 1.0
        seed[:, :, -m:, :] = 1.0
        seed[:, :, :, :m] = 1.0
        seed[:, :, :, -m:] = 1.0

        top = work[:, :m, :, :].reshape(B, -1, Cp)
        bottom = work[:, -m:, :, :].reshape(B, -1, Cp)
        left = work[:, :, :m, :].reshape(B, -1, Cp)
        right = work[:, :, -m:, :].reshape(B, -1, Cp)
        samples = _iutils_torch.cat([top, bottom, left, right], dim=1)

        med = samples.median(dim=1).values
        absdev = (samples - med.unsqueeze(1)).abs()
        mad = absdev.median(dim=1).values + 1e-6

        base = _iutils_torch.full_like(med, float(fuzz_percent) / 100.0)
        if str(fuzz_mode) == "adaptive":
            base = base + float(adaptive_k) * mad
        thr_ch = base.clamp(0.0, 1.0)

        border_rgb = med.view(B, 1, 1, Cp)
        thr_ch_4d = thr_ch.view(B, 1, 1, Cp)

        grown = _iutils_region_grow(work, seed, border_rgb, thr_ch_4d, int(max_growth_iter))
        nonborder = (1.0 - grown).clamp(0, 1)
        boxes = _iutils_bbox_from_mask(nonborder, int(pad_px))

        # Crop
        crops = []
        for b, (l, t, wc, hc) in enumerate(boxes):
            crops.append(img[b : b + 1, t : t + hc, l : l + wc, :])
        max_h = max(c.shape[1] for c in crops)
        max_w = max(c.shape[2] for c in crops)
        out = []
        for c in crops:
            pad_h = max_h - c.shape[1]
            pad_w = max_w - c.shape[2]
            out.append(_iutils_F.pad(c, (0, 0, 0, pad_w, 0, pad_h)))
        out_img = _iutils_torch.cat(out, dim=0).clamp(0, 1)

        border_mask = grown.permute(0, 2, 3, 1)
        if (border_mask.shape[1] != out_img.shape[1]) or (border_mask.shape[2] != out_img.shape[2]):
            border_mask = (
                _iutils_F.interpolate(
                    border_mask.permute(0, 3, 1, 2),
                    size=(out_img.shape[1], out_img.shape[2]),
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)
            )

        # Enforce even dimensions (safe)
        H_out, W_out = out_img.shape[1], out_img.shape[2]

        # Width
        if (W_out % 2) == 1:
            if W_out > 1:
                out_img = out_img[:, :, : W_out - 1, :]
                border_mask = border_mask[:, :, : W_out - 1, :]
                W_out = W_out - 1
            else:
                # pad one column by repeating last column
                pad_col = out_img[:, :, -1:, :]
                out_img = _iutils_torch.cat([out_img, pad_col], dim=2)
                pad_col_m = border_mask[:, :, -1:, :]
                border_mask = _iutils_torch.cat([border_mask, pad_col_m], dim=2)
                W_out = 2

        # Height
        if (H_out % 2) == 1:
            if H_out > 1:
                out_img = out_img[:, : H_out - 1, :, :]
                border_mask = border_mask[:, : H_out - 1, :, :]
                H_out = H_out - 1
            else:
                # pad one row by repeating last row
                pad_row = out_img[:, -1:, :, :]
                out_img = _iutils_torch.cat([out_img, pad_row], dim=1)
                pad_row_m = border_mask[:, -1:, :, :]
                border_mask = _iutils_torch.cat([border_mask, pad_row_m], dim=1)
                H_out = 2

        # Reported crop box (single image): clamp to >=2
        if B == 1:
            l, t, wc, hc = boxes[0]
            wc = wc if (wc % 2 == 0) else (wc - 1 if wc > 1 else 2)
            hc = hc if (hc % 2 == 0) else (hc - 1 if hc > 1 else 2)
        else:
            l = t = wc = hc = 0

        return (
            out_img,
            int(l),
            int(t),
            int(wc),
            int(hc),
            border_mask if return_border_mask else _iutils_torch.zeros_like(border_mask),
        )


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
    "AutoCropBorders": AutoCropBorders,
    "CropImageByMargins": CropImageByMargins,
    "CropMaskByMargins": CropMaskByMargins,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropBorders": "Auto-Crop Borders (fuzzy)",
    "CropImageByMargins": "Crop by Margins (Image)",
    "CropMaskByMargins": "Crop by Margins (Mask)",
}
