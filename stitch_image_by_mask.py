import torch
import torch.nn.functional as F

def _to_bhwc(img: torch.Tensor) -> torch.Tensor:
    # Accept [H,W,C] or [B,H,W,C]; ensure float32 0..1
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4 or img.size(-1) not in (1, 3, 4):
        raise ValueError(f"IMAGE must be [B,H,W,C] (C=1/3/4), got {tuple(img.shape)}")
    return img.to(torch.float32).clamp(0.0, 1.0)


def _resize_bhwc(x: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
    t = x.permute(0, 3, 1, 2)
    t = F.interpolate(
        t,
        size=(h, w),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )
    return t.permute(0, 2, 3, 1)


def _mask_to_bhw1(m: torch.Tensor) -> torch.Tensor:
    """
    MASK arrives as [B,H,W] or [B,H,W,1] or [H,W] or [H,W,1] (float 0..1).
    Return [B,H,W,1] float32 0..1.
    """
    if m.dim() == 2:  # [H,W]
        m = m.unsqueeze(0).unsqueeze(-1)
    elif m.dim() == 3:
        # Could be [B,H,W] or [H,W,1]
        if m.shape[-1] == 1:  # [H,W,1]
            m = m.unsqueeze(0)
        else:  # [B,H,W]
            m = m.unsqueeze(-1)
    elif m.dim() == 4:
        if m.shape[-1] != 1:
            raise ValueError("MASK 4D must be [B,H,W,1]")
    else:
        raise ValueError(f"Unexpected MASK shape {tuple(m.shape)}")
    return m.to(torch.float32).clamp(0.0, 1.0)


def _make_disk_kernel(radius: int, device) -> torch.Tensor:
    if radius <= 0:
        k = torch.ones((1, 1, 1, 1), device=device)
        return k
    d = 2 * radius + 1
    yy, xx = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(d, device=device),
        indexing="ij",
    )
    cy = cx = radius
    disk = (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius).float()
    if disk.sum() <= 0:
        disk = torch.ones_like(disk)
    disk = disk / disk.sum()
    return disk.view(1, 1, d, d)


def _gaussian_blur_chw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    # Separable approx
    rad = max(1, int(torch.ceil(torch.tensor(3.0 * max(0.1, sigma))).item()))
    xs = torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = k / k.sum().clamp_min(1e-6)
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    C = x.shape[1]
    y = F.conv2d(x, kx.expand(C, 1, 1, kx.shape[-1]), padding=(0, rad), groups=C)
    y = F.conv2d(y, ky.expand(C, 1, ky.shape[-2], 1), padding=(rad, 0), groups=C)
    return y


class StitchByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mask": ("MASK", {"default": None}),  # optional when bypassing
                "invert_mask": ("BOOLEAN", {"default": False}),
                "bypass_mask": ("BOOLEAN", {"default": False}),
                "opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "feather_radius": (
                    "INT",
                    {"default": 5, "min": 0, "max": 100, "step": 1},
                ),
                "force_size": ("BOOLEAN", {"default": False}),
                "target_width": (
                    "INT",
                    {"default": 1344, "min": 16, "max": 8192, "step": 1},
                ),
                "target_height": (
                    "INT",
                    {"default": 768, "min": 16, "max": 8192, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("stitched", "processed_mask")
    FUNCTION = "blend"
    CATEGORY = "Image/Composite"

    def blend(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        mask: torch.Tensor = None,
        invert_mask: bool = False,
        bypass_mask: bool = False,
        opacity: float = 1.0,
        feather_radius: int = 5,
        force_size: bool = True,
        target_width: int = 1440,
        target_height: int = 1080,
    ):

        a = _to_bhwc(image_a)
        b = _to_bhwc(image_b)
        if mask is not None:
            m = _mask_to_bhw1(mask)  # [B,H,W,1]
        else:
            m = None

        # Align sizes (do before feather so radius is in target pixels when forcing size)
        if force_size:
            Ht, Wt = target_height, target_width
            a = _resize_bhwc(a, Ht, Wt, mode="bilinear")
            b = _resize_bhwc(b, Ht, Wt, mode="bilinear")
            if m is not None:
                # nearest for mask to keep edges before feathering
                m = _resize_bhwc(m, Ht, Wt, mode="nearest")
        else:
            if not (a.shape[1:3] == b.shape[1:3]):
                raise ValueError(
                    f"Size mismatch: A {a.shape[1:3]}  B {b.shape[1:3]}"
                )
            if m is not None and m.shape[1:3] != a.shape[1:3]:
                raise ValueError(
                    f"Mask size mismatch: expected {a.shape[1:3]}, got {m.shape[1:3]}"
                )

        if bypass_mask:
            mask_scalar = torch.full(
                (a.shape[0], a.shape[1], a.shape[2], 1),
                float(opacity),
                device=a.device,
                dtype=a.dtype,
            )
            blend_mask = (
                mask_scalar
                if a.size(-1) == 1
                else mask_scalar.expand(-1, -1, -1, a.size(-1))
            )
            out = blend_mask * b + (1.0 - blend_mask) * a
            return (out.clamp(0.0, 1.0), mask_scalar.clamp(0.0, 1.0))

        if m is None:
            raise ValueError("Mask input required unless 'bypass_mask' is enabled.")

        # Invert & apply opacity
        if invert_mask:
            m = 1.0 - m
        if opacity < 1.0:
            m = m * float(opacity)

        # ----- Edge-safe feathering -----
        # Make a binary core from original mask (protect original coverage)
        core = (m >= 0.5).float()  # [B,H,W,1]
        # Dilate (expand) by feather_radius using a disk kernel
        if feather_radius > 0:
            k = _make_disk_kernel(feather_radius, device=m.device)  # [1,1,d,d]
            core_chw = core.permute(0, 3, 1, 2)  # [B,1,H,W]
            # convolution-based dilation approx: if any pixel inside disk -> >0 after conv
            dil = F.conv2d(core_chw, k, padding=feather_radius)  # normalized sum
            expanded = (dil > 0).float()  # binary expanded
            # Feather by Gaussian blur with sigma ~ radius/2
            sigma = max(0.5, feather_radius / 2.0)
            feathered = _gaussian_blur_chw(expanded, sigma=sigma).clamp(0.0, 1.0)
            m_feather = feathered.permute(0, 2, 3, 1)
            # Guarantee we never unmask what was masked: take max with original soft mask
            m_safe = torch.maximum(m_feather, m)
        else:
            m_safe = m

        # Broadcast mask to match image channels
        if a.size(-1) != 1 and m_safe.size(-1) == 1:
            m_safe = m_safe.expand(-1, -1, -1, a.size(-1))

        out = m_safe * b + (1.0 - m_safe) * a
        return (out.clamp(0.0, 1.0), m_safe.clamp(0.0, 1.0))


NODE_CLASS_MAPPINGS = {
    "StitchByMask": StitchByMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StitchByMask": "Stitch Two Images by Mask",
}
