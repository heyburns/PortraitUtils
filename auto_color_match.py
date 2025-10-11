from __future__ import annotations
import torch
import torch.nn.functional as F

# ---------------------------
# sRGB <-> Linear helpers
# ---------------------------
def srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)).pow(2.4))

def linear_to_srgb(x):
    a = 0.055
    return torch.where(x <= 0.0031308, x * 12.92, (1 + a) * x.pow(1/2.4) - a)

# ---------------------------
# RGB <-> XYZ <-> Lab (D65)
# ---------------------------
# Matrices for D65 sRGB
_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
_M_XYZ2RGB = torch.tensor([[ 3.2404542, -1.5371385, -0.4985314],
                           [-0.9692660,  1.8760108,  0.0415560],
                           [ 0.0556434, -0.2040259,  1.0572252]], dtype=torch.float32)
# D65 white point
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883

def rgb_to_lab(rgb):  # [B,H,W,3] in 0..1 sRGB
    B,H,W,C = rgb.shape
    M = _M_RGB2XYZ.to(rgb.device, rgb.dtype)
    # sRGB -> linear -> XYZ
    lin = srgb_to_linear(rgb)
    xyz = torch.einsum('bhwc,cd->bhwd', lin, M)
    # Normalize by white point
    x = xyz[...,0] / _Xn
    y = xyz[...,1] / _Yn
    z = xyz[...,2] / _Zn
    eps = 216/24389
    kappa = 24389/27

    def f(t):
        return torch.where(t > eps, t.pow(1/3), (kappa * t + 16) / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.stack([L, a, b], dim=-1)  # [B,H,W,3]

def lab_to_rgb(lab):  # [B,H,W,3]
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    fy = (L + 16)/116
    fx = fy + a/500
    fz = fy - b/200

    eps = 216/24389
    kappa = 24389/27

    def finv(ft):
        t3 = ft**3
        return torch.where(t3 > eps, t3, (116*ft - 16)/kappa)

    xr, yr, zr = finv(fx), finv(fy), finv(fz)
    X = xr * _Xn
    Y = yr * _Yn
    Z = zr * _Zn
    xyz = torch.stack([X,Y,Z], dim=-1)
    M = _M_XYZ2RGB.to(lab.device, lab.dtype)
    lin = torch.einsum('bhwc,cd->bhwd', xyz, M)
    srgb = linear_to_srgb(lin).clamp(0,1)
    return srgb

# ---------------------------
# Utility
# ---------------------------
def to_bhwc(x):
    if x.dim()==3: x = x.unsqueeze(0)
    assert x.dim()==4 and x.shape[-1] in (3,4), f"Expected [B,H,W,3/4], got {tuple(x.shape)}"
    if x.shape[-1]==4: x = x[...,:3]
    return x.float().clamp(0,1)

def resize_bhwc(x, h, w, mode='bilinear'):
    return F.interpolate(x.permute(0,3,1,2), size=(h,w), mode=mode, align_corners=False).permute(0,2,3,1)

def luminance(img):
    # Rec.709 luma
    return 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]

def per_image_mean_std(x):
    # x: [B,H,W,3]
    B = x.shape[0]
    flat = x.view(B, -1, 3)
    mean = flat.mean(dim=1, keepdim=True)
    std  = flat.std (dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return mean, std

# ---------------------------
# Methods
# ---------------------------
def wb_grayworld(img):
    # scale channels so mean becomes gray
    B = img.shape[0]
    mean = img.view(B,-1,3).mean(dim=1)  # [B,3]
    gray = mean.mean(dim=1, keepdim=True) # [B,1]
    gains = (gray / mean).view(B,1,1,3)
    out = (img * gains).clamp(0,1)
    return out

def wb_highlight(img, percentile=95.0):
    # use brightest-percent luminance pixels as "white patch"
    B,H,W,_ = img.shape
    Y = luminance(img).view(B, -1)
    thr = torch.quantile(Y, percentile/100.0, dim=1, keepdim=True)  # [B,1]
    mask = (Y >= thr).view(B,H,W,1)
    # avoid empty mask
    eps = 1e-6
    sel = img * mask
    count = mask.sum(dim=(1,2,3), keepdim=True).clamp_min(1.0)
    mean_sel = sel.sum(dim=(1,2,3), keepdim=True) / count  # [B,1,1,3]
    target_white = torch.ones_like(mean_sel) * 0.95  # bring selected whites near 95% to avoid clipping
    gains = (target_white / mean_sel).clamp(0.5, 2.0)
    return (img * gains).clamp(0,1)

def reinhard_match(src, ref, l_only=False):
    # Convert to Lab, match mean/std
    src_lab = rgb_to_lab(src)
    ref_lab = rgb_to_lab(ref)
    src_mean, src_std = per_image_mean_std(src_lab)
    ref_mean, ref_std = per_image_mean_std(ref_lab)

    if l_only:
        # match L only
        L = (src_lab[...,0:1] - src_mean[...,0:1]) / src_std[...,0:1] * ref_std[...,0:1] + ref_mean[...,0:1]
        out_lab = torch.cat([L, src_lab[...,1:],], dim=-1)
    else:
        out_lab = (src_lab - src_mean) / src_std * ref_std + ref_mean

    out = lab_to_rgb(out_lab).clamp(0,1)
    return out

# ---------------------------
# Comfy node
# ---------------------------
class AutoWBColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": ([
                    "wb_grayworld",
                    "wb_highlight",
                    "reinhard_lab",
                    "lab_l_only",
                    "wb_highlight+reinhard",
                ], {"default": "wb_highlight+reinhard"}),
                "percentile": ("FLOAT", {"default": 95.0, "min": 80.0, "max": 99.9, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip_gamut": ("BOOLEAN", {"default": True}),
                "force_size": ("BOOLEAN", {"default": False}),
                "target_width": ("INT", {"default": 1440, "min": 16, "max": 8192, "step": 1}),
                "target_height": ("INT", {"default": 1080, "min": 16, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Image/Color"

    def run(self, image, reference, method="wb_highlight+reinhard",
            percentile=95.0, strength=1.0, clip_gamut=True,
            force_size=False, target_width=1440, target_height=1080):

        src = to_bhwc(image)
        ref = to_bhwc(reference)

        if force_size:
            th, tw = target_height, target_width
            src_small = resize_bhwc(src, th, tw)
            ref_small = resize_bhwc(ref, th, tw)
        else:
            src_small, ref_small = src, ref

        # 1) white balance / base correction
        if method in ("wb_grayworld",):
            base = wb_grayworld(src_small)
        elif method in ("wb_highlight", "wb_highlight+reinhard"):
            base = wb_highlight(src_small, percentile=float(percentile))
        elif method in ("reinhard_lab","lab_l_only"):
            base = src_small
        else:
            base = src_small

        # 2) color match
        if method == "reinhard_lab":
            matched = reinhard_match(base, ref_small, l_only=False)
        elif method == "lab_l_only":
            matched = reinhard_match(base, ref_small, l_only=True)
        elif method == "wb_highlight+reinhard":
            matched = reinhard_match(base, ref_small, l_only=False)
        else:
            matched = base

        # If we resized for stats, reapply the transform back to original resolution
        if force_size and (src.shape[1]!=matched.shape[1] or src.shape[2]!=matched.shape[2]):
            matched = resize_bhwc(matched, src.shape[1], src.shape[2])

        # 3) blend strength
        out = (1 - strength) * src + strength * matched
        if clip_gamut:
            out = out.clamp(0,1)

        return (out,)

NODE_CLASS_MAPPINGS = {
    "AutoWBColorMatch": AutoWBColorMatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoWBColorMatch": "Auto White-Balance + Color Match",
}
