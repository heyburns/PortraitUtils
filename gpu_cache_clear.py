import gc as _iutils_gc
import torch as _iutils_torch


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


class GpuSyncCacheClear:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "image/utils"

    def apply(self, image):
        try:
            if _iutils_torch.cuda.is_available():
                _iutils_torch.cuda.synchronize()
        except Exception:
            pass
        _iutils_gc.collect()
        try:
            if _iutils_torch.cuda.is_available():
                _iutils_torch.cuda.empty_cache()
        except Exception:
            pass
        return (image,)


class SeedVR2Prep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ensure_even_dims": ("BOOLEAN", {"default": True}),
                "to_fp16_on_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "image/utils"

    def apply(self, image, ensure_even_dims=True, to_fp16_on_gpu=True):
        x = _iutils_ensure_bhwc_rgb(image)
        if ensure_even_dims:
            H, W = x.shape[1], x.shape[2]
            # Width
            if (W % 2) == 1:
                if W > 1:
                    x = x[:, :, : W - 1, :]
                else:
                    # pad a column by repeating the last column
                    x = _iutils_torch.cat([x, x[:, :, -1:, :]], dim=2)
            # Height
            H, W = x.shape[1], x.shape[2]
            if (H % 2) == 1:
                if H > 1:
                    x = x[:, : H - 1, :, :]
                else:
                    # pad a row by repeating last row
                    x = _iutils_torch.cat([x, x[:, -1:, :, :]], dim=1)

            if to_fp16_on_gpu and _iutils_torch.cuda.is_available():
                if x.device.type != "cuda":
                    x = x.to("cuda", non_blocking=True)
                x = x.to(dtype=_iutils_torch.float16)
        return (x,)


NODE_CLASS_MAPPINGS = {
    "GpuSyncCacheClear": GpuSyncCacheClear,
    "SeedVR2Prep": SeedVR2Prep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GpuSyncCacheClear": "GPU Sync & Cache Clear",
    "SeedVR2Prep": "SeedVR2 Prep (sanitize)",
}
