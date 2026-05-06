import torch

def enforce_image_format(image, force_rgb: bool = False) -> torch.Tensor:
    """
    Standardize an image tensor into [B, H, W, C] float32 clamped to [0.0, 1.0].
    If force_rgb is True, automatically converts 1-channel (grayscale) 
    and 4-channel (RGBA) images to 3-channel RGB by repeating or slicing.
    """
    t = image
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(t.shape)}")
        
    t = t.to(dtype=torch.float32)
    
    if force_rgb:
        if t.shape[-1] == 1:
            t = t.repeat(1, 1, 1, 3)
        elif t.shape[-1] > 3:
            t = t[..., :3]
            
    return t.clamp(0.0, 1.0)
