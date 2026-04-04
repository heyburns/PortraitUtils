import torch
import torch.nn.functional as F

def _ensure_bhwc_rgb(image: torch.Tensor) -> torch.Tensor:
    t = image
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    t = t.to(torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        raise ValueError(f"Expected tensor with 4 dims (B,H,W,C), got {tuple(t.shape)}")
    ch = t.shape[-1]
    if ch == 3:
        return t.clamp(0.0, 1.0)
    if ch == 1:
        return t.repeat(1, 1, 1, 3).clamp(0.0, 1.0)
    return t[..., :3].clamp(0.0, 1.0)

def _rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype)
    return torch.tensordot(rgb, weights, dims=([-1], [0]))

def _scan_edge(lines: torch.Tensor, fuzz_tol: float, edge_unif: float) -> int:
    limit = lines.shape[0] // 2
    trim = 0
    N = 24  # Maximum gap width to forgive over graphical blocks
    
    prev_base = None
    C = lines.shape[2]
    
    while trim < limit:
        current_line = lines[trim]
        base_color = current_line.median(dim=0).values
        
        mad = (current_line - base_color.view(1, C)).abs().mean(dim=-1).median()
        thresh = min(0.15, float(fuzz_tol) + float(mad.item()) * 2.5)
        
        if prev_base is not None:
            color_jump = (base_color - prev_base).abs().mean().item()
            if color_jump < max(0.10, thresh * 2.0):
                break
                
        self_dist = (current_line - base_color.view(1, C)).abs().mean(dim=-1)
        self_uf = (self_dist <= thresh).float().mean()
        
        # Ensures respect for bounding edge parameter immediately rather than hard-locking
        if self_uf < float(edge_unif):
            break
            
        remaining = lines[trim:limit]
        dist = (remaining - base_color.view(1, 1, C)).abs().mean(dim=-1)
        uf = (dist <= thresh).float().mean(dim=1)
        matches = (uf >= float(edge_unif)).float()
        
        L = matches.shape[0]
        if L < N:
            # Revert to linear contiguous scan on margins too thin to gap-jump
            run = int(matches.cummin(dim=0).values.sum().item())
        else:
            is_zero = (matches == 0.0).float()
            kernel = torch.ones(N, device=matches.device)
            # Find all chunks exhibiting an N-pixel solid failure wall
            zero_blocks = F.conv1d(is_zero.view(1, 1, -1), kernel.view(1, 1, N)).view(-1)
            
            # Snap the break point to the very first triggering gap wall instance
            break_points = torch.where(zero_blocks > N - 0.5)[0]
            
            if len(break_points) == 0:
                run = L
            else:
                run = int(break_points[0].item())
                
        if run == 0:
            break
            
        trim += run
        prev_base = base_color
        
    return trim

class IntelligentAutoCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strip_bottom_banner": ("BOOLEAN", {"default": True}),
                "detect_borders": ("BOOLEAN", {"default": True}),
                "fuzz_tolerance": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.01}),
                "edge_uniformity": ("FLOAT", {"default": 0.95, "min": 0.3, "max": 0.99, "step": 0.01}),
                "pad_px": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("image", "trim_left", "trim_top", "trim_right", "trim_bottom", "detected")
    FUNCTION = "run"
    CATEGORY = "Image/Transform"

    def run(self, image, strip_bottom_banner=True, detect_borders=True, fuzz_tolerance=0.04, edge_uniformity=0.75, pad_px=0):
        img = _ensure_bhwc_rgb(image)
        B, H, W, C = img.shape
        
        out_images = []
        l_total = t_total = r_total = b_total = 0
        detected_any = False
        
        for b in range(B):
            work = img[b] # [H,W,C]
            curr_h, curr_w = H, W
            curr_bottom_trim = 0
            
            # 1. Strip Bottom Banner
            if strip_bottom_banner and curr_h > 32:
                gray = _rgb_to_luma(work)
                scan_rows = max(1, int(curr_h * 0.15))
                region = gray[-scan_rows:, :]
                
                dark_thresh = 0.22
                bright_thresh = 0.70
                
                row_mean = region.mean(dim=1)
                bright_frac = (region >= bright_thresh).float().mean(dim=1)
                is_dark_row = row_mean <= dark_thresh
                
                trim_count = 0
                seen_bright = False
                for r in range(scan_rows - 1, -1, -1):
                    if is_dark_row[r]:
                        trim_count += 1
                        if bright_frac[r] >= 0.005:
                            seen_bright = True
                    else:
                        break
                
                if trim_count >= 2 and seen_bright:
                    curr_bottom_trim = trim_count + 2
                    detected_any = True
            
            w_start, w_end = 0, curr_h - curr_bottom_trim
            h_start, h_end = 0, curr_w
            
            # 2. Adaptive Rolling Scan
            if detect_borders and (w_end - w_start) > 16 and (h_end - h_start) > 16:
                roi = work[w_start:w_end, h_start:h_end, :]
                
                # Scan all four borders iteratively and independently
                r_top = _scan_edge(roi, fuzz_tolerance, edge_uniformity)
                r_bot = _scan_edge(roi.flip(0), fuzz_tolerance, edge_uniformity)
                c_lef = _scan_edge(roi.permute(1, 0, 2), fuzz_tolerance, edge_uniformity)
                c_rig = _scan_edge(roi.permute(1, 0, 2).flip(0), fuzz_tolerance, edge_uniformity)
                
                if r_top > 0 or r_bot > 0 or c_lef > 0 or c_rig > 0:
                    r_top = max(0, r_top - pad_px)
                    r_bot = max(0, r_bot - pad_px)
                    c_lef = max(0, c_lef - pad_px)
                    c_rig = max(0, c_rig - pad_px)
                    
                    new_w_start = w_start + r_top
                    new_w_end = w_end - r_bot
                    new_h_start = h_start + c_lef
                    new_h_end = h_end - c_rig
                    
                    if new_w_start < new_w_end and new_h_start < new_h_end:
                        w_start, w_end = new_w_start, new_w_end
                        h_start, h_end = new_h_start, new_h_end
                        detected_any = True
            
            out_images.append(work[w_start:w_end, h_start:h_end, :])
            if b == 0:
                l_total = h_start
                t_total = w_start
                r_total = W - h_end
                b_total = H - w_end

        if B > 1:
            max_h = max(c.shape[0] for c in out_images)
            max_w = max(c.shape[1] for c in out_images)
            padded = []
            for c in out_images:
                pad_h = max_h - c.shape[0]
                pad_w = max_w - c.shape[1]
                padded.append(F.pad(c, (0, 0, 0, pad_w, 0, pad_h)).unsqueeze(0))
            final_out = torch.cat(padded, dim=0)
        else:
            final_out = out_images[0].unsqueeze(0)
            
        return (final_out.clamp(0.0, 1.0), int(l_total), int(t_total), int(r_total), int(b_total), detected_any)

NODE_CLASS_MAPPINGS = {
    "IntelligentAutoCrop": IntelligentAutoCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntelligentAutoCrop": "Intelligent AutoCrop (GPU)",
}
