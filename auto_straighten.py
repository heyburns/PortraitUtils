import torch
import numpy as np
import cv2

class ProcessScannedPhoto:
    """Automatically straightens and crops scanned photos in a single pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "straighten": ("BOOLEAN", {"default": True}),
                "crop_mode": (["Inner Photo Frame", "Scanner Bed Only", "None"], {"default": "Inner Photo Frame"}),
                "padding": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "PortraitUtils/Transform"

    def process(self, image, straighten, crop_mode, padding, threshold):
        out_images = []
        for img_tensor in image:
            # Convert to numpy
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            if img_np.shape[-1] >= 3:
                gray = cv2.cvtColor(img_np[..., :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np.squeeze()

            thresh_val = int(threshold * 255)

            # --- STEP 1: STRAIGHTEN ---
            if straighten:
                # Use Hybrid Edge Detection to find outer bounds
                gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(gray_blur, 30, 100)
                edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

                _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                combined_mask = cv2.bitwise_or(edges, thresh_img)

                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_points = [c for c in contours if cv2.contourArea(c) > 100 or len(c) > 50]

                if valid_points:
                    all_points = np.vstack(valid_points)
                    hull = cv2.convexHull(all_points)
                    rect = cv2.minAreaRect(hull)
                    center, size, angle = rect
                    box = cv2.boxPoints(rect)
                    
                    edges_info = []
                    for i in range(4):
                        p1 = box[i]
                        p2 = box[(i+1)%4]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        length = np.hypot(dx, dy)
                        angle_deg = np.degrees(np.arctan2(dy, dx))
                        edges_info.append((length, angle_deg))
                        
                    longest_edge = max(edges_info, key=lambda x: x[0])
                    longest_angle = longest_edge[1]

                    closest_90 = round(longest_angle / 90.0) * 90.0
                    angle_to_rotate = longest_angle - closest_90

                    h, w = img_np.shape[:2]
                    M = cv2.getRotationMatrix2D(center, angle_to_rotate, 1.0)

                    abs_cos = abs(M[0, 0])
                    abs_sin = abs(M[0, 1])
                    bound_w = int(h * abs_sin + w * abs_cos)
                    bound_h = int(h * abs_cos + w * abs_sin)
                    M[0, 2] += bound_w / 2 - center[0]
                    M[1, 2] += bound_h / 2 - center[1]

                    border_val = (255, 255, 255, 255) if img_np.shape[-1] == 4 else ((255, 255, 255) if img_np.shape[-1] == 3 else 255)
                    img_np = cv2.warpAffine(img_np, M, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
                    
                    # Update gray image for crop step
                    if img_np.shape[-1] >= 3:
                        gray = cv2.cvtColor(img_np[..., :3], cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_np.squeeze()

            # --- STEP 2: CROP ---
            if crop_mode != "None":
                h_img, w_img = img_np.shape[:2]
                
                if crop_mode == "Scanner Bed Only":
                    # Use Hybrid logic on rotated image
                    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(gray_blur, 30, 100)
                    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
                    
                    _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                    combined = cv2.bitwise_or(edges, thresh_img)
                    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    valid_points = [c for c in contours if cv2.contourArea(c) > 100 or len(c) > 50]
                    if valid_points:
                        x, y, cw, ch = cv2.boundingRect(np.vstack(valid_points))
                        
                        x_min = max(0, x - padding)
                        y_min = max(0, y - padding)
                        x_max = min(w_img, x + cw + padding)
                        y_max = min(h_img, y + ch + padding)
                        img_np = img_np[y_min:y_max, x_min:x_max]

                elif crop_mode == "Inner Photo Frame":
                    # Use Morphological Contour logic to bypass borders
                    _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                    kernel = np.ones((15, 15), np.uint8)
                    closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, cw, ch = cv2.boundingRect(largest_contour)

                        x_min = max(0, x - padding)
                        y_min = max(0, y - padding)
                        x_max = min(w_img, x + cw + padding)
                        y_max = min(h_img, y + ch + padding)
                        img_np = img_np[y_min:y_max, x_min:x_max]

            # Convert back to tensor
            out_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
            out_images.append(out_tensor)

        # Batching logic
        if len(out_images) == 1:
            return (out_images[0],)
        
        max_h = max(img.shape[1] for img in out_images)
        max_w = max(img.shape[2] for img in out_images)
        padded_images = []
        for img in out_images:
            h_i, w_i = img.shape[1:3]
            pad_h = max_h - h_i
            pad_w = max_w - w_i
            if pad_h > 0 or pad_w > 0:
                c = img.shape[-1]
                padded = torch.ones((1, max_h, max_w, c), dtype=img.dtype, device=img.device)
                padded[0, :h_i, :w_i, :] = img[0]
                padded_images.append(padded)
            else:
                padded_images.append(img)
                
        return (torch.cat(padded_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "ProcessScannedPhoto": ProcessScannedPhoto
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProcessScannedPhoto": "Process Scanned Photo"
}
