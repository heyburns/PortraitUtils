import torch


class OutpaintConfigNode:
    """
    Control panel for outpainting preferences (no image).
    Outputs raw settings for Set/Get.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Percent", "Pixels"], {"default": "Percent"}),
                "gravity": (
                    [
                        "center",
                        "left",
                        "right",
                        "top",
                        "bottom",
                        "top left",
                        "top right",
                        "bottom right",
                        "bottom left",
                    ],
                    {"default": "center"},
                ),
                "horizontal_percent": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 10000.0, "step": 0.1},
                ),
                "vertical_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 10000.0, "step": 0.1},
                ),
                "left_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "right_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "top_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "bottom_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "mode",
        "gravity",
        "horizontal_percent",
        "vertical_percent",
        "left_px",
        "right_px",
        "top_px",
        "bottom_px",
    )
    FUNCTION = "apply"
    CATEGORY = "config"

    def apply(
        self,
        mode,
        gravity,
        horizontal_percent,
        vertical_percent,
        left_px,
        right_px,
        top_px,
        bottom_px,
    ):

        horizontal_percent = max(0.0, float(horizontal_percent))
        vertical_percent = max(0.0, float(vertical_percent))
        left_px = max(0, int(left_px))
        right_px = max(0, int(right_px))
        top_px = max(0, int(top_px))
        bottom_px = max(0, int(bottom_px))

        return (
            mode,
            gravity,
            horizontal_percent,
            vertical_percent,
            left_px,
            right_px,
            top_px,
            bottom_px,
        )


class OutpaintPaddingComputeNode:
    """
    Compute absolute pixel paddings (left, top, right, bottom) for outpainting.
    - Pixels mode: gravity ignored; enforce even final size.
    - Percent mode: compute by % + gravity (including corners); enforce even final size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": ("STRING", {"default": "Percent"}),
                "gravity": ("STRING", {"default": "center"}),
                "horizontal_percent": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 10000.0, "step": 0.1},
                ),
                "vertical_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 10000.0, "step": 0.1},
                ),
                "left_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "right_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "top_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "bottom_px": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("left", "top", "right", "bottom")
    FUNCTION = "compute"
    CATEGORY = "image/transform"

    def _get_wh(self, image_tensor):
        if not torch.is_tensor(image_tensor):
            image_tensor = torch.tensor(image_tensor)
        if image_tensor.dim() != 4:
            raise ValueError("Expected IMAGE tensor of shape [B,H,W,C].")
        _, H, W, _ = image_tensor.shape
        return int(W), int(H)

    def _enforce_even_final_size(self, W, H, left, top, right, bottom):
        if (W + left + right) % 2 != 0:
            right += 1
        if (H + top + bottom) % 2 != 0:
            bottom += 1
        return left, top, right, bottom

    def compute(
        self,
        image,
        mode,
        gravity,
        horizontal_percent,
        vertical_percent,
        left_px,
        right_px,
        top_px,
        bottom_px,
    ):

        W, H = self._get_wh(image)

        g = str(gravity).lower().strip()
        valid = {
            "center",
            "left",
            "right",
            "top",
            "bottom",
            "top left",
            "top right",
            "bottom right",
            "bottom left",
        }
        if g not in valid:
            g = "center"

        hp = max(0.0, float(horizontal_percent))
        vp = max(0.0, float(vertical_percent))
        lp = max(0, int(left_px))
        rp = max(0, int(right_px))
        tp = max(0, int(top_px))
        bp = max(0, int(bottom_px))

        if mode == "Pixels":
            left, top, right, bottom = lp, tp, rp, bp
            left, top, right, bottom = self._enforce_even_final_size(
                W, H, left, top, right, bottom
            )
            return (left, top, right, bottom)

        if hp == 0.0:
            left = 0
            right = 0
        else:
            horz_total = int(round(W * hp / 100.0))
            if g in ("left", "top left", "bottom left"):
                left, right = horz_total, 0
            elif g in ("right", "top right", "bottom right"):
                right, left = horz_total, 0
            else:
                left = horz_total // 2
                right = horz_total - left

        if vp == 0.0:
            top = 0
            bottom = 0
        else:
            vert_total = int(round(H * vp / 100.0))
            if g in ("top", "top left", "top right"):
                top, bottom = vert_total, 0
            elif g in ("bottom", "bottom left", "bottom right"):
                bottom, top = vert_total, 0
            else:
                top = vert_total // 2
                bottom = vert_total - top

        left, top, right, bottom = self._enforce_even_final_size(
            W, H, left, top, right, bottom
        )
        return (
            max(0, int(left)),
            max(0, int(top)),
            max(0, int(right)),
            max(0, int(bottom)),
        )


NODE_CLASS_MAPPINGS = {
    "OutpaintConfigNode": OutpaintConfigNode,
    "OutpaintPaddingComputeNode": OutpaintPaddingComputeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OutpaintConfigNode": "Outpaint Config",
    "OutpaintPaddingComputeNode": "Outpaint Padding Compute",
}
