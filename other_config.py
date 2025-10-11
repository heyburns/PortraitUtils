class OtherConfigNode:
    """
    Outputs:
      - initial_upscale (megapixels)
      - final_upscale (megapixels)
      - tight_crop (BOOLEAN, default False)
      - blend_factor (FLOAT 0-1)
      - input_dir (STRING)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_upscale_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "label": "initial_upscale (megapixels)",
                    },
                ),
                "final_upscale_megapixels": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "label": "final_upscale (megapixels)",
                    },
                ),
                "tight_crop": ("BOOLEAN", {"default": False}),
                "blend_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "stitch_bypass_mask": ("BOOLEAN", {"default": False}),
                "input_dir": (
                    "STRING",
                    {"default": "", "multiline": False, "placeholder": "Input Directory"},
                ),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "BOOLEAN",
        "FLOAT",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "initial_upscale (megapixels)",
        "final_upscale (megapixels)",
        "tight_crop",
        "blend_factor",
        "stitch_bypass_mask",
        "input_dir",
    )
    FUNCTION = "apply"
    CATEGORY = "config"

    def apply(
        self,
        initial_upscale_megapixels,
        final_upscale_megapixels,
        tight_crop,
        blend_factor,
        stitch_bypass_mask,
        input_dir,
    ):
        initial_upscale_megapixels = max(0.1, float(initial_upscale_megapixels))
        final_upscale_megapixels = max(0.1, float(final_upscale_megapixels))
        blend_factor = min(1.0, max(0.0, float(blend_factor)))
        return (
            initial_upscale_megapixels,
            final_upscale_megapixels,
            bool(tight_crop),
            blend_factor,
             bool(stitch_bypass_mask),
            str(input_dir),
        )


NODE_CLASS_MAPPINGS = {
    "OtherConfigNode": OtherConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OtherConfigNode": "Other Config",
}
