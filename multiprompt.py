class MultiPromptNode:
    """Simple shim that exposes four multiline prompt sockets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_mask": (
                    "STRING",
                    {"default": "", "multiline": True, "placeholder": "subject mask"},
                ),
                "inpaint_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "placeholder": "inpaint prompt"},
                ),
                "outpaint_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "placeholder": "outpaint prompt"},
                ),
                "stitch_mask": (
                    "STRING",
                    {"default": "", "multiline": True, "placeholder": "stitch mask"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("subject_mask", "inpaint_prompt", "outpaint_prompt", "stitch_mask")
    FUNCTION = "apply"
    CATEGORY = "config"

    def apply(self, subject_mask, inpaint_prompt, outpaint_prompt, stitch_mask):
        return (
            str(subject_mask),
            str(inpaint_prompt),
            str(outpaint_prompt),
            str(stitch_mask),
        )


NODE_CLASS_MAPPINGS = {
    "MultiPromptNode": MultiPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiPromptNode": "Multi-Prompt",
}
