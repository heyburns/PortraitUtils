"""
DEPRECATED: MultiPromptNode is superseded by UniversalProjectConfig.

All four prompt fields (subject_mask, inpaint_prompt, outpaint_prompt, stitch_mask)
are now declared in config_schema.json and output directly from UniversalProjectConfig.

This module is kept in place so that existing saved workflows that reference
"MultiPromptNode" continue to load without hard errors. Migrate your workflows
to UniversalProjectConfig and this node can be removed in a future cleanup.
"""

import logging

_log = logging.getLogger(__name__)
_warned = False  # emit the deprecation notice at most once per session


class MultiPromptNode:
    """[DEPRECATED] Use UniversalProjectConfig instead."""

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
        global _warned
        if not _warned:
            _log.warning(
                "[PortraitUtils] MultiPromptNode is DEPRECATED. "
                "Migrate to UniversalProjectConfig — all prompt fields are now in config_schema.json."
            )
            _warned = True
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
    "MultiPromptNode": "Multi-Prompt [DEPRECATED]",
}
