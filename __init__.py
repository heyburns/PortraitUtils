"""Register current PortraitUtils nodes without retired legacy interfaces.

Processing implementations used by configured nodes remain internal; their
old node IDs are deliberately not exported to ComfyUI.
"""

from . import (
    auto_crop_preview,
    comparison_gate,
    crop_margins_preview,
    filename_append_suffix,
    paired_loader_preview,
    photo_loader_preview,
    image_saver_preview,
    scanned_photo_preview,
    stitch_preview,
    smart_photo_prepare,
    white_balance_preview,
    workflow_config_v2,
    preparation_nodes_v2,
    photo_panels,
)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _module in (
    auto_crop_preview,
    comparison_gate,
    crop_margins_preview,
    filename_append_suffix,
    paired_loader_preview,
    photo_loader_preview,
    image_saver_preview,
    scanned_photo_preview,
    stitch_preview,
    smart_photo_prepare,
    white_balance_preview,
    workflow_config_v2,
    preparation_nodes_v2,
    photo_panels,
):
    _duplicates = NODE_CLASS_MAPPINGS.keys() & _module.NODE_CLASS_MAPPINGS.keys()
    if _duplicates:
        raise RuntimeError(f"Duplicate PortraitUtils node IDs: {sorted(_duplicates)}")
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
