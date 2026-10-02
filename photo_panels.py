"""Separate a composite photograph into a ragged IMAGE list for an image chooser."""
import json

from .core.panels import DETECTION_MODES, SENSITIVITIES, PanelOptions, split_photo_panels


class PhotoPanelSplitV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_mode": (list(DETECTION_MODES), {"default": "Gutters only",
                    "tooltip": "Gutters only is safest. Collage mode also accepts interrupted straight seams; verify the thumbnails."}),
                "sensitivity": (list(SENSITIVITIES), {"default": "Balanced"}),
                "min_panel_area_percent": ("FLOAT", {"default": 3.0, "min": .1, "max": 40.0, "step": .5,
                    "tooltip": "Minimum automatic panel area as a percentage of the original input. Manual splits bypass this limit."}),
                "max_panels": ("INT", {"default": 16, "min": 2, "max": 64,
                    "tooltip": "Automatic detection stops at this count without discarding unsplit regions."}),
            },
            "optional": {
                "vertical_splits": ("STRING", {"default": "", "tooltip": "Optional override: x positions such as '38.9%' or '357:364' to exclude a gutter. Separate multiple positions with commas. Any override replaces automatic detection."}),
                "horizontal_splits": ("STRING", {"default": "", "tooltip": "Optional override: y positions such as '50%' or '760:768'. Positions refer to original source pixels, not editor resolution."}),
                "column_horizontal_splits": ("STRING", {"default": "", "tooltip": "With vertical_splits, give different y cuts per column separated by |, e.g. '258,520 | 205,581 | 258,520'. Do not also set horizontal_splits."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("panels", "analysis")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split"
    CATEGORY = "PortraitUtils/Photo Edit V2"
    DESCRIPTION = ("Detect separate rectangular photographs inside one composite and extract their original pixels. "
                   "Connect panels to Image Chooser Classic's images input; choose one before AutoCrop/mask generation/Photo Prepare. "
                   "Different panel sizes are kept as a list, never resized or padded into a batch.")

    def split(self, image, detection_mode="Gutters only", sensitivity="Balanced",
              min_panel_area_percent=3.0, max_panels=16, vertical_splits="", horizontal_splits="",
              column_horizontal_splits=""):
        result = split_photo_panels(image, PanelOptions(detection_mode, sensitivity, min_panel_area_percent, max_panels),
                                    vertical_splits, horizontal_splits, column_horizontal_splits)
        return list(result.images), json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)


NODE_CLASS_MAPPINGS = {"PortraitPhotoPanelSplitV2": PhotoPanelSplitV2}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitPhotoPanelSplitV2": "Detect & Split Photo Panels"}
