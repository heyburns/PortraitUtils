"""Configured compositing with opacity applied AFTER outward feathering."""

from copy import deepcopy


from .workflow_config_v2 import StitchByMaskV2

from .core.composite import (_resize, _feather, _channels, CompositeEngine)


class PortraitStitchPreview(CompositeEngine):
    @classmethod
    def INPUT_TYPES(cls):
        return deepcopy(StitchByMaskV2.INPUT_TYPES())

    RETURN_TYPES = StitchByMaskV2.RETURN_TYPES
    RETURN_NAMES = StitchByMaskV2.RETURN_NAMES
    FUNCTION = "blend"
    CATEGORY = "PortraitUtils/Composite V2"


NODE_CLASS_MAPPINGS = {"PortraitStitchPreview": PortraitStitchPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitStitchPreview": "Stitch by Mask"}
