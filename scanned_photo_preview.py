"""Conservative scan deskew/crop without an 8-bit processing round trip."""

from copy import deepcopy


from .auto_straighten import ProcessScannedPhoto

from .core.scans import (
    _LOG,
    _ANALYSIS_SIDE,
    _rgb,
    _sample,
    _print_rectangle,
    _rotation,
    _box,
    _inner_frame,
    ScannedPhotoEngine,
)


class PortraitScannedPhotoPreview(ScannedPhotoEngine):
    @classmethod
    def INPUT_TYPES(cls):
        return deepcopy(ProcessScannedPhoto.INPUT_TYPES())

    RETURN_TYPES = ProcessScannedPhoto.RETURN_TYPES
    FUNCTION = "process"
    CATEGORY = "PortraitUtils/Transform"


NODE_CLASS_MAPPINGS = {"PortraitScannedPhotoPreview": PortraitScannedPhotoPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitScannedPhotoPreview": "Process Scanned Photo"}
