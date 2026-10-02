"""Atomic photo exports; never commit a partially encoded destination image."""

from copy import deepcopy
import os
import shutil

import folder_paths
import torch
from comfy.cli_args import args

from .simple_image_saver import SimpleImageSaver, JPEG_COMMENT_MAX_BYTES

from .core.export import (
    _LOG,
    _component,
    _directory,
    _metadata as _engine_metadata,
    _png16,
    _encode,
    _commit,
    PhotoExportEngine,
    ExportEnvironment,
)


def _metadata(include, prompt, extra, file_format="PNG"):
    return _engine_metadata(include, prompt, extra, file_format, args.disable_metadata)


class PortraitImageSaverPreview:
    @classmethod
    def INPUT_TYPES(cls):
        schema = deepcopy(SimpleImageSaver.INPUT_TYPES())
        schema["optional"] = {
            "png_bit_depth": (["8-bit", "16-bit"], {"default": "8-bit",
                "tooltip": "16-bit PNG preserves more of the float image's precision. JPEG remains 8-bit."}),
            "jpeg_alpha_background": (["White", "Black"], {"default": "White",
                "tooltip": "Composite RGBA on this background for JPEG; PNG retains alpha."}),
        }
        return schema

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "PortraitUtils/IO"

    @torch.no_grad()
    def save(self, images, output_path="", filename="ComfyUI", suffix="", file_format="PNG",
             jpeg_quality=95, include_metadata=True, unique_filenames=True, prompt=None,
             extra_pnginfo=None, png_bit_depth="8-bit", jpeg_alpha_background="White"):
        environment = ExportEnvironment(folder_paths.get_output_directory(),
                                        folder_paths.get_temp_directory(), args.disable_metadata)
        return PhotoExportEngine(environment, _encode).save(
            images, output_path, filename, suffix, file_format, jpeg_quality,
            include_metadata, unique_filenames, prompt, extra_pnginfo,
            png_bit_depth, jpeg_alpha_background)


NODE_CLASS_MAPPINGS = {"PortraitImageSaverPreview": PortraitImageSaverPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitImageSaverPreview": "Simple Image Saver"}
