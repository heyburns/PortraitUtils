"""Independent 2.0 loader preview; the existing loader is deliberately unchanged."""

from __future__ import annotations

import glob
import os
import re
import threading
import weakref

import folder_paths

from .core.contracts import SOURCE_INFO_TYPE, SourceInfoV2, validate_config
from .core.state import TransactionalCursor
from .core.validation import boolean

from .core.photo_io import (
    EXTENSIONS,
    COLOR_POLICIES,
    ALPHA_BACKGROUNDS,
    _natural_key,
    _fingerprint,
    _bit_depth,
    _orient_array,
    _tiff16,
    _decode,
)


EMPTY_SELECTION = "(no images uploaded)"
# ComfyUI caches node objects by ID. Weak references let IS_CHANGED inspect that
# object's cursor without retaining dead nodes or sharing cursors by folder.
_LIVE_NODES = weakref.WeakValueDictionary()
_LIVE_LOCK = threading.RLock()


def _text(value, name):
    if not isinstance(value, str):
        raise ValueError(f"Photo Loader: '{name}' must be text, not {type(value).__name__}.")
    return value.strip()


def _batch_files(input_dir, pattern):
    directory = _text(input_dir, "input_dir")
    if not directory:
        raise ValueError("Photo Loader: Batch mode requires input_dir. Use an absolute folder "
                         "path or a folder relative to ComfyUI/input.")
    directory = os.path.expanduser(directory)
    if not os.path.isabs(directory):
        directory = os.path.join(folder_paths.get_input_directory(), directory)
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise ValueError(f"Photo Loader: input folder does not exist: {directory}")
    pattern = _text(pattern, "pattern") or "*"
    if os.path.isabs(pattern) or ".." in pattern.replace("\\", "/").split("/"):
        raise ValueError("Photo Loader: pattern must be a relative filename glob inside "
                         "input_dir (for example *.jpg or **/*.png).")
    files = sorted({os.path.abspath(path) for path in
                    glob.glob(os.path.join(directory, pattern), recursive=True)
                    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in EXTENSIONS},
                   key=_natural_key)
    if not files:
        raise ValueError(f"Photo Loader: no supported images in '{directory}' matching '{pattern}'.")
    return (directory, pattern), files


def _single_path(image):
    image = _text(image, "image")
    if not image or image == EMPTY_SELECTION:
        raise ValueError("Photo Loader: select or upload an image for Single mode.")
    path = folder_paths.get_annotated_filepath(image)
    if not os.path.isfile(path):
        raise ValueError(f"Photo Loader: selected image no longer exists: {path}. "
                         "Select it again or upload a replacement.")
    if os.path.splitext(path)[1].lower() not in EXTENSIONS:
        raise ValueError(f"Photo Loader: unsupported image extension: {path}")
    return path


class PortraitPhotoLoaderPreview:
    """One photo per execution, with independent and transactional batch state."""

    CATEGORY = "PortraitUtils/IO"
    DESCRIPTION = ("Independent folder sequencing, EXIF orientation, optional "
                   "sRGB conversion, and source info. Does not crop or resize.")
    RETURN_TYPES = ("IMAGE", SOURCE_INFO_TYPE)
    RETURN_NAMES = ("image", "source_info")
    FUNCTION = "load"

    def __init__(self):
        self._cursor = TransactionalCursor()
        self._positions = self._cursor.positions
        self._lock = self._cursor.lock

    @classmethod
    def INPUT_TYPES(cls):
        directory = folder_paths.get_input_directory()
        files = sorted((entry.name for entry in os.scandir(directory)
                        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in EXTENSIONS),
                       key=_natural_key) if os.path.isdir(directory) else []
        return {"required": {
            "mode": (["Single", "Batch"], {"default": "Single"}),
            "input_dir": ("STRING", {"default": "", "tooltip": "Batch folder; relative to ComfyUI/input or absolute."}),
            "output_dir": ("STRING", {"default": "", "tooltip": "Passed to Save Info from Source; blank keeps saver defaults."}),
            "pattern": ("STRING", {"default": "*", "tooltip": "Batch glob; **/*.png includes subfolders."}),
            "strip_trailing_numbers": ("BOOLEAN", {"default": False, "tooltip": "Strip only a trailing copy marker such as (2), not normal filename digits."}),
            "repeat_last": ("BOOLEAN", {"default": False, "tooltip": "Batch: hold the last successfully loaded photo; first use loads the first file."}),
            "image": (files or [EMPTY_SELECTION], {"image_upload": True}),
            "color_management": (COLOR_POLICIES, {"default": "Convert to sRGB", "tooltip": "Convert embedded 8-bit ICC profiles to sRGB. Untagged photos are assumed sRGB."}),
            "alpha_background": (ALPHA_BACKGROUNDS, {"default": "White", "tooltip": "Composite transparency onto this background; no alpha mask is emitted."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    def _select(self, key, files, repeat_last):
        return self._cursor.select(key, files, repeat=repeat_last)

    def load(self, mode, input_dir, output_dir, pattern, strip_trailing_numbers,
             repeat_last, image, color_management="Convert to sRGB",
             alpha_background="White", unique_id=None):
        boolean(strip_trailing_numbers, "Photo Loader.strip_trailing_numbers")
        boolean(repeat_last, "Photo Loader.repeat_last")
        if mode not in ("Single", "Batch"):
            raise ValueError("Photo Loader: mode must be Single or Batch.")
        output_dir = _text(output_dir, "output_dir")
        if unique_id is not None:
            with _LIVE_LOCK:
                _LIVE_NODES[str(unique_id)] = self
        with self._lock:
            key = None
            if mode == "Batch":
                key, files = _batch_files(input_dir, pattern)
                path = self._select(key, files, repeat_last)
            else:
                path = _single_path(image)
            tensor = _decode(path, color_management, alpha_background)
            stem = os.path.splitext(os.path.basename(path))[0]
            if strip_trailing_numbers:
                stem = re.sub(r"\s*\([0-9]+\)$", "", stem).strip() or stem
            info = validate_config(SourceInfoV2(stem, output_dir, int(tensor.shape[2]), int(tensor.shape[1])), "source_info")
            if key is not None:
                # Commit only after decoding and preparing every output succeeds.
                self._cursor.commit(key, path)
            return tensor, info

    @classmethod
    def IS_CHANGED(cls, mode, input_dir, output_dir, pattern, strip_trailing_numbers,
                   repeat_last, image, color_management="Convert to sRGB",
                   alpha_background="White", unique_id=None):
        if mode == "Batch" and not repeat_last:
            return float("nan")  # A new queued execution must advance, not reuse an old photo.
        try:
            if mode == "Batch":
                key, files = _batch_files(input_dir, pattern)
                with _LIVE_LOCK:
                    node = _LIVE_NODES.get(str(unique_id)) if unique_id is not None else None
                if node is None:
                    path = files[0]
                else:
                    with node._lock:
                        path = node._select(key, files, True)
            else:
                path = _single_path(image)
            # Other widget values are already included in ComfyUI's cache key.
            return _fingerprint(path)
        except (OSError, ValueError):
            return float("nan")  # Force execution to present the actionable load error.

    @classmethod
    def VALIDATE_INPUTS(cls, mode=None, input_dir=None, pattern=None, image=None):
        # Explicit image validation also bypasses irrelevant dropdown membership
        # in Batch mode. Linked values are validated when load() resolves them.
        try:
            if mode is None:
                return True
            if mode not in ("Single", "Batch"):
                return "Photo Loader: mode must be Single or Batch."
            if mode == "Batch":
                if input_dir is not None and pattern is not None:
                    _batch_files(input_dir, pattern)
            elif image is not None:
                _single_path(image)
            return True
        except (OSError, ValueError) as exc:
            return str(exc)


NODE_CLASS_MAPPINGS = {"PortraitPhotoLoaderPreview": PortraitPhotoLoaderPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitPhotoLoaderPreview": "Load Image + Source Info"}
