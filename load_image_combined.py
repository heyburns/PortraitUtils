# load_image_combined.py
import os
import re
import glob
import numpy as np
from PIL import Image, ImageOps
import torch
import folder_paths

from .core.photo_io import _fingerprint
from .core.state import TransactionalCursor

# ===========================
# Shared utils / constants 
# ===========================

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
_VALID_EXTS = VALID_EXTS  # alias for any older references


def _coerce_str(value) -> str:
    """Accept strings coming in as plain text, tuples, lists, or simple dict wrappers."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _coerce_str(value[0])
    if isinstance(value, dict):
        for key in ("string", "value", "text", "path"):
            if key in value:
                return _coerce_str(value[key])
    return str(value)


def _coerce_pattern(value) -> str:
    s = _coerce_str(value).strip()
    if not s or s.lower() in {"<none>", "none"}:
        return "*"
    # Treat legacy numeric values (likely old batch_index) as request for default "*"
    try:
        float(s)
        return "*"
    except ValueError:
        pass
    return s


def _resolve_input_dir(input_dir: str) -> str:
    """Resolve input directory against ComfyUI's input folder if relative."""
    if not input_dir:
        return folder_paths.get_input_directory()
    if os.path.isabs(input_dir):
        return input_dir
    return os.path.join(folder_paths.get_input_directory(), input_dir)

def _basename_no_ext(filename: str, strip_numbers=False) -> str:
    """Remove extension and optionally strip trailing (1)/(2) from filenames."""
    base = os.path.splitext(filename)[0]
    if strip_numbers:
        base = re.sub(r"\(\d+\)$", "", base).strip()
    return base

# ============================================================
# Node: Load Image (Combined)
# ============================================================

class LoadImageCombined:
    """Internal loading implementation inherited by LoadImageCombinedV2."""

    def __init__(self):
        self._cursor = TransactionalCursor()

    @classmethod
    def INPUT_TYPES(s):
        # Enumerate files directly from input dir,
        # not folder_paths.get_filename_list("input")
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "mode": (["Single", "Batch"], {"default": "Single"}),
                "input_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path or relative to ComfyUI/input"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Output Directory"
                }),
                "pattern": ("STRING", {
                    "default": "*",
                    "multiline": False,
                    "placeholder": "e.g., *.png"
                }),
                "strip_trailing_numbers": ("BOOLEAN", {"default": False}),
                "repeat_last": ("BOOLEAN", {"default": False}),
                "image": (sorted(files), {"image_upload": True}),
            }
        }

    CATEGORY = "PortraitUtils/IO"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "filename_no_ext", "output_dir", "width", "height")
    FUNCTION = "load_image"

    def _load_pil(self, path):
        """Open, EXIF-transpose, and convert to float32 RGB numpy array.

        Both the original Image.open handle and any separate EXIF-transposed
        copy are explicitly closed to avoid file-handle and memory leaks.
        """
        raw = Image.open(path)
        transposed = None
        try:
            transposed = ImageOps.exif_transpose(raw)
            # Both branches of the original if/else were identical — just convert.
            arr = np.array(transposed.convert("RGB")).astype(np.float32) / 255.0
        finally:
            # Always close the raw handle.
            raw.close()
            # exif_transpose may return a NEW Image; close it too if so.
            if transposed is not None and transposed is not raw:
                transposed.close()
        return arr

    def _single_mode(self, image_choice, strip_numbers):
        image_path = folder_paths.get_annotated_filepath(image_choice)
        filename = os.path.basename(image_path)
        filename_no_ext = _basename_no_ext(filename, strip_numbers)
        arr = self._load_pil(image_path)
        img_t = torch.from_numpy(arr)[None,]
        h, w = arr.shape[0], arr.shape[1]
        return img_t, filename_no_ext, int(w), int(h)

    def _gather_batch_files(self, input_dir, pattern):
        if input_dir is None or str(input_dir).strip() == "":
            raise ValueError("Batch mode requires 'input_dir'. Please specify a folder (absolute or relative to ComfyUI/input).")
        base = _resolve_input_dir(input_dir)
        if not os.path.isdir(base):
            raise ValueError(f"Input directory not found: {base}")

        pat = pattern.strip() if pattern and pattern.strip() != "" else "*"
        search_glob = os.path.join(base, pat)
        candidates = glob.glob(search_glob)
        files = [
            f for f in candidates
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in _VALID_EXTS
        ]
        files_sorted = sorted(files, key=lambda s: s.lower())
        return base, pat, files_sorted

    def _batch_mode_auto_advance(self, input_dir, pattern, strip_numbers, repeat_last):
        base, pat, files_sorted = self._gather_batch_files(input_dir, pattern)
        if not files_sorted:
            raise ValueError(f"No images found in '{base}' with pattern '{pat}'")

        key = (os.path.abspath(base), pat, bool(strip_numbers))
        with self._cursor.lock:
            path = self._cursor.select(key, files_sorted, repeat=repeat_last)
            # A failed decode leaves the cursor unchanged, so retry gets the same photo.
            arr = self._load_pil(path)
            filename_no_ext = _basename_no_ext(os.path.basename(path), strip_numbers)
            img_t = torch.from_numpy(arr)[None,]
            h, w = arr.shape[:2]
            self._cursor.commit(key, path)
        return img_t, filename_no_ext, int(w), int(h)

    def load_image(self, mode, input_dir, output_dir, pattern, strip_trailing_numbers, repeat_last, image):
        input_dir = _coerce_str(input_dir).strip()
        output_dir = _coerce_str(output_dir).strip()
        pattern = _coerce_pattern(pattern)
        if str(mode) == "Batch":
            if not input_dir:
                raise ValueError("Batch mode requires 'input_dir'. Please specify a folder (absolute or relative to ComfyUI/input).")
            img_t, filename_no_ext, w, h = self._batch_mode_auto_advance(
                input_dir, pattern, strip_trailing_numbers, repeat_last
            )
            return img_t, filename_no_ext, str(output_dir or ""), w, h
        else:
            img_t, filename_no_ext, w, h = self._single_mode(image, strip_trailing_numbers)
            return img_t, filename_no_ext, str(output_dir or ""), w, h

    @classmethod
    def IS_CHANGED(s, mode, input_dir, output_dir, pattern, strip_trailing_numbers, repeat_last, image):
        if str(mode) == "Batch":
            # The cursor is per node instance, so each queued run must execute.
            return float("nan")
        try:
            return _fingerprint(folder_paths.get_annotated_filepath(image))
        except (OSError, ValueError):
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(s, mode, input_dir, output_dir, pattern, strip_trailing_numbers, repeat_last, image):
        input_dir = _coerce_str(input_dir).strip()
        pattern = _coerce_pattern(pattern)
        if str(mode) == "Batch":
            if not input_dir:
                return "Batch mode requires 'input_dir'."
            base = _resolve_input_dir(input_dir)
            if not os.path.isdir(base):
                return f"Input directory not found: {base}"
            pat = pattern.strip() if pattern and pattern.strip() != "" else "*"
            candidates = [
                f for f in glob.glob(os.path.join(base, pat))
                if os.path.isfile(f) and os.path.splitext(f)[1].lower() in _VALID_EXTS
            ]
            if not candidates:
                return f"No images found in '{base}' with pattern '{pat}'"
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
