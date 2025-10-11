# load_image_combined.py
import os
import re
import glob
import hashlib
import numpy as np
from PIL import Image, ImageOps
import torch
import folder_paths

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

# ----------------------------------------------------------------
# Batch listing index tracking (per-listing, in-process persistence)
# Keyed by (base, listing key), advances on each call unless 
# repeat_last=True; not persisted to disk.
# ----------------------------------------------------------------
_INDEX_STATE = {}  # { base_dir: { listing_key: last_used_index } }

def _listing_key(files_sorted, strip_numbers, pattern):
    """
    Stable identifier for this specific listing. Combine directory path,
    pattern, strip flag, and file names to disambiguate.
    """
    if not files_sorted:
        return "empty"
    dir_path = os.path.dirname(files_sorted[0])
    parts = [os.path.basename(f).lower() for f in files_sorted]
    seed = f"{os.path.abspath(dir_path).lower()}|pat={pattern}|strip={bool(strip_numbers)}|" + "|".join(parts)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()

def _peek_last_index(base_dir, key):
    return int(_INDEX_STATE.get(base_dir, {}).get(key, -1))

def _choose_index_and_update(base_dir, key, n, repeat_last):
    last = _peek_last_index(base_dir, key)
    if last < 0:
        next_idx = 0
    else:
        next_idx = min(last, n - 1) if repeat_last else ((last + 1) % n)
    d = _INDEX_STATE.get(base_dir)
    if d is None:
        d = {}
        _INDEX_STATE[base_dir] = d
    d[key] = next_idx
    return next_idx

# ============================================================
# Node: Load Image (Combined)
# ============================================================

class LoadImageCombined:
    def __init__(self):
        pass

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

    CATEGORY = "image/io"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "filename_no_ext", "output_dir", "width", "height")
    FUNCTION = "load_image"

    def _load_pil(self, path):
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        # Always discard alpha (convert to RGB) to avoid downstream errors
        if "A" in im.getbands():
            arr = np.array(im.convert("RGB")).astype(np.float32) / 255.0
        else:
            arr = np.array(im.convert("RGB")).astype(np.float32) / 255.0
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

        key = _listing_key(files_sorted, strip_numbers, pat)
        n = len(files_sorted)
        use_idx = _choose_index_and_update(base, key, n, repeat_last)

        path = files_sorted[use_idx]
        filename = os.path.basename(path)
        filename_no_ext = _basename_no_ext(filename, strip_numbers)
        arr = self._load_pil(path)
        img_t = torch.from_numpy(arr)[None,]
        h, w = arr.shape[0], arr.shape[1]
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
        input_dir = _coerce_str(input_dir).strip()
        output_dir = _coerce_str(output_dir).strip()
        pattern = _coerce_pattern(pattern)
        if str(mode) != "Batch":
            image_path = folder_paths.get_annotated_filepath(image)
            m = hashlib.sha256()
            # Hash file bytes to detect actual content changes; also include strip flag
            try:
                with open(image_path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        m.update(chunk)
            except Exception:
                m.update(b"ERROR:cannot_read_single_image")
            m.update(f"|strip={bool(strip_trailing_numbers)}|".encode("utf-8"))
            m.update(f"|output_dir={str(output_dir)}|".encode("utf-8"))
            return m.digest().hex()

        # Batch mode hashing: reflect directory listing, file sizes/mtimes, last-used index, strip flag, and pattern.
        if not input_dir:
            m = hashlib.sha256()
            m.update(b"ERROR:no_input_dir_for_batch")
            return m.digest().hex()

        base = _resolve_input_dir(input_dir)
        pat = pattern if pattern else "*"
        candidates = [
            f for f in glob.glob(os.path.join(base, pat))
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in _VALID_EXTS
        ]
        files_sorted = sorted(candidates, key=lambda s: s.lower())

        m = hashlib.sha256()
        for fp in files_sorted:
            try:
                st = os.stat(fp)
                m.update(os.path.abspath(fp).lower().encode("utf-8"))
                m.update(str(st.st_size).encode("utf-8"))
                m.update(str(int(st.st_mtime)).encode("utf-8"))
            except Exception:
                m.update(os.path.abspath(fp).lower().encode("utf-8"))
            m.update(b"\n")

        key = _listing_key(files_sorted, strip_trailing_numbers, pat)
        last_used = _peek_last_index(base, key)
        m.update(f"|last_used={last_used}|".encode("utf-8"))
        m.update(f"|strip={bool(strip_trailing_numbers)}|".encode("utf-8"))
        m.update(f"|pattern={pat}|".encode("utf-8"))
        m.update(f"|output_dir={str(output_dir)}|".encode("utf-8"))
        return m.digest().hex()

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
