import json
import logging
import os
import random
import string
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args

from .core.export import ExportEnvironment, PhotoExportEngine


JPEG_COMMENT_MAX_BYTES = 65500  # Conservative buffer below 64 KiB JPEG comment cap.
INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _sanitize_name_component(name: str, allow_empty: bool = False) -> str:
    cleaned = _coerce_str(name).strip()
    if not cleaned:
        return "" if allow_empty else "_"
    safe = cleaned.translate({ord(ch): "_" for ch in INVALID_FILENAME_CHARS})
    # rstrip(".") already collapses "." and ".." to empty — no separate check needed.
    safe = safe.strip().rstrip(".")
    if not safe:
        if allow_empty:
            return ""
        raise ValueError("Filename component resolves to an empty string after sanitization.")
    return safe


def _resolve_output_directory(path_value: str) -> str:
    base_dir = os.path.abspath(folder_paths.get_output_directory())
    requested = _coerce_str(path_value).strip()
    if not requested:
        return base_dir

    if os.path.isabs(requested):
        return os.path.abspath(requested)

    normalized = os.path.normpath(requested)
    if normalized.startswith(".."):
        raise ValueError("Relative output path cannot traverse above the base output directory.")

    resolved = os.path.abspath(os.path.join(base_dir, normalized))
    common = os.path.commonpath([base_dir, resolved])
    if common != base_dir:
        raise ValueError("Resolved output directory escapes the base output directory.")
    return resolved


def _encode_png_metadata(prompt: Optional[Any], extra_pnginfo: Optional[Any]) -> Optional[PngInfo]:
    if args.disable_metadata:
        return None

    has_prompt = prompt is not None
    has_extra = isinstance(extra_pnginfo, dict) and bool(extra_pnginfo)
    if not has_prompt and not has_extra:
        return None

    metadata = PngInfo()
    if has_prompt:
        metadata.add_text("prompt", json.dumps(prompt))
    if has_extra:
        for key, value in extra_pnginfo.items():
            metadata.add_text(key, json.dumps(value))
    return metadata


def _encode_jpeg_comment(
    prompt: Optional[Any],
    extra_pnginfo: Optional[Any],
) -> Optional[bytes]:
    if args.disable_metadata:
        return None

    payload: Dict[str, Any] = {}
    if prompt is not None:
        payload["prompt"] = prompt
    if isinstance(extra_pnginfo, dict) and extra_pnginfo:
        payload["extra_pnginfo"] = extra_pnginfo

    if not payload:
        return None

    try:
        encoded = json.dumps(payload, default=str).encode("utf-8")
    except (TypeError, ValueError):
        logging.warning("SimpleImageSaver: failed to serialize metadata to JSON; skipping metadata for JPEG.")
        return None

    if len(encoded) > JPEG_COMMENT_MAX_BYTES:
        logging.warning(
            "SimpleImageSaver: metadata payload (%d bytes) exceeds JPEG comment limit; metadata skipped.",
            len(encoded),
        )
        return None
    return encoded


def _save_jpeg(image: Image.Image, file_path: str, quality: int, comment: Optional[bytes]) -> None:
    """Save *image* as JPEG to *file_path* with consistent quality/subsampling settings.

    Extracted to a shared helper so that the primary save and the preview proxy copy
    both use identical settings — preventing the proxy from silently downgrading quality.

    JPEG does not support alpha or palette modes.  Any non-RGB image is converted
    directly to RGB, discarding transparency without compositing.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    save_kwargs: Dict[str, Any] = {"quality": quality}
    if quality >= 90:
        # Disable chroma subsampling at high quality to preserve colour fidelity.
        save_kwargs["subsampling"] = 0
    if comment:
        save_kwargs["comment"] = comment
    image.save(file_path, format="JPEG", **save_kwargs)


class SimpleImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Directory path (absolute or relative to ComfyUI/output)."}),
                "filename": ("STRING", {"default": "ComfyUI", "multiline": False, "tooltip": "Base filename without extension."}),
                "suffix": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional suffix appended with a dash when provided."}),
                # Renamed from "format" to "file_format" to avoid shadowing the Python built-in.
                # Existing saved workflows will need to reconnect this input after reloading.
                "file_format": (["PNG", "JPG"], {"default": "PNG"}),
                "jpeg_quality": ("INT", {"default": 95, "min": 0, "max": 100, "tooltip": "JPEG quality (0-100)."}),
                "include_metadata": ("BOOLEAN", {"default": True, "tooltip": "Include workflow metadata (prompt + extras)."}),
                "unique_filenames": ("BOOLEAN", {"default": True, "tooltip": "Append a counter to avoid overwriting existing files."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "PortraitUtils/IO"

    def save(
        self,
        images: torch.Tensor,
        output_path: str,
        filename: str,
        suffix: str,
        file_format: str,
        jpeg_quality: int,
        include_metadata: bool,
        unique_filenames: bool = True,
        prompt: Optional[Dict[str, Any]] = None,
        extra_pnginfo: Optional[Dict[str, Any]] = None,
    ):
        environment = ExportEnvironment(
            folder_paths.get_output_directory(),
            folder_paths.get_temp_directory(),
            args.disable_metadata,
        )
        return PhotoExportEngine(environment).save(
            images, output_path, filename, suffix, file_format, jpeg_quality,
            include_metadata, unique_filenames, prompt, extra_pnginfo,
        )

    @staticmethod
    def _relative_subfolder(target_dir: str, base_dir: str) -> str:
        target_abs = os.path.abspath(target_dir)
        try:
            common = os.path.commonpath([base_dir, target_abs])
        except ValueError:
            return ""
        if common != base_dir:
            return ""
        rel = os.path.relpath(target_abs, base_dir)
        return "" if rel == "." else rel.replace("\\", "/")
