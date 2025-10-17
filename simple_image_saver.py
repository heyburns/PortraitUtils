import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


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
    safe = safe.strip().rstrip(".")
    if not safe:
        if allow_empty:
            return ""
        raise ValueError("Filename component resolves to an empty string after sanitization.")
    if safe in {".", ".."}:
        raise ValueError(f"Disallowed filename component: {safe}")
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


class SimpleImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Directory path (absolute or relative to ComfyUI/output)."}),
                "filename": ("STRING", {"default": "ComfyUI", "multiline": False, "tooltip": "Base filename without extension."}),
                "suffix": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional suffix appended with a dash when provided."}),
                "format": (["PNG", "JPG"], {"default": "PNG"}),
                "jpeg_quality": ("INT", {"default": 95, "min": 0, "max": 100, "tooltip": "JPEG quality (0-100)."}),
                "include_metadata": ("BOOLEAN", {"default": True, "tooltip": "Include workflow metadata (prompt + extras)."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save(
        self,
        images: torch.Tensor,
        output_path: str,
        filename: str,
        suffix: str,
        format: str,
        jpeg_quality: int,
        include_metadata: bool,
        prompt: Optional[Dict[str, Any]] = None,
        extra_pnginfo: Optional[Dict[str, Any]] = None,
    ):
        if images is None:
            return {"ui": {"images": []}}

        if isinstance(images, (list, tuple)) and len(images) == 0:
            return {"ui": {"images": []}}

        if not isinstance(images, torch.Tensor):
            raise TypeError("Expected `images` to be a torch.Tensor or empty input.")

        if images.numel() == 0:
            return {"ui": {"images": []}}

        if images.ndim == 3:
            images = images.unsqueeze(0)
        elif images.ndim != 4:
            raise ValueError("Expected image tensor with shape [batch, height, width, channels].")

        if images.shape[0] == 0:
            return {"ui": {"images": []}}

        resolved_dir = _resolve_output_directory(output_path)
        os.makedirs(resolved_dir, exist_ok=True)

        base_name = _sanitize_name_component(filename)
        suffix_component = _sanitize_name_component(suffix, allow_empty=True)

        fmt = _coerce_str(format).strip().upper()
        if fmt not in {"PNG", "JPG"}:
            raise ValueError("Format must be either 'PNG' or 'JPG'.")

        jpeg_quality = int(max(0, min(100, jpeg_quality)))

        should_embed_metadata = bool(include_metadata)

        batch_count = images.shape[0]
        results = []
        base_output = folder_paths.get_output_directory()
        base_output_abs = os.path.abspath(base_output)

        for index in range(batch_count):
            tensor = images[index]
            if tensor.ndim != 3:
                raise ValueError("Each image tensor must have shape [height, width, channels].")
            array = tensor.clamp(0, 1).cpu().numpy()
            array = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)

            image = Image.fromarray(array)

            name_parts = [base_name]
            if suffix_component:
                name_parts.append(suffix_component)
            if batch_count > 1:
                name_parts.append(f"{index:04d}")
            final_name = "-".join(name_parts)
            extension = ".png" if fmt == "PNG" else ".jpg"
            final_filename = f"{final_name}{extension}"
            file_path = os.path.join(resolved_dir, final_filename)

            if fmt == "PNG":
                metadata = _encode_png_metadata(prompt if should_embed_metadata else None, extra_pnginfo if should_embed_metadata else None)
                image.save(file_path, pnginfo=metadata, compress_level=4)
            else:
                comment = _encode_jpeg_comment(prompt if should_embed_metadata else None, extra_pnginfo if should_embed_metadata else None)
                save_kwargs = {"quality": jpeg_quality, "subsampling": 0 if jpeg_quality >= 95 else "keep"}
                if comment:
                    save_kwargs["comment"] = comment
                image.save(file_path, format="JPEG", **save_kwargs)

            results.append(
                {
                    "filename": final_filename,
                    "subfolder": self._relative_subfolder(resolved_dir, base_output_abs),
                    "type": "output",
                }
            )

        return {"ui": {"images": results}}

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


NODE_CLASS_MAPPINGS = {
    "SimpleImageSaver": SimpleImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleImageSaver": "Simple Image Saver",
}
