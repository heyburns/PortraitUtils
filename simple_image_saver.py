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

        fmt = _coerce_str(file_format).strip().upper()
        if fmt not in {"PNG", "JPG"}:
            raise ValueError("Format must be either 'PNG' or 'JPG'.")

        # Clamp quality and warn loudly when a caller passes an out-of-range value via
        # the API (the widget schema already enforces [0, 100] from the UI).
        clamped_quality = int(max(0, min(100, jpeg_quality)))
        if clamped_quality != int(jpeg_quality):
            logging.warning(
                "SimpleImageSaver: jpeg_quality %d is out of range [0, 100]; clamped to %d.",
                jpeg_quality,
                clamped_quality,
            )
        jpeg_quality = clamped_quality

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

            # Image.fromarray is inside the try block so that if it raises, `image` is
            # never unbound and the finally clause never produces a secondary NameError.
            image: Optional[Image.Image] = None
            try:
                image = Image.fromarray(array)

                name_parts = [base_name]
                if suffix_component:
                    name_parts.append(suffix_component)
                if batch_count > 1:
                    name_parts.append(f"{index:04d}")
                final_name = "-".join(name_parts)
                extension = ".png" if fmt == "PNG" else ".jpg"
                final_filename = f"{final_name}{extension}"

                if unique_filenames:
                    # Atomically claim a filename slot using O_CREAT | O_EXCL.  This
                    # eliminates the TOCTOU race that exists between os.path.exists()
                    # and a subsequent open/write in a multi-process environment.
                    counter = 1
                    while True:
                        file_path = os.path.join(resolved_dir, final_filename)
                        try:
                            fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                            os.close(fd)
                            break  # Slot claimed; PIL will overwrite the empty placeholder.
                        except FileExistsError:
                            final_filename = f"{final_name}-{counter:04d}{extension}"
                            counter += 1
                else:
                    file_path = os.path.join(resolved_dir, final_filename)

                if fmt == "PNG":
                    metadata = _encode_png_metadata(
                        prompt if should_embed_metadata else None,
                        extra_pnginfo if should_embed_metadata else None,
                    )
                    image.save(file_path, pnginfo=metadata, compress_level=4)
                else:
                    comment = _encode_jpeg_comment(
                        prompt if should_embed_metadata else None,
                        extra_pnginfo if should_embed_metadata else None,
                    )
                    _save_jpeg(image, file_path, jpeg_quality, comment)

                rel_sub = self._relative_subfolder(resolved_dir, base_output_abs)

                if rel_sub == "" and resolved_dir != base_output_abs:
                    # The real file has been written to an absolute path outside the
                    # ComfyUI output tree.  Write a lightweight proxy to the temp dir
                    # so the UI preview widget has something to display.
                    temp_dir = folder_paths.get_temp_directory()
                    temp_name = (
                        f"proxy_{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}{extension}"
                    )
                    temp_path = os.path.join(temp_dir, temp_name)

                    if fmt == "PNG":
                        image.save(temp_path, compress_level=1)
                    else:
                        # Use the same quality as the real save — not a hard-coded value.
                        _save_jpeg(image, temp_path, jpeg_quality, None)

                    results.append({
                        "filename": temp_name,
                        "subfolder": "",
                        "type": "temp",
                    })
                else:
                    results.append(
                        {
                            "filename": final_filename,
                            "subfolder": rel_sub,
                            "type": "output",
                        }
                    )
            finally:
                if image is not None:
                    image.close()

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
