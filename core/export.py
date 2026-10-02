"""Atomic file export with an explicit environment; no ComfyUI dependencies."""
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import zlib
import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
from .tensors import image_tensor, integer
from .validation import boolean
JPEG_COMMENT_MAX_BYTES = 65533

_LOG = logging.getLogger("PortraitUtils.image_saver_preview")

def _component(value, label, fallback=""):
    if not isinstance(value, str):
        raise ValueError(f"Image Saver preview: {label} must be text.")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", value.strip()).rstrip(". ") or fallback
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                      *(f"LPT{i}" for i in range(1, 10))}:
        value = "_" + value
    if len(value.encode("utf-8")) > 180:
        raise ValueError(f"Image Saver preview: {label} is too long; shorten it to at most 180 UTF-8 bytes.")
    return value

def _directory(value, output_root):
    if not isinstance(value, str):
        raise ValueError("Image Saver preview: output_path must be text.")
    base = Path(output_root).resolve()
    requested = Path(value.strip()).expanduser() if value.strip() else base
    relative = not requested.is_absolute()
    target = ((base / requested) if relative else requested).resolve()
    if relative and target != base and base not in target.parents:
        raise ValueError("Image Saver preview: relative output_path escapes ComfyUI/output "
                         "(including through a symlink). Use an explicit absolute path if intended.")
    return base, target

def _sanitize_metadata_numbers(value, path, replaced, active):
    """Copy metadata containers, replacing non-finite float values without changing the input."""
    if isinstance(value, float) and not math.isfinite(value):
        replaced["count"] += 1
        if len(replaced["paths"]) < 5:
            replaced["paths"].append(path)
        return None
    if not isinstance(value, (dict, list, tuple)):
        return value
    identity = id(value)
    if identity in active:
        raise ValueError(f"circular workflow metadata at {path}")
    active.add(identity)
    try:
        if isinstance(value, dict):
            return {key: _sanitize_metadata_numbers(item, f"{path}[{key!r}]", replaced, active)
                    for key, item in value.items()}
        return [_sanitize_metadata_numbers(item, f"{path}[{index}]", replaced, active)
                for index, item in enumerate(value)]
    finally:
        active.remove(identity)


def _metadata(include, prompt, extra, file_format="PNG", disabled=False):
    if not include or disabled:
        return None, None
    payload = {}
    if prompt is not None:
        payload["prompt"] = prompt
    if isinstance(extra, dict) and extra:
        payload["extra_pnginfo"] = extra
    if not payload:
        return None, None
    info = PngInfo()
    replaced = {"count": 0, "paths": []}
    try:
        payload = _sanitize_metadata_numbers(payload, "metadata", replaced, set())
        values = {}
        if "prompt" in payload:
            values["prompt"] = payload["prompt"]
        if "extra_pnginfo" in payload:
            values.update({key: value for key, value in payload["extra_pnginfo"].items()
                           if key != "prompt"})
        if not values:
            return None, None
        for key, value in values.items():
            if not isinstance(key, str) or "\0" in key or not 1 <= len(key.encode("latin-1")) <= 79:
                raise ValueError("PNG metadata keys must be 1–79 Latin-1 bytes without NUL characters")
            info.add_text(key, json.dumps(value, ensure_ascii=False, allow_nan=False))
        comment = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"Image Saver preview: workflow metadata is not valid JSON/PNG text: {exc}. "
                         "Fix the metadata or disable include_metadata.") from exc
    if replaced["count"]:
        _LOG.warning("Image Saver preview: replaced %d non-finite workflow metadata value(s) "
                     "with JSON null at %s%s. Original workflow data was not changed.",
                     replaced["count"], ", ".join(replaced["paths"]),
                     " (first five shown)" if replaced["count"] > len(replaced["paths"]) else "")
    if len(comment) > JPEG_COMMENT_MAX_BYTES:
        if file_format == "JPG":
            _LOG.warning("Image Saver preview: JPEG workflow metadata exceeds the comment limit; omitted. "
                         "Use PNG to retain this workflow.")
        comment = None
    return info, comment

def _png16(path, array, metadata):
    if array.shape[-1] == 1:
        encoded_input = array[..., 0]
    else:
        encoded_input = array[..., [2, 1, 0] if array.shape[-1] == 3 else [2, 1, 0, 3]]
    success, data = cv2.imencode(".png", np.ascontiguousarray(encoded_input), [cv2.IMWRITE_PNG_COMPRESSION, 4])
    if not success:
        raise ValueError("Image Saver preview: PNG16 encoder failed.")
    data = data.tobytes()
    # PNG starts with an 8-byte signature and the 25-byte IHDR chunk. Text
    # chunks can precede IDAT; their CRCs do not affect the encoded image data.
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR" or data[24] != 16:
        raise ValueError("Image Saver preview: encoder did not produce a 16-bit PNG.")
    chunks = []
    if metadata is not None:
        # Pillow 10 uses two-element tuples for ordinary text; newer releases
        # also include the after-IDAT flag. Text belongs before IDAT here.
        for chunk in metadata.chunks:
            kind, payload = chunk[:2]
            chunks.append(struct.pack(">I", len(payload)) + kind + payload
                          + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    with open(path, "wb") as handle:
        handle.write(data[:33])
        for chunk in chunks:
            handle.write(chunk)
        handle.write(data[33:])

def _encode(path, array, file_format, quality, metadata, comment, png_bit_depth, matte):
    if file_format == "PNG" and png_bit_depth == "16-bit":
        _png16(path, np.floor(array * 65535 + 0.5).astype(np.uint16), metadata)
        return
    if file_format == "JPG" and array.shape[-1] == 4:
        alpha = array[..., 3:4]
        array = array[..., :3] * alpha + (1 if matte == "White" else 0) * (1 - alpha)
    pixels = np.floor(array * 255 + 0.5).astype(np.uint8)
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    with Image.fromarray(pixels) as image:
        if file_format == "PNG":
            image.save(path, format="PNG", pnginfo=metadata, compress_level=4)
        else:
            options = {"quality": quality, "subsampling": 0 if quality >= 90 else 2}
            if comment:
                options["comment"] = comment
            image.save(path, format="JPEG", **options)

def _commit(temporary, directory, stem, extension, unique):
    candidate = directory / (stem + extension)
    if not unique:
        os.replace(temporary, candidate)
        return candidate
    counter = 0
    while True:
        try:
            # Atomic no-overwrite publication of the COMPLETE encoded file.
            # Unlike empty O_EXCL placeholders, concurrent readers see no
            # half-written image and a failed encoder leaves no claimed slot.
            os.link(temporary, candidate)
            return candidate
        except FileExistsError:
            counter += 1
            candidate = directory / f"{stem}-{counter:04d}{extension}"
        except OSError as exc:
            raise OSError("Image Saver preview: cannot atomically publish a unique filename in "
                          f"{directory}. Use a filesystem supporting hard links; no existing file was overwritten.") from exc


@dataclass(frozen=True)
class ExportEnvironment:
    output_root: str
    temp_root: str
    metadata_disabled: bool = False


class PhotoExportEngine:
    def __init__(self, environment, encoder=None):
        self.environment = environment
        self.encoder = encoder or _encode

    @torch.no_grad()
    def save(self, images, output_path="", filename="ComfyUI", suffix="", file_format="PNG",
             jpeg_quality=95, include_metadata=True, unique_filenames=True, prompt=None,
             extra_pnginfo=None, png_bit_depth="8-bit", jpeg_alpha_background="White"):
        boolean(include_metadata, "Image Saver preview.include_metadata")
        boolean(unique_filenames, "Image Saver preview.unique_filenames")
        if images is None or (isinstance(images, (list, tuple)) and not images):
            return {"ui": {"images": []}}
        if isinstance(images, torch.Tensor) and images.numel() == 0:
            return {"ui": {"images": []}}
        images = image_tensor(images, "Image Saver preview: images")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("Image Saver preview: images contain NaN or infinite pixels; nothing was saved.")
        if file_format not in ("PNG", "JPG") or png_bit_depth not in ("8-bit", "16-bit"):
            raise ValueError("Image Saver preview: select PNG/JPG and 8-bit/16-bit PNG depth.")
        if jpeg_alpha_background not in ("White", "Black"):
            raise ValueError("Image Saver preview: JPEG alpha background must be White or Black.")
        integer(jpeg_quality, "Image Saver preview: jpeg_quality", 0, 100)
        base, directory = _directory(output_path, self.environment.output_root)
        stem = _component(filename, "filename", "ComfyUI")
        suffix = _component(suffix, "suffix")
        if suffix:
            stem += "-" + suffix
        if len(stem.encode("utf-8")) > 230:
            raise ValueError("Image Saver preview: combined filename and suffix are too long; shorten them.")
        metadata, comment = _metadata(include_metadata, prompt, extra_pnginfo, file_format, self.environment.metadata_disabled)
        directory.mkdir(parents=True, exist_ok=True)
        extension = ".png" if file_format == "PNG" else ".jpg"
        results = []
        for index, frame in enumerate(images):
            name = stem + (f"-{index:04d}" if len(images) > 1 else "")
            temporary = None
            try:
                fd, temporary = tempfile.mkstemp(prefix=".portrait-save-", suffix=".tmp", dir=directory)
                os.close(fd)
                array = frame.detach().to(device="cpu", dtype=torch.float64 if frame.dtype == torch.float64
                                          else torch.float32).clamp(0, 1).numpy()
                self.encoder(temporary, array, file_format, jpeg_quality, metadata, comment,
                        png_bit_depth, jpeg_alpha_background)
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                path = _commit(temporary, directory, name, extension, unique_filenames)
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
            if directory == base or base in directory.parents:
                relative = directory.relative_to(base)
                results.append({"filename": path.name, "subfolder": "" if relative == Path(".")
                                else relative.as_posix(), "type": "output"})
            else:
                # Copy encoded bytes, not a re-encode that could change quality.
                temp_directory = Path(self.environment.temp_root)
                temp_directory.mkdir(parents=True, exist_ok=True)
                proxy = None
                try:
                    fd, proxy = tempfile.mkstemp(prefix="portrait-preview-", suffix=extension, dir=temp_directory)
                    with os.fdopen(fd, "wb") as target, path.open("rb") as source:
                        shutil.copyfileobj(source, target)
                except OSError as exc:
                    if proxy is not None:
                        Path(proxy).unlink(missing_ok=True)
                    raise OSError(f"Image Saver preview: image was saved successfully to {path}, "
                                  "but its UI preview could not be created. Check the ComfyUI temp directory.") from exc
                results.append({"filename": Path(proxy).name, "subfolder": "", "type": "temp"})
        return {"ui": {"images": results}}
