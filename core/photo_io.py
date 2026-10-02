"""Photo decoding and file signatures with explicit color/alpha policy."""
from contextlib import ExitStack, closing
import hashlib
from io import BytesIO
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
import torch

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

COLOR_POLICIES = ["Convert to sRGB", "Keep encoded values"]

ALPHA_BACKGROUNDS = ["White", "Black"]

def _natural_key(path):
    return (tuple((1, int(part)) if part.isascii() and part.isdigit()
                  else (0, part.casefold())
                  for part in re.split(r"([0-9]+)", path)), path)

def _fingerprint(path):
    digest = hashlib.sha256()
    digest.update(os.path.abspath(path).encode("utf-8"))
    digest.update(b"\0")
    # Content, not coarse timestamps: replacing an image in the same second must
    # invalidate downstream caches. Stream instead of allocating another file copy.
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _bit_depth(raw, path):
    if raw.format == "TIFF":
        bits = raw.tag_v2.get(258, (8,))
        return max(bits) if isinstance(bits, (tuple, list)) else bits
    if raw.format == "PNG":
        with open(path, "rb") as handle:
            return handle.read(25)[24]  # IHDR bit depth, including RGB16 that PIL reduces to RGB8
    if raw.mode.startswith("I;16"):
        return 16
    return 32 if raw.mode in ("I", "F") else 8

def _orient_array(array, orientation):
    if orientation == 2:
        return array[:, ::-1]
    if orientation == 3:
        return array[::-1, ::-1]
    if orientation == 4:
        return array[::-1]
    if orientation == 5:
        return array.swapaxes(0, 1)
    if orientation == 6:
        return np.rot90(array, 3)
    if orientation == 7:
        return array.swapaxes(0, 1)[::-1, ::-1]
    if orientation == 8:
        return np.rot90(array, 1)
    return array

def _tiff16(path):
    # OpenCV's TIFF decoder can apply orientation internally even with
    # IMREAD_UNCHANGED, and can fail on transposed dimensions. Read raw samples
    # and apply the tag ourselves, exactly once, without going through RGB8.
    try:
        import tifffile
    except ImportError as exc:
        raise ValueError("16-bit TIFF decoding requires tifffile. Install this suite's "
                         "requirements in ComfyUI's Python environment.") from exc
    with tifffile.TiffFile(path) as file:
        page = file.pages[0]
        if page.photometric not in (0, 1, 2) or page.axes not in ("YX", "YXS", "SYX"):
            raise ValueError("Unsupported 16-bit TIFF color/layout. Export a standard RGB "
                             "or grayscale PNG/TIFF first.")
        orientation = page.tags.get("Orientation")
        orientation = orientation.value if orientation else 1
        premultiplied = bool(page.extrasamples and page.extrasamples[0] == 1)
        fallback = False
        try:
            data = page.asarray()
        except ValueError as exc:
            if "imagecodecs" not in str(exc):
                raise
            # Existing OpenCV can decode common LZW TIFFs losslessly. Only use
            # it when orientation/alpha cannot invoke its problematic transforms.
            if orientation != 1 or page.photometric not in (1, 2) or premultiplied:
                raise ValueError("This compressed 16-bit TIFF needs imagecodecs. Install "
                                 "imagecodecs in ComfyUI's Python environment, or export "
                                 "an uncompressed 16-bit RGB/grayscale TIFF or PNG.") from exc
            data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if data is None:
                raise ValueError("Cannot decode this compressed 16-bit TIFF. Install "
                                 "imagecodecs or export an uncompressed 16-bit copy.") from exc
            if data.ndim == 3 and data.shape[-1] in (3, 4):
                data = data[..., [2, 1, 0] if data.shape[-1] == 3 else [2, 1, 0, 3]]
            fallback = True
        if page.axes == "SYX" and not fallback:
            data = np.moveaxis(data, 0, -1)
        if page.photometric == 0:  # MINISWHITE grayscale
            data = 65535 - data
        return data, orientation, premultiplied

def _decode(path, color_management, alpha_background):
    if color_management not in COLOR_POLICIES or alpha_background not in ALPHA_BACKGROUNDS:
        raise ValueError("Photo Loader: invalid color-management or alpha-background setting.")
    try:
        with ExitStack() as stack:
            raw = stack.enter_context(Image.open(path))
            bits = _bit_depth(raw, path)
            profile = raw.info.get("icc_profile")
            if bits > 8:
                if bits != 16:
                    raise ValueError("32-bit/HDR images are not supported. Export a "
                                     "display-referred 8-bit or 16-bit PNG/TIFF first.")
                if profile and color_management == "Convert to sRGB":
                    raise ValueError("This 16-bit image has an ICC profile. The available ICC "
                                     "converter is 8-bit, so conversion would lose precision. "
                                     "Export a 16-bit sRGB copy without an embedded profile, "
                                     "or select 'Keep encoded values' only if its values are "
                                     "already appropriate for your workflow.")
                premultiplied = False
                if raw.format == "TIFF":
                    data, orientation, premultiplied = _tiff16(path)
                else:
                    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    orientation = raw.getexif().get(274, 1)
                    if data is not None and data.ndim == 3 and data.shape[-1] in (3, 4):
                        data = data[..., [2, 1, 0] if data.shape[-1] == 3 else [2, 1, 0, 3]]
                if data is None or data.dtype != np.uint16:
                    raise ValueError("The decoder could not preserve this image's 16-bit samples. "
                                     "Export a standard 16-bit RGB or grayscale PNG/TIFF.")
                if data.ndim == 2:
                    data = np.repeat(data[..., None], 3, axis=-1)
                elif data.shape[-1] not in (3, 4):
                    raise ValueError("Unsupported 16-bit channel layout; use RGB or grayscale.")
                data = _orient_array(data, orientation)
                rgb = data[..., :3].astype(np.float32) / 65535.0
                alpha = data[..., 3:4].astype(np.float32) / 65535.0 if data.shape[-1] == 4 else None
                if premultiplied and alpha is not None:
                    rgb = np.clip(np.divide(rgb, alpha, out=np.zeros_like(rgb), where=alpha > 0), 0, 1)
            else:
                oriented = stack.enter_context(closing(ImageOps.exif_transpose(raw)))
                has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
                rgba = stack.enter_context(closing(oriented.convert("RGBA"))) if has_alpha else None
                alpha = (np.array(rgba.getchannel("A"), dtype=np.float32)[..., None] / 255.0
                         if rgba is not None else None)
                # Preserve CMYK/LAB source channels for the ICC transform; RGBA/P
                # need an RGB image with the unassociated color channels retained.
                source = oriented if oriented.mode in ("RGB", "CMYK", "LAB", "L") else (
                    stack.enter_context(closing(oriented.convert("RGB"))))
                if profile and color_management == "Convert to sRGB":
                    source = stack.enter_context(closing(ImageCms.profileToProfile(
                        source, ImageCms.ImageCmsProfile(BytesIO(profile)),
                        ImageCms.createProfile("sRGB"), outputMode="RGB",
                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)))
                elif source.mode != "RGB":
                    source = stack.enter_context(closing(source.convert("RGB")))
                rgb = np.array(source, dtype=np.float32) / 255.0
            if alpha is not None:
                background = 1.0 if alpha_background == "White" else 0.0
                rgb = rgb * alpha + background * (1.0 - alpha)
            return torch.from_numpy(np.ascontiguousarray(rgb))[None, ...]
    except (OSError, ValueError, ImageCms.PyCMSError, Image.DecompressionBombError, cv2.error) as exc:
        raise ValueError(f"Photo Loader: cannot load '{path}': {exc}. "
                         "Batch position has not advanced. For an invalid ICC profile, "
                         "re-export the image or use 'Keep encoded values'.") from exc
