# Load Image + Source Info

Node ID: `PortraitPhotoLoaderPreview`. Category: `PortraitUtils/IO`.

This production loader replaces `LoadImageCombinedV2`, which is no longer
registered. It has independent transactional batch cursors and content-based
change detection. It loads photos; it does not crop, straighten, resize, adjust
levels, or remove borders. The v5.3 workflow already uses it.

## Wiring

Restart ComfyUI and add **Load Image + Source Info**. Its two outputs are:

- `image` goes where the old loader's image went, including Intelligent AutoCrop.
- `source_info` goes to the existing source-info Set node or **Save Info from
  Source** reader. It is the same `PORTRAIT_SOURCE_INFO_V2` type, not a new bus.

Mask generation and editor/upscaler routing do not change.

## Controls

The first seven controls correspond to the current loader:

- `mode`: **Single** loads the uploaded/selected image. **Batch** loads one photo
  from a folder per queued execution, not a multi-image tensor batch.
- `input_dir`: Required in Batch mode. An absolute path, a `~` path, or a path
  relative to ComfyUI/input. Ignored in Single mode.
- `output_dir`: Passed unchanged apart from surrounding whitespace to source
  info. Blank leaves output-directory selection to the saver. Loading does not
  create folders or write any images.
- `pattern`: Batch glob; blank means `*`. `**/*.png` includes subfolders.
  Only PNG, JPEG, WebP, TIFF, and BMP files are eligible. Sorting is deterministic
  and natural: `photo2` comes before `photo10`.
- `strip_trailing_numbers`: Removes a final copy marker such as `photo (2)`.
  It does not strip normal digits such as `scan2024`.
- `repeat_last`: Holds the last successfully loaded batch photo. With no previous
  photo, loads the first one. Missing/deleted previous files restart at the first
  available file. A held unchanged photo remains cacheable downstream.
- `image`: Selection/upload for Single mode. It is ignored in Batch mode, even
  if that selection is missing or ComfyUI/input has no uploaded images.

Two additional controls make decoding explicit:

- `color_management`: **Convert to sRGB** (default) honors embedded 8-bit ICC
  profiles. Untagged RGB photos are assumed sRGB. **Keep encoded values** skips
  ICC conversion, useful for comparison with the old loader. Non-RGB sources
  still need conversion to RGB; untagged CMYK conversion is approximate.
- `alpha_background`: **White** (default) or **Black** for transparency.
  Transparency is composited, not discarded. No alpha mask is emitted.

## Reliability and image fidelity

Each node instance keeps its own cursor for each folder/pattern, with up to 32
recent listings retained. Two loaders pointed at the same folder do not advance
each other's positions. A failed decode never commits a new cursor position.
Repair or replace the offending file to retry; it is not silently skipped.
Adding files does not reset a cursor whose last file still exists.

State lasts only while ComfyUI retains the node object. Restarting the server,
clearing its object cache, or recreating the node starts from the first file.
ComfyUI's no-cache mode does not retain node objects, so folder advancement
requires a normal caching mode. Identical preview nodes remain separate in the
output cache because their hidden node IDs are included in ComfyUI's cache keys.

The image output is CPU float32 RGB, `[1, height, width, 3]`. EXIF orientation is
applied before recording source dimensions. No spatial resampling occurs.
Single and held-batch images use streamed content hashes, so even replacements
with the same filename, size, and timestamp invalidate cached results.

Standard 16-bit grayscale/RGB/RGBA PNG/TIFF decoding preserves 16-bit samples.
TIFF uses `tifffile` (already installed in this development environment and now
listed in requirements) to avoid decoder-specific orientation problems. Common
compressed files without orientation/premultiplied-alpha transforms can use the
existing OpenCV decoder as a lossless fallback. Other compressed TIFF variants
may additionally need `imagecodecs`; a missing codec is reported with install or
export guidance rather than silently losing precision.
Embedded ICC conversion for these files is deliberately refused because the
available Pillow ICC converter operates at 8-bit precision. Export a 16-bit sRGB
copy without an embedded profile, or deliberately select **Keep encoded values**
if the encoded colors are appropriate. 32-bit/HDR files fail with export guidance
instead of being silently clipped or normalized. Animated/multipage inputs load
only their first frame/page; this is a still-photo loader.

The preview does not remove Alamy or other image-hosting borders. Those remain
the responsibility of the following **Intelligent AutoCrop** stage.
