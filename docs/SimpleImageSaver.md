# Simple Image Saver

Production node ID: `PortraitImageSaverPreview`. The old
`SimpleImageSaver` ID is removed. This node uses the atomic 2.0 export engine.
[Details](ReplacementPreviews.md#simple-image-saver).

## Actual inputs

- `images`: IMAGE tensor to save.
- `output_path`: Destination directory; blank uses ComfyUI/output. Relative
  paths are under that directory; absolute paths are supported.
- `filename`: Base filename without extension.
- `suffix`: Optional text appended with a dash.
- `file_format`: PNG or JPG.
- `jpeg_quality`: 0–100, default 95; ignored for PNG.
- `include_metadata`: Include workflow prompt/extras when global metadata is
  enabled. JPEG metadata is a comment, not EXIF; large comments are skipped.
- `unique_filenames`: Append a counter rather than overwrite an existing name.

This is an output node with **no graph output sockets**. It supplies ComfyUI's
saved-image UI preview, not a filepath output or an image pass-through. Missing
destination directories are created. Batch filenames include an index.

The node defaults to 8-bit output, offers optional 16-bit PNG and an explicit
JPEG alpha-background control, and publishes only complete encoded files.
