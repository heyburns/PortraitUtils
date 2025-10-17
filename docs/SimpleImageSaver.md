# SimpleImageSaver

## Overview
`SimpleImageSaver` is a lightweight output node for writing image batches to disk without the extra automation layers in the built-in ComfyUI saver. It supports PNG or JPG output, controllable JPEG quality, workflow metadata toggles, and per-save path customization including suffix-aware filenames.

## Inputs
- `images` (`IMAGE`): Batch of images to write. If the input is empty, the node exits quietly.
- `output_path` (`STRING`, default empty): Absolute path or subfolder inside `ComfyUI/output` where files are placed. Missing folders are created automatically.
- `filename` (`STRING`, default `ComfyUI`): Base name applied to every image in the batch.
- `suffix` (`STRING`, default empty): Optional text appended after the base name using a dash separator.
- `format` (`PNG` / `JPG`, default `PNG`): Output format. PNG retains lossless data; JPG writes with the selected quality.
- `jpeg_quality` (`INT`, default `95`): JPEG quality from 0–100. Values ≥95 use no chroma subsampling.
- `include_metadata` (`BOOLEAN`, default `True`): When enabled, embeds the workflow prompt and `EXTRA_PNGINFO`. JPEGs store this JSON in the Exif user comment; PNGs use text chunks.

## Outputs
- _(none)_ – This is an output node. The UI response includes a list of saved files for the ComfyUI frontend.

## Processing Notes
- Paths are normalized against `ComfyUI/output`, and attempts to traverse outside of that tree are rejected.
- File components are sanitized to remove characters illegal on Windows/macOS/Linux.
- Batched inputs receive indexed suffixes (`-0000`, `-0001`, …) to maintain unique filenames.
- Metadata is skipped entirely when `include_metadata` is `False`, matching ImageMagick’s `-strip` behaviour.
- JPEG metadata payloads above ~64 KB are ignored, since Exif comments cannot exceed that limit.

## Tips
- Combine with `FilenameAppendSuffix` or other naming helpers to reuse upstream file stems.
- Use the metadata toggle to create public-safe variants without embedding workflow JSON.
- Chain several `SimpleImageSaver` instances in the same workflow to generate intermediate checkpoints without cluttering the main output directory.
