# LoadImageCombined
![Screenshot](screenshots/load_image_combined.png)


## Overview
`LoadImageCombined` unifies single-image selection and batch iteration within one node. In single mode it behaves like ComfyUI’s standard loader; in batch mode it iterates through a directory listing on each execution, keeping track of the last-delivered file to maintain sequence order.

## Inputs
- `mode` (`STRING`, default `"Single"`): `"Single"` loads a user-selected file; `"Batch"` scans a directory and auto-advances each run.
- `input_dir` (`STRING`, default empty): Directory to scan in batch mode. Relative paths resolve against `ComfyUI/input`.
- `output_dir` (`STRING`, default empty): Convenience string passed through for downstream savers/loggers.
- `pattern` (`STRING`, default `"*"`): Glob pattern (e.g., `*.png`) applied in batch mode.
- `strip_trailing_numbers` (`BOOLEAN`, default `False`): When `True`, removes trailing `(1)`, `(2)`, etc., from filenames when computing the `filename_no_ext` output.
- `repeat_last` (`BOOLEAN`, default `False`): Keeps returning the last image instead of advancing on subsequent runs.
- `image` (`STRING`): File selector used in single mode. Only relevant when `mode == "Single"`.

## Outputs
- `IMAGE`: RGB tensor loaded from disk (alpha channels are dropped).
- `filename_no_ext` (`STRING`): Base filename without extension (and optionally without trailing number suffix).
- `output_dir` (`STRING`): Passthrough of the input field.
- `width` (`INT`), `height` (`INT`): Dimensions of the loaded image.

## Processing Notes
- Directory listings are case-insensitive and filtered to common image extensions (`png`, `jpg`, `jpeg`, `webp`, `tif`, `tiff`, `bmp`).
- Batch iteration state is held in-memory per directory/pattern combination and resets when the listing changes (new/removed files) or when `repeat_last` is enabled.
- All images are converted to RGB using Pillow, with Exif orientation applied.

## Tips
- Chain this node with `PortraitUtils_PairedImageLoader` or comparison nodes to quickly inspect before/after renders.
- Use `repeat_last` when you need to re-run the same batch frame after adjusting downstream parameters.
- Combine with `FilenameAppendSuffix` to build descriptive output paths based on `filename_no_ext`.*** End Patch
