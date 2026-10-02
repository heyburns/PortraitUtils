# Load Image + Source Info

The production [Load Image + Source Info](PhotoLoaderPreview.md) node
(`PortraitPhotoLoaderPreview`) loads one photo per execution and emits compact
source information. The `LoadImageCombined` and `LoadImageCombinedV2` IDs
have been removed. See the linked guide for its color-management controls.

## Inputs

- `mode`: Single uses the uploaded/selected `image`; Batch reads a folder.
- `input_dir`: Required in Batch mode. Relative paths resolve under ComfyUI/input.
- `output_dir`: Passed into source information for later use by the saver.
- `pattern`: Batch filename glob, such as `*.png`; defaults to `*`.
- `strip_trailing_numbers`: Remove a trailing `(n)` from the output filename stem.
- `repeat_last`: In Batch mode, repeat the last selected file instead of advancing.
- `image`: ComfyUI image selection/upload used in Single mode.

Batch mode filters supported image extensions, sorts filenames, and loops after
the final file. It loads one image, not a multi-image tensor batch. Batch position
is in-process state; it is not persisted when ComfyUI restarts.

## Outputs

- `image`: RGB float image tensor.
- `source_info`: `PORTRAIT_SOURCE_INFO_V2`, containing filename without extension,
  output directory, and original width/height.

Connect `source_info` to **Save Info from Source** beside Simple Image Saver.
The reader exposes filename and output directory. Alpha masks are not emitted;
use a separate mask loader/generator when needed.

See [Typed Workflow Configuration](ConfigurationV2.md) for the source-info bus.
