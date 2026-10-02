# Paired Image Loader

Production node ID: `PortraitPairedLoaderPreview`. The retired
`PairedImageLoader` ID is no longer registered. The production loader retries
failed pairs without advancing and uses per-instance cursors.
[Details](ReplacementPreviews.md#paired-image-loader).

## Actual inputs and outputs

Inputs are `source_dir`, `output_dir`, `reverse`, and
`strip_trailing_numbers`. Directories can be absolute or relative to the process
working directory. Matching is case-insensitive by basename. Different
extensions can match; natural sorting puts image2 before image10. A trailing
`(n)` marker is ignored for matching only when requested.

Outputs, in order: **output_image**, **source_image**, **filename** (the actual
source stem without its extension). Each execution advances one pair and wraps;
reverse steps backward. There is no pattern template, shuffle, singleton output,
auto-next/loop switch, or index socket; previous documentation claiming those
features was inaccurate.

The cursor commits only after both files decode. Decoding uses the
color/precision-aware Photo Loader engine.
