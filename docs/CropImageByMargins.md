# CropImageByMargins


## Overview
`CropImageByMargins` removes a specified number of pixels from each edge of an RGB image. It supports optional snapping of the resulting dimensions to a chosen multiple, which is useful when downstream models require even or block-aligned shapes.

## Inputs
- `image` (`IMAGE`): Tensor `[B, H, W, C]`. The node internally converts data to float32 and clamps to `[0, 1]`.
- `left_px` (`INT`, default `0`): Pixels to remove from the left edge.
- `top_px` (`INT`, default `0`): Pixels to remove from the top edge.
- `right_px` (`INT`, default `0`): Pixels to remove from the right edge.
- `bottom_px` (`INT`, default `0`): Pixels to remove from the bottom edge.
- `snap_multiple` (`INT`, default `1`): Optional modulus applied to the resulting width and height. The crop box is shrunk (never expanded) so that width/height are multiples of this value. Set to `1` to disable snapping.

## Outputs
- `IMAGE`: Cropped tensor with the same batch size as the input.

## Processing Notes
- Margins are clamped so the crop never goes outside the image bounds; at least one pixel is preserved in both axes.
- Snapping is performed after margins are applied, preserving the upper-left origin.
- Alpha channels are implicitly discarded by the helper utilities shared across the suite; the output is RGB.

## Tips
- Use a `snap_multiple` of `64` when preparing tiles for SDXL-style upscalers or patch-based inference.
- Pair this node with `CropMaskByMargins` to keep images and masks aligned.*** End Patch
