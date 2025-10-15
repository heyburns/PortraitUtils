# Crop By Margins Suite

## Overview
`CropImageByMargins` and `CropMaskByMargins` apply identical margin logic to different tensor types. The image variant trims RGB tensors, while the mask variant normalises mask layouts before cropping. Use them together to keep images and masks aligned when removing edge padding or snapping to model-friendly dimensions.

## CropImageByMargins

### Inputs
- `image` (`IMAGE`): Tensor `[B, H, W, C]`; converted to float32 and clamped to `[0, 1]`.
- `left_px` (`INT`, default `0`): Pixels to remove from the left edge.
- `top_px` (`INT`, default `0`): Pixels to remove from the top edge.
- `right_px` (`INT`, default `0`): Pixels to remove from the right edge.
- `bottom_px` (`INT`, default `0`): Pixels to remove from the bottom edge.
- `snap_multiple` (`INT`, default `1`): Optional modulus applied after cropping. The box is shrunk (never expanded) so the width and height are multiples of this value; set to `1` to disable.

### Outputs
- `IMAGE`: Cropped RGB tensor with the same batch size as the input.

### Processing Notes
- Margins are clamped so the crop remains within bounds and preserves at least a 1×1 region.
- Snapping runs after margins are applied and preserves the top-left anchor.
- Alpha channels are dropped by the shared helpers, ensuring consistent RGB output.

## CropMaskByMargins

### Inputs
- `mask` (`MASK`): Accepts `[H, W]`, `[H, W, 1]`, `[1, H, W]`, or `[B, H, W, 1]`; normalised to `[B, H, W, 1]` float32 internally.
- `left_px` (`INT`, default `0`), `top_px` (`INT`, default `0`), `right_px` (`INT`, default `0`), `bottom_px` (`INT`, default `0`): Identical semantics to the image variant.
- `snap_multiple` (`INT`, default `1`): Applies the same post-crop snapping logic to keep masks aligned with images.

### Outputs
- `MASK`: Cropped mask tensor with the same batch size as the normalised input.

### Processing Notes
- Unsupported mask layouts raise descriptive `ValueError`s, prompting conversion before use.
- Margin clamping mirrors the image node, keeping the crop in bounds and at least 1×1.
- Snapping shares the same helper as the image variant, guaranteeing matched dimensions when both nodes receive identical parameters.

## Tips
- Drive both nodes from a shared set of sliders/inputs to guarantee image/mask alignment.
- Set `snap_multiple` to `8`, `16`, or `64` when preparing content for models that operate on fixed block sizes.
- When chaining with other cropping utilities, apply `CropMaskByMargins` to masks first so subsequent mask-aware steps inherit the aligned dimensions.
