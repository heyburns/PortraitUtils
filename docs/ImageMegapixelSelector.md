# ImageMegapixelSelector


## Overview
`ImageMegapixelSelector` rescales an RGB image to the closest target in {1, 2, 3} megapixels (rounded with ties favouring larger). The resized output can enforce divisibility constraints on width and height, making it handy for models that require dimensions aligned to a specific multiple (e.g., 8 or 64).

## Inputs
- `image` (`IMAGE`): Source tensor `[B, H, W, C]`. Batching is not yet supported (`B` must be 1). Any alpha channel is discarded prior to processing.
- `divisible_by` (`INT`, default `8`): Ensures both width and height in the result are multiples of this value. Set to `1` to disable snapping.

## Outputs
- `image` (`IMAGE`): RGB tensor resized via Lanczos filtering to the chosen megapixel target, respecting the divisibility constraint.
- `target_megapixels` (`FLOAT`): The actual megapixel value achieved after rounding and modulus corrections.

## Processing Notes
- The node computes the raw scale necessary to hit the nearest whole-megapixel target, then searches a neighbourhood of candidate widths that satisfy `divisible_by`.
- Heights are rounded by preserving aspect ratio where possible; several rounding strategies are tried to reduce distortion.
- Resampling uses PIL’s Lanczos filter on the CPU, preserving RGB ordering and clamping to `[0, 1]`.

## Tips
- Combine this node with `FluxResolutionPrepare` when you want predictable intermediate resolutions prior to Flux cropping.
- Leave `divisible_by` at 8 for Stable Diffusion-style latents, or increase to 64 when feeding grids/tilers that operate on 64-pixel blocks.
- Because the node always outputs RGB, downstream mask compositors can assume three-channel data.*** End Patch
