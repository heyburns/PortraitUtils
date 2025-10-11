# AutoCropBorders

## Overview
`AutoCropBorders` trims uniform borders from an image by sampling edge pixels, estimating their colour statistics, and region-growing inward until the content changes. It optionally returns the detected border mask and reports the crop rectangle, making it useful for automated subject isolation or pre-processing scanned artwork.

## Inputs
- `image` (`IMAGE`): Tensor `[B, H, W, C]`. Values are clamped to `[0, 1]` and converted to float32.
- `fuzz_mode` (`STRING`, default `"adaptive"`): `"percent"` uses a fixed threshold derived from `fuzz_percent`; `"adaptive"` widens the tolerance based on border variance (`adaptive_k` multiplier).
- `fuzz_percent` (`FLOAT`, default `5.0`): Maximum per-channel deviation (in 0–100%) allowed when classifying border pixels in `"percent"` mode.
- `adaptive_k` (`FLOAT`, default `2.0`): Multiplier applied to the median absolute deviation to set per-channel thresholds in `"adaptive"` mode.
- `edge_margin_px` (`INT`, default `4`): Width of the seed strip sampled along each edge to build the border colour model.
- `pad_px` (`INT`, default `0`): Expands the resulting crop box outward by this many pixels (after detection).
- `use_luma_only` (`BOOLEAN`, default `False`): If `True`, detection runs on luma; otherwise it operates in RGB space.
- `max_growth_iter` (`INT`, default `4096`): Upper bound on the number of dilation iterations when growing the border region.
- `return_border_mask` (`BOOLEAN`, default `False`): When enabled, outputs the binary border mask aligned with the cropped image.
- `use_gpu` (`BOOLEAN`, default `True`): Moves temporary tensors to CUDA when available for faster region-growing.

## Outputs
- `image` (`IMAGE`): Cropped RGB tensor with even dimensions enforced.
- `left` (`INT`), `top` (`INT`), `width` (`INT`), `height` (`INT`): Crop rectangle relative to the original image (`width`/`height` coerced to even when possible).
- `border_mask` (`MASK`): Binary mask of removed borders, resized to the cropped output. Zero tensor when `return_border_mask` is `False`.

## Processing Notes
- The algorithm samples all four edges, computes medians and median absolute deviations, then grows a binary mask starting from the border strips.
- Detected borders are dilated and optionally padded, after which the surviving region is cropped and padded to even dimensions for latent-friendly downstream nodes.
- When `return_border_mask` is `False`, the mask socket is still supplied but contains zeros to simplify downstream wiring.

## Tips
- Raise `edge_margin_px` for thicker frames to capture more representative samples.
- Switch to `"percent"` mode with a low `fuzz_percent` when dealing with synthetic borders that have exact RGB values.
- Disable `use_gpu` on systems with limited VRAM; the logic gracefully continues on CPU.*** End Patch
