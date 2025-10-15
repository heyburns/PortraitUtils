# FluxResolutionPrepare


## Overview
`FluxResolutionPrepare` crops and resizes an image to the closest legal resolution supported by Flux-family models. It optionally pre-upscales low-resolution inputs to a minimum megapixel budget, respects an optional user-supplied crop window, and reports the chosen aspect ratio along with crop efficiency metrics.

## Inputs
- `image` (`IMAGE`): Source tensor `[1, H, W, C]`. Batched inputs are not supported.
- `min_megapixels` (`FLOAT`, default `0.95`): Minimum megapixel target used when `enable_pre_upscale` is `True`. The node upsamples the image if the current pixel count is below this threshold.
- `enable_pre_upscale` (`BOOLEAN`, default `True`): Toggles the pre-upscale pass. When disabled, the original resolution is used for cropping.
- `crop_width` (`INT`, optional, default `-1`): Width of a user-specified region of interest. Values `<= 0` disable the manual crop.
- `crop_height` (`INT`, optional, default `-1`): Height of the manual crop ROI.
- `crop_x` (`INT`, optional, default `0`): Left offset of the manual crop (pixels, before pre-upscale).
- `crop_y` (`INT`, optional, default `0`): Top offset of the manual crop (pixels, before pre-upscale).

## Outputs
- `image` (`IMAGE`): Processed tensor sized to a Flux-compatible resolution.
- `ratio` (`STRING`): Human-readable aspect label (e.g., `3:4`, `9:21`).
- `target_width` (`INT`): Final width selected from the built-in Flux whitelist.
- `target_height` (`INT`): Final height.
- `area_loss_percent` (`FLOAT`): Percentage of the (pre-upscaled) working area discarded during cropping.
- `pre_scale_factor` (`FLOAT`): Scale multiplier applied by the pre-upscale stage (1.0 when disabled or unnecessary).

## Processing Notes
- The node first pre-upscales by the minimum factor required to hit `min_megapixels`, using bicubic interpolation.
- If a manual crop is specified, it is interpreted in original-image coordinates and mapped to the pre-upscaled space.
- The best Flux resolution is chosen by minimizing scale deviation, area loss, and ratio error across a fixed candidate list (landscape/portrait pairs included).
- Final cropping is centred unless constrained by the boundaries of the manual crop; resizing uses bicubic interpolation.

## Tips
- Feed the emitted `ratio` into UI labels/logs so you can audit whether the chosen aspect aligns with your prompt or downstream model.
- When passing manual crop dimensions, ensure they leave enough room for the target aspect ratio; otherwise, the node will fall back to the nearest fit that clamps to image bounds.
- Pair this node with `MQBBoxMin` and `FitAspectHeadSafe` to derive a subject-aware crop before committing to Flux sizing.*** End Patch
