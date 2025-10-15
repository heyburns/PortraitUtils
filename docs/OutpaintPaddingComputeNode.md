# OutpaintPaddingComputeNode


## Overview
`OutpaintPaddingComputeNode` converts outpainting preferences (percentages or absolute pixels plus gravity) into concrete padding values for a given image. It ensures the final canvas dimensions remain even—useful for latent-based diffusion models—and supports both symmetric and directional growth.

## Inputs
- `image` (`IMAGE`): Reference image whose width/height determine the base canvas.
- `mode` (`STRING`, default `"Percent"`): Matches the setting emitted by `OutpaintConfigNode`. `"Pixels"` bypasses percentage math.
- `gravity` (`STRING`, default `"center"`): Indicates where additional space should be allocated when working in percent mode. Supports centre, edges, and corners.
- `horizontal_percent` (`FLOAT`, default `20.0`): Horizontal expansion percentage.
- `vertical_percent` (`FLOAT`, default `10.0`): Vertical expansion percentage.
- `left_px`, `right_px`, `top_px`, `bottom_px` (`INT`, defaults `0`): Absolute pixel paddings used when `mode` is `"Pixels"`.

## Outputs
- `left` (`INT`)
- `top` (`INT`)
- `right` (`INT`)
- `bottom` (`INT`)

## Processing Notes
- The node first reads the image dimensions from the BHWC tensor and converts all numeric inputs to safe ranges (`>= 0`).
- In `"Pixels"` mode the provided per-edge values are used directly, followed by even-size enforcement.
- In `"Percent"` mode the node computes the total horizontal/vertical padding and splits it according to `gravity` (e.g., bottom-heavy when gravity is `"bottom"`).
- Even-dimension enforcement increments the right and/or bottom padding as needed so `(width + left + right)` and `(height + top + bottom)` are divisible by 2.

## Tips
- Feed the outputs straight into padding-aware samplers or ControlNet preprocessing nodes.
- When chaining with `OutpaintConfigNode`, keep the gravity strings in sync to avoid unintended splits.*** End Patch
