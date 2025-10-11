# OutpaintConfigNode
![Screenshot](screenshots/outpaint_config_node.png)


## Overview
`OutpaintConfigNode` collects user preferences for outpainting operations—gravity, measurement mode, and per-edge padding. Its outputs are raw values that can feed into padding calculators or custom scripts without additional parsing.

## Inputs
- `mode` (`STRING`, default `"Percent"`): Determines whether outpainting requests are expressed as percentages of the current canvas or absolute pixels (`"Pixels"`).
- `gravity` (`STRING`, default `"center"`): Direction that should receive the majority of the expansion. Supports centre, single-direction, and corner presets.
- `horizontal_percent` (`FLOAT`, default `20.0`): Horizontal expansion percentage used when `mode` is `"Percent"`.
- `vertical_percent` (`FLOAT`, default `10.0`): Vertical expansion percentage used when `mode` is `"Percent"`.
- `left_px`, `right_px`, `top_px`, `bottom_px` (`INT`, defaults `0`): Explicit pixel padding for each side (used when `mode` is `"Pixels"`).

## Outputs
- `mode` (`STRING`)
- `gravity` (`STRING`)
- `horizontal_percent` (`FLOAT`)
- `vertical_percent` (`FLOAT`)
- `left_px` (`INT`)
- `right_px` (`INT`)
- `top_px` (`INT`)
- `bottom_px` (`INT`)

## Usage Notes
- All outputs are emitted regardless of mode, allowing downstream nodes to choose the correct interpretation.
- Percentages and pixel values are clamped to sensible minima (`>= 0`) before emission.
- Combine with `OutpaintPaddingComputeNode` to convert settings into absolute padding values for a specific image.*** End Patch
