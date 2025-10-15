# Outpaint Suite

## Overview
`OutpaintConfigNode` captures user preferences for how much canvas to add and where, while `OutpaintPaddingComputeNode` converts those preferences into concrete per-edge padding suitable for diffusion workflows. Use them together to keep UI controls simple while still feeding precise values into downstream samplers, ControlNets, or compositing nodes.

## OutpaintConfigNode

### Inputs
- `mode` (`STRING`, default `"Percent"`): When `"Percent"`, horizontal/vertical settings are interpreted as percentages of the current canvas; `"Pixels"` switches to absolute per-side values.
- `gravity` (`STRING`, default `"center"`): Direction that should receive the majority of the expansion. Supports centre, single edges, and corner presets.
- `horizontal_percent` (`FLOAT`, default `20.0`): Total horizontal expansion percentage used in percent mode.
- `vertical_percent` (`FLOAT`, default `10.0`): Total vertical expansion percentage used in percent mode.
- `left_px`, `right_px`, `top_px`, `bottom_px` (`INT`, defaults `0`): Absolute pixel padding for each side, used when `mode` is `"Pixels"`.

### Outputs
- `mode` (`STRING`)
- `gravity` (`STRING`)
- `horizontal_percent` (`FLOAT`)
- `vertical_percent` (`FLOAT`)
- `left_px` (`INT`)
- `right_px` (`INT`)
- `top_px` (`INT`)
- `bottom_px` (`INT`)

### Usage Notes
- All outputs are emitted regardless of mode, so downstream nodes can decide which values to read.
- Percentages and pixel values are clamped to be non-negative before emission.
- Pair directly with `OutpaintPaddingComputeNode` to translate these preferences into edge-specific padding.

## OutpaintPaddingComputeNode

### Inputs
- `image` (`IMAGE`): Reference tensor used to read the current width/height.
- `mode` (`STRING`, default `"Percent"`): Mirrors the setting from `OutpaintConfigNode`. `"Pixels"` bypasses percentage calculations.
- `gravity` (`STRING`, default `"center"`): Guides how percent-based padding is split between sides (e.g., `"bottom"` pushes most of the vertical growth downward).
- `horizontal_percent` (`FLOAT`, default `20.0`): Total horizontal growth percentage.
- `vertical_percent` (`FLOAT`, default `10.0`): Total vertical growth percentage.
- `left_px`, `right_px`, `top_px`, `bottom_px` (`INT`, defaults `0`): Absolute per-edge padding used when `mode` is `"Pixels"`.

### Outputs
- `left` (`INT`)
- `top` (`INT`)
- `right` (`INT`)
- `bottom` (`INT`)

### Processing Notes
- Image dimensions are derived from the BHWC tensor; all numeric inputs are coerced to safe (non-negative) ranges.
- In `"Pixels"` mode, the given per-edge values are used directly, followed by even-size enforcement to keep `(width + left + right)` and `(height + top + bottom)` divisible by two.
- In `"Percent"` mode, total padding is computed from the percentages and split according to `gravity`; fractional splits are rounded so the final dimensions remain even.
- Gravity presets include centre, top/bottom, left/right, and diagonal corners (e.g., `"top-right"` allocates vertical padding upward and horizontal padding to the right).

### Tips
- When wiring from `OutpaintConfigNode`, keep the `gravity` dropdowns synchronised to avoid unexpected splits.
- Feed the computed `left/top/right/bottom` values into padding-aware samplers or ControlNet preprocessors to avoid manual math.
- Use `"Pixels"` mode for workflows that already know the exact size to add (e.g., tile-based upscalers), and switch to `"Percent"` when you want relative growth that adapts to varying canvas sizes.
