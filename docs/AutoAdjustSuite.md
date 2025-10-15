# AutoAdjust Suite

## Overview
`AutoAdjustNode` performs the heavy lifting for global levels, tone, and colour balancing, while `AutoColorConfigNode` keeps the corresponding toggles in sync across multiple branches. Use the config node to drive one or more adjustment nodes so you can change a single widget panel and broadcast the desired combination of levels/tone/colour/flip options to every branch that needs them.

## AutoAdjustNode

### Inputs
- `image` (`IMAGE`): Source tensor shaped `[B, H, W, C]` (C must be 3 or 4; alpha is dropped).
- `precision` (`STRING`): `"Histogram (fast)"` uses 256-bin histograms for percentile estimates; `"Exact"` calls `torch.quantile`.
- `auto_levels` (`BOOLEAN`, default `True`): Enables global min/max remapping based on luma percentiles.
- `levels_shadow_clip_pct` (`FLOAT`, default `0.1`): Percentage of darkest pixels to clip for the low anchor.
- `levels_highlight_clip_pct` (`FLOAT`, default `0.1`): Percentage of brightest pixels to clip for the high anchor.
- `levels_gamma_normalize` (`BOOLEAN`, default `False`): Pulls midtones toward 50% grey when the median luma sits between 35% and 65%.
- `auto_tone` (`BOOLEAN`, default `True`): Toggles the tone-mapping stage.
- `tone_mode` (`STRING`, default `"Per-channel"`): `"Per-channel"` stretches each channel independently; `"Monochromatic"` uses a luma-derived curve.
- `tone_shadow_clip_pct` (`FLOAT`, default `0.1`): Shadow percentile for the tone stage.
- `tone_highlight_clip_pct` (`FLOAT`, default `0.1`): Highlight percentile for the tone stage.
- `auto_color` (`BOOLEAN`, default `True`): Recentres the Cb/Cr components in YCbCr space.
- `snap_neutral_midtones` (`BOOLEAN`, default `False`): Restricts the colour-balancing statistics to pixels with luma in `[0.2, 0.8]`.
- `flip_horizontal` (`BOOLEAN`, default `False`): Horizontally mirrors the result as a final step.

### Outputs
- `IMAGE`: RGB tensor with the enabled stages applied in order (levels → tone → colour → optional flip).

### Processing Notes
- All operations run in float32 `[0, 1]`; the node coerces inputs and clamps outputs accordingly.
- Levels and tone stages guard against degenerate ranges (identical min/max) and fall back to minimal stretching.
- Colour balancing works in YCbCr; `snap_neutral_midtones` switches from mean to masked median stats to avoid highlight/shadow bias.
- Any supplied alpha channel is intentionally removed so downstream nodes receive consistent RGB tensors.

## AutoColorConfigNode

### Inputs
- `auto_levels` (`BOOLEAN`, default `False`): Desired state for the levels stage.
- `auto_tone` (`BOOLEAN`, default `False`): Desired state for the tone stage.
- `auto_color` (`BOOLEAN`, default `False`): Desired state for the colour balance stage.
- `flip_horizontal` (`BOOLEAN`, default `False`): Indicates whether downstream nodes should flip horizontally.

### Outputs
- `auto_levels` (`BOOLEAN`)
- `auto_tone` (`BOOLEAN`)
- `auto_color` (`BOOLEAN`)
- `flip_horizontal` (`BOOLEAN`)

### Usage Notes
- Wire the outputs directly into matching sockets on one or more `AutoAdjustNode` instances to keep their toggles synchronised.
- Because outputs are plain booleans, they work well with conditional routers or scripting nodes that expect simple data types.
- The node stores no internal state; it simply forwards the widget values, making it safe to duplicate anywhere you need the same preset.

## Tips
- `"Exact"` precision is typically faster for batch workloads despite the name; switch back to `"Histogram (fast)"` only if you see quantile-related performance issues.
- Set both clip percentages to zero whenever you must preserve the full tonal range (for HDR hand-offs or forensic workflows).
- Use `AutoColorConfigNode` to maintain a master switch panel that feeds parallel branches (e.g., differing crops or upscale streams) so every branch stays in lockstep without duplicating toggle widgets.
