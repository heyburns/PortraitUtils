# AutoAdjustNode
![Screenshot](screenshots/auto_adjust_node.png)


## Overview
`AutoAdjustNode` performs a chained series of global tone corrections on an RGB image. It can optionally apply automatic levels, per-channel or monochrome tone stretching, neutral colour balancing in YCbCr, and a final horizontal flip. The implementation assumes floating-point tensors in `[0, 1]` and always returns RGB output.

## Inputs
- `image` (`IMAGE`): Source image tensor shaped `[B, H, W, C]` (C must be 3 or 4; the alpha channel is discarded).
- `precision` (`STRING`): Selects percentile estimation mode for level/tone calculations. `"Histogram (fast)"` approximates quantiles via 256-bin histograms, while `"Exact"` uses `torch.quantile`.
- `auto_levels` (`BOOLEAN`, default `True`): Enables global min/max remapping based on luma percentiles.
- `levels_shadow_clip_pct` (`FLOAT`, default `0.1`): Percentage of darkest pixels to clip when computing the low anchor for levels.
- `levels_highlight_clip_pct` (`FLOAT`, default `0.1`): Percentage of brightest pixels to clip when computing the high anchor.
- `levels_gamma_normalize` (`BOOLEAN`, default `False`): When enabled, nudges midtones toward 50% grey if the median luma falls between 35% and 65%.
- `auto_tone` (`BOOLEAN`, default `True`): Toggles the tone mapping stage.
- `tone_mode` (`STRING`, default `"Per-channel"`): `"Per-channel"` stretches each channel independently; `"Monochromatic"` applies a single stretch derived from luma.
- `tone_shadow_clip_pct` (`FLOAT`, default `0.1`): Shadow percentile used by the tone stage.
- `tone_highlight_clip_pct` (`FLOAT`, default `0.1`): Highlight percentile used by the tone stage.
- `auto_color` (`BOOLEAN`, default `True`): Enables neutral balance in YCbCr space (centres Cb/Cr components).
- `snap_neutral_midtones` (`BOOLEAN`, default `False`): When `True`, only midtone pixels (luma in 0.2–0.8) contribute to the Cb/Cr statistics, reducing highlight/shadow bias.
- `flip_horizontal` (`BOOLEAN`, default `False`): Horizontally mirrors the result.

## Outputs
- `IMAGE`: The processed RGB tensor with all three stages (levels, tone, colour) applied in order, plus any optional flip.

## Processing Notes
- All stages operate on float32 tensors in `[0, 1]`; the node casts incoming data accordingly and clamps the final output.
- Levels and tone stages guard against degenerate ranges (when high equals low) and fall back to minimal stretching.
- The colour balance stage works in YCbCr and uses either masked medians or simple means depending on `snap_neutral_midtones`.
- Alpha channels are intentionally dropped so downstream nodes always receive RGB tensors.

## Tips
- Theoretically, use `"Histogram (fast)"` when performance matters or when feeding large batches; switch to `"Exact"` for reproducibility-sensitive work. In actual practice, `"Exact"` is usually faster, so there's no reason not to use it.
- Keep `levels_shadow_clip_pct`/`levels_highlight_clip_pct` at zero when you need strict dynamic range preservation (e.g., HDR preparation).
- Combine this node with `AutoColorConfigNode` to share toggle states across multiple branches of a workflow.*** End Patch
