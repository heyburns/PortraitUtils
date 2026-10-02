# Auto White-Balance + Color Match

Production node ID: `PortraitWhiteBalancePreview`. The removed
`AutoWBColorMatch` ID is not available. See the current
[white-balance guide](WhiteBalancePreview.md) for its additional control.

## Actual interface

The node has two required image connections, `image` and `reference`, and one
`IMAGE` output. It does **not** have a debug output, median/midtone/shadow modes,
or `preserve_luma` / `clip_extremes` controls; previous documentation describing
those controls was inaccurate.

Its widgets are:

- `method`: `wb_grayworld`, `wb_highlight`, `reinhard_lab`, `lab_l_only`, or
  `wb_highlight+reinhard` (default).
- `percentile`: highlight selection, default 95.
- `strength`: sRGB blend amount, default 1.
- `clip_gamut`: final clamp, default true.
- `force_size`: resize only the statistics sample, default false.
- `target_width` / `target_height`: statistics-sample bounds, defaults 1440×1080.

WB-only methods calculate from the source rather than matching the reference.
Reinhard and lightness-only methods use reference statistics. The combined mode
applies highlight WB followed by reference matching.

## Corrected behavior

Highlight gains are per channel, the D65 Lab conversion uses correctly oriented
matrices, and corrections are applied at full source resolution. The production node adds a `preserve_luminance` control, enabled by default.
