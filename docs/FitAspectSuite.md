# FitAspect Suite

## Overview
`MQBBoxMin` finds a reliable subject bounding box from a mask, and `FitAspectHeadSafe` expands that box into a camera-friendly crop that honours headroom, footroom, and target aspects. Use them together when you need to translate noisy mask data into consistent portrait framing for downstream workflows.

## MQBBoxMin

### Inputs
- `mask` (`MASK`): Input tensor in any supported mask layout; internally converted to a 2D float array.
- `invert_mask` (`STRING`, default `"auto"`): Determines which side of the mask is treated as foreground. `"auto"` picks the side with less border mass, `"false"` leaves the mask untouched, `"true"` flips it.
- `q_left`, `q_right`, `q_top`, `q_bottom` (`FLOAT`, defaults `0.005` / `0.995`): Quantiles used to trim outliers along each axis and focus on the subject.
- `min_span_px` (`INT`, default `8`): Enforces a minimum width/height after quantile trimming to prevent degenerate boxes.
- `tight_crop` (`BOOLEAN`, default `False`): When `False`, returns the full image bounds. When `True`, resolves a tight subject box with safety margins and matches legal aspect ratios.

### Outputs
- `x`, `y`, `w`, `h` (`INT`): Pixel-space bounding box coordinates relative to the mask.
- `debug` (`STRING`): Notes on foreground selection, quantile bounds, padding, and any aspect adjustments performed.

### Processing Notes
- Masks are normalised to `[0, 1]` with automatic dtype conversion when required.
- Axis-wise cumulative sums yield stable quantile bounds even for noisy masks or partial coverage.
- The routine inflates the quantile box by proportional padding before searching for a minimal-area rectangle that matches the image aspect ratio; common portrait ratios (1:1, 2:3, 3:4, 9:16, etc.) are evaluated as fallbacks.
- If no aspect candidate can contain the subject, the raw quantile rectangle is returned.

### Tips
- Feed the resulting `x`, `y`, `w`, `h` directly into `FitAspectHeadSafe` for refined framing.
- Increase `min_span_px` when working with very thin masks (e.g., profile subjects) to stabilise the output.
- Prefer `"true"` for `invert_mask` when you know the mask stores background as white; it avoids mis-detection in border-heavy images.

## FitAspectHeadSafe

### Inputs
- `image` (`IMAGE`): Reference image used to read base dimensions.
- `x`, `y`, `w`, `h` (`INT`): Subject bounding box (typically from `MQBBoxMin`). Coordinates are absolute pixels relative to the original image.
- `aspects_csv` (`STRING`, default `"2:3,3:4,1:1,9:16,16:9,5:8,8:5"`): Comma-separated list of target aspect ratios (`width:height`), supporting portrait and landscape mixes.
- `match_to` (`STRING`, default `"mq_box"`): `"mq_box"` compares the candidate list to the subject box; `"image"` matches the full image aspect.
- `headroom_ratio` (`FLOAT`, default `0.12`): Fraction of the candidate height reserved above the subject when expanding the crop.
- `footroom_ratio` (`FLOAT`, default `0.06`): Fraction reserved below the subject.
- `side_margin_ratio` (`FLOAT`, default `0.08`): Lateral padding as a fraction of the candidate width.
- `bottom_priority` (`FLOAT`, default `0.75`): Weighting applied when reconciling head/foot constraint violations; higher values favour keeping the lower margin (useful when feet should remain in frame).
- `horiz_gravity` (`STRING`, default `"center"`): Horizontal anchoring for the final crop (`"left"`, `"center"`, `"right"`).

### Outputs
- `w` (`INT`), `h` (`INT`): Width and height of the computed crop rectangle.
- `x` (`INT`), `y` (`INT`): Top-left coordinate of the recommended crop within the original image.
- `aspect_used` (`STRING`): Chosen ratio from `aspects_csv`.
- `debug` (`STRING`): Trace information describing ratio selection, margin application, and constraint violations.

### Processing Notes
- Candidate aspect ratios are parsed, compared against the chosen `match_to` basis, and the closest ratio is selected.
- The subject rectangle is expanded with the requested margins; `_cover_min_rect` finds the smallest rectangle matching the target aspect that still covers the desired region.
- Vertical placement is scored using weighted violations so the crop respects head/foot priorities before clamping to image bounds.
- All outputs are rounded to integers; the crop never extends beyond the source image.

### Tips
- Supply both portrait and landscape ratios when a single workflow needs to handle varying orientations.
- Inspect the `debug` string while tuning headroom/footroom values; it reports the exact padding applied and any adjustments made during clamping.
- When subjects include footwear or lower-body details, raise `bottom_priority` to bias the crop toward preserving footroom.
