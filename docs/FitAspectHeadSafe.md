# FitAspectHeadSafe

## Overview
`FitAspectHeadSafe` chooses an aspect ratio from a user-supplied list and computes a crop rectangle that keeps a detected subject centred with controllable headroom, footroom, and side margins. It is designed to work with the bounding boxes emitted by `MQBBoxMin`, maintaining portrait-friendly framing while matching downstream aspect targets.

## Inputs
- `image` (`IMAGE`): Reference image for dimension lookup only.
- `x`, `y`, `w`, `h` (`INT`): Subject bounding box (e.g., from `MQBBoxMin`). Coordinates are expressed in pixels relative to the original image.
- `aspects_csv` (`STRING`, default `"2:3,3:4,1:1,9:16,16:9,5:8,8:5"`): Comma-separated aspect ratios (width:height). Both portrait and landscape ratios may be provided.
- `match_to` (`STRING`, default `"mq_box"`): Determines the target ratio. `"mq_box"` compares the candidate list to the subject box, `"image"` matches the full image ratio.
- `headroom_ratio` (`FLOAT`, default `0.12`): Fraction of the candidate height reserved above the subject when expanding the crop.
- `footroom_ratio` (`FLOAT`, default `0.06`): Fraction of height reserved below the subject.
- `side_margin_ratio` (`FLOAT`, default `0.08`): Lateral padding as a fraction of the candidate width.
- `bottom_priority` (`FLOAT`, default `0.75`): Bias when reconciling head/foot violations. Values closer to `1` favour the bottom margin (useful for keeping feet in frame).
- `horiz_gravity` (`STRING`, default `"center"`): Horizontal anchoring for the final crop. `"left"` and `"right"` pin the rectangle to the image edges; `"center"` balances around the subject.

## Outputs
- `w` (`INT`), `h` (`INT`): Width and height of the recommended crop.
- `x` (`INT`), `y` (`INT`): Top-left coordinate of the crop within the original image.
- `aspect_used` (`STRING`): The ratio selected from `aspects_csv`.
- `debug` (`STRING`): Trace information describing the decision process (chosen ratio, applied margins, violation metrics).

## Processing Notes
- The node parses the aspect list, finds the closest ratio to either the image or subject box, then expands the subject rectangle with head/foot/side margins.
- `_cover_min_rect` ensures the expanded rectangle is the smallest that matches the chosen ratio while fully covering the desired region.
- Vertical placement is refined by evaluating candidate positions and selecting the one with minimum weighted violation, using `bottom_priority` to bias the result.
- The final rectangle is clamped to the image boundaries and rounded to integers.

## Tips
- Pair with `MQBBoxMin` to generate the `x`, `y`, `w`, `h` inputs automatically from a mask.
- Supply both portrait and landscape ratios in `aspects_csv` when you intend to reuse the node for multi-orientation workflows.
- Inspect the `debug` string when tuning headroom/footroom; it lists the applied margin sizes and any constraint violations.*** End Patch
