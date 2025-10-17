# FitAspect Suite

`MQBBoxMin` finds a reliable subject box inside a mask, and `FitAspectHeadSafe` grows that box into a crop that respects your target aspect ratios with controllable head and foot room. Together they turn imperfect masks into camera-friendly framing.

---

## MQBBoxMin

### Inputs
- `mask` – Binary or soft mask of your subject. Any layout supported by ComfyUI works.
- `invert_mask` – Tells the node which side of the mask is foreground: `auto` guesses, `true` flips, `false` keeps as-is.
- `q_left`, `q_right`, `q_top`, `q_bottom` – Percentile cuts that trim stray pixels. Raise them if the mask includes clutter.
- `min_span_px` – Minimum width/height allowed after trimming to avoid zero-sized boxes.
- `tight_crop` – When `True`, emit the trimmed subject box; when `False`, fall back to the full image bounds.

### Outputs
- `x`, `y`, `w`, `h` – Bounding box coordinates in pixels.
- `debug` – Text summary of the mask decision and adjustments made.

---

## FitAspectHeadSafe

### Inputs
- `image` – The original image used to read dimensions.
- `x`, `y`, `w`, `h` – Subject box from `MQBBoxMin` or any custom bounding box.
- `aspects_csv` – Comma-separated aspect ratios like `2:3,3:4,1:1`. Order them by priority.
- `match_to` – Compare candidate ratios to the subject box (`mq_box`) or the full image (`image`).
- `headroom_ratio` – Portion of the crop reserved above the subject.
- `footroom_ratio` – Portion reserved below the subject.
- `side_margin_ratio` – Horizontal padding on each side.
- `bottom_priority` – Weight that favours keeping the lower margin when space is tight (1.0 = full priority to the bottom).
- `horiz_gravity` – Anchor the crop left, centre, or right when extra width remains.

### Outputs
- `w`, `h`, `x`, `y` – Final crop dimensions and position.
- `aspect_used` – Ratio that won the selection.
- `debug` – Notes about ratio choice, padding, and clamping.

---

## Where It Fits

Use the suite when you receive portrait masks from detectors or rotoscopers and need predictable headroom before feeding images into Flux, upscalers, or design layouts. It’s also handy for social deliverables that demand specific aspect ratios (1:1, 4:5, 9:16) without chopping off heads or feet.

---

## Tuning Tips

- Increase `min_span_px` in `MQBBoxMin` for skinny subjects so the box doesn’t collapse.
- List your preferred aspects first in `aspects_csv`; the node tries them in order of closeness.
- Adjust `headroom_ratio` and `footroom_ratio` to match the style you’re after—more headroom for fashion, more footroom for full-body shots.
- Use the `debug` outputs while tuning; they spell out why a particular aspect or offset was chosen.

---

## Troubleshooting

- **Subject gets clipped** – Raise padding ratios or reduce `bottom_priority` so the crop floats higher.
- **Wrong aspect selected** – Ensure your desired ratio is included in `aspects_csv` and check the debug log to see why another ratio won.
- **Output exceeds image bounds** – The node clamps to the original dimensions; if you need extra space, pad the image before running the suite.

---

Screenshot: `docs/screenshots/fit_aspect_head_safe.png`
