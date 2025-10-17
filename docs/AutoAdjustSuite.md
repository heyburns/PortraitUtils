# AutoAdjust Suite

`AutoAdjustNode` gives photos a quick “auto levels / auto tone / auto color” pass similar to Photoshop. `AutoColorConfigNode` is the helper dial that lets you toggle those same options across several branches at once.

---

## AutoAdjustNode

### Inputs
- `image` – The picture you want to clean up; feed it from any loader or crop.
- `precision` – Choose `Histogram (fast)` for everyday batches or `Exact` when you notice uneven results and want a pixel-perfect read.
- `auto_levels` – Stretch shadows and highlights so the full brightness range is used.
- `levels_shadow_clip_pct` – Percentage of darkest pixels to shave off while stretching. Lower = gentler.
- `levels_highlight_clip_pct` – Same idea for the brightest pixels.
- `levels_gamma_normalize` – Gently nudge midtones toward middle grey so portraits don’t look flat.
- `auto_tone` – Adds a second balancing pass that adjusts each colour channel separately.
- `tone_mode` – `Per-channel` boosts each colour independently for punchier contrast; `Monochromatic` keeps colour relationships intact.
- `tone_shadow_clip_pct` / `tone_highlight_clip_pct` – How aggressive that second pass should be in shadows and highlights.
- `auto_color` – Remove colour casts (too cool, too warm) automatically.
- `snap_neutral_midtones` – Focus the colour fix on midtones so highlights don’t overpower the adjustment.
- `flip_horizontal` – Mirror the image left/right without adding another node.

### Output
- `image` – The adjusted photo, clamped to `[0, 1]` and ready for your next node.

---

## AutoColorConfigNode

Use this helper when you want multiple `AutoAdjustNode` blocks to respond to the same switches.

### Inputs / Outputs
- `auto_levels`
- `auto_tone`
- `auto_color`
- `flip_horizontal`

Each input is a simple checkbox, and each output mirrors that value. Wire the outputs straight into matching inputs on any `AutoAdjustNode` instances you want to keep in lockstep.

---

## Where It Fits

Drop AutoAdjust at the top of a portrait pipeline when you need a neutral base grade before creative work. It also shines when batch-cleaning camera JPGs with mixed lighting or prepping reference frames for diffusion and upscaling jobs.

---

## Tuning Tips

- Lower the clip percentages when you must protect detail in highlights or shadows.
- Switch to `Exact` precision for small batches, archival scans, or whenever histogram estimates wobble.
- Enable `snap_neutral_midtones` for tungsten or mixed indoor lighting to keep skin tones believable.
- Use the flip toggle during QA passes to spot composition issues or prep mirrored training data.

---

## Troubleshooting

- **Image looks flat** – Try disabling `auto_tone` or reduce the tone clip percentages.
- **Colours drift too far** – Turn off `auto_color`, or enable `snap_neutral_midtones` for a gentler fix.
- **No visible change** – Confirm `auto_levels` is on and both clip percentages aren’t set to zero.

---

See `docs/screenshots/auto_adjust_node.png` for the widget layout inside ComfyUI.
