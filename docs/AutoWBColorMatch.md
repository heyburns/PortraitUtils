# AutoWBColorMatch

`AutoWBColorMatch` adjusts the white balance of an image so it lines up with a reference frame. Point it at the look you want—studio grey card, hero shot, graded still—and it nudges the target image into the same colour temperature.

---

## Inputs
- `image` – The photo you want to correct.
- `reference_image` – The frame whose white balance you trust. It can be a single still or any image tensor.
- `mode` – Pick the algorithm:
  - `Average` matches simple channel means; quick and gentle.
  - `Median` ignores outliers to handle mixed lighting.
  - `Shadows`, `Midtones`, `Highlights` lean on specific tonal ranges when you care about one band more than the others.
- `strength` – Blend amount between the original image (`0`) and full correction (`1`). Use fractional values for subtle shifts.
- `preserve_luma` – Keep existing brightness and adjust colour only; helpful when exposure already looks solid.
- `clip_extremes` – Clamp output values to prevent halos when the correction pushes colours too far.

---

## Outputs
- `image` – The white-balanced photo.
- `debug` – Text summary of the chosen mode and detected colour shift.

---

## Where It Fits

Use AutoWBColorMatch when stills arrive from multiple cameras or lighting setups and you need a quick neutral baseline before creative grading. It also works well before composite work to keep foreground and background plates in the same colour family.

---

## Tuning Tips

- Start with `Average` mode and `strength` at `0.8`; adjust up or down depending on how close the source already is.
- Switch to `Median` when the reference scene includes practical lights or coloured gels that could skew a simple average.
- Toggle `preserve_luma` if the match suddenly brightens or darkens the shot more than you expected.

---

## Troubleshooting

- **Output looks oversaturated** – Lower `strength` or enable `clip_extremes` to cap the adjustment.
- **Skin tones drift** – Try the `Midtones` mode so highlights or deep shadows don’t drive the calculation.
- **Nothing changes** – Confirm the reference image is connected and not identical to the source; the node skips correction when both inputs already match.

---

Screenshot: `docs/screenshots/auto_wb_color_match.png`
