# Crop By Margins Suite

`CropImageByMargins` and `CropMaskByMargins` trim uniform margins from RGB images and their matching masks. Use them together to keep crops aligned across colour and alpha channels.

---

## CropImageByMargins

### Inputs
- `image` – The RGB image to crop.
- `left`, `right`, `top`, `bottom` – Margin sizes in pixels. Positive numbers trim inward; negative values add padding.
- `enforce_even` – Force even output dimensions for latent-friendly pipelines.
- `fill_color` – RGB colour used when padding outward (defaults to black).

### Output
- `image` – Cropped or padded image.

---

## CropMaskByMargins

### Inputs
- `mask` – The matching mask tensor.
- `left`, `right`, `top`, `bottom` – Use the same values as the image node for perfect alignment.
- `enforce_even` – Keep mask dimensions even to match the image branch.

### Output
- `mask` – Cropped or padded mask.

---

## Where It Fits

Use the suite when you need consistent framing across multiple outputs—portrait plus silhouette, subject plus matte, etc. It also helps when you want to clear unwanted edges while keeping a predictable amount of breathing room on each side.

---

## Tuning Tips

- Mirror the same margin values across both nodes so the subject stays aligned.
- Set `enforce_even` to `True` when downstream models require even resolutions; leave it off for pixel-perfect edits.
- Choose a neutral `fill_color` (e.g., mid-grey) when you plan to blend the padded area later.

---

## Troubleshooting

- **Mask no longer lines up** – Double-check that both nodes share identical margin values and ordering.
- **Unexpected border colour** – Adjust `fill_color` when padding outward; the default is pure black.
- **Output still has odd dimensions** – Confirm `enforce_even` is enabled if your workflow expects even sizes.

---

Screenshot: `docs/screenshots/crop_image_by_margins.png`
