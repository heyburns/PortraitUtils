# FluxResolutionPrepare

`FluxResolutionPrepare` crops and resizes an image to the closest resolution supported by Flux-family models. It can also pre-upscale small sources so you always hit a minimum megapixel target before the final resize.

---

## Inputs
- `image` – The photo to process. Works with a single image tensor.
- `min_megapixels` – Minimum size threshold when pre-upscaling is allowed. If the current image falls below this, the node scales it up before anything else.
- `enable_pre_upscale` – Toggle the pre-upscale step on or off.
- `crop_width`, `crop_height` – Optional manual crop dimensions. Set to values greater than zero to lock the crop size.
- `crop_x`, `crop_y` – Offsets for the manual crop. Measured in pixels from the original image before upscale.

---

## Outputs
- `image` – The resized image that fits a Flux-legal resolution.
- `ratio` – Human-readable aspect ratio like `3:4` or `9:21`.
- `target_width`, `target_height` – Final output dimensions.
- `area_loss_percent` – Percentage of the working area trimmed to hit the target ratio.
- `pre_scale_factor` – Amount of pre-upscaling applied (1.0 if no upscale happened).

---

## Where It Fits

Run this node before sending photos into Flux-based diffusion models or any workflow that expects exact width/height pairs. It removes the guesswork when preparing portrait batches for model-friendly input sizes.

---

## Tuning Tips

- Keep `enable_pre_upscale` on when starting with small headshots; it preserves detail before the final resize.
- Use manual crop dimensions only when you must enforce a custom region—otherwise let the automatic crop centre the subject.
- Watch `area_loss_percent`; high values mean you’re trimming a lot of the image to meet the target ratio, so consider adjusting the source crop first.

---

## Troubleshooting

- **Output looks soft** – Check if the pre-upscale factor is very high; consider disabling it and doing a manual upscale earlier in the pipeline.
- **Wrong area cropped** – Confirm manual crop values are valid and within image bounds. Clearing them (`-1` defaults) restores automatic centering.
- **Aspect ratio unexpected** – Review the candidate list in the node’s widget; the closest match wins. Adjust the list if you need specific pairs.

---

Screenshot: `docs/screenshots/flux_resolution_prepare.png`
