# AutoCropBorders

`AutoCropBorders` trims uniform borders from photos—think scanner edges, letterbox bars, or projector frames—before the image moves deeper into your workflow.

---

## Inputs
- `image` – The photo you want to clean; connect any RGB image tensor.
- `fuzz_mode` – Choose how tolerant the detector should be:
  - `adaptive` widens the tolerance automatically based on border noise.
  - `percent` sticks to the value from `fuzz_percent`.
- `fuzz_percent` – Maximum colour variation allowed when `percent` mode is active. Lower values assume a perfectly uniform border.
- `adaptive_k` – Multiplier used in `adaptive` mode; higher values accept noisier borders.
- `edge_margin_px` – Width of the strip sampled on each side to learn the border colour. Increase for thicker frames.
- `pad_px` – Expand the final crop box by this many pixels if you want a little breathing room.
- `use_luma_only` – When `True`, the decision runs on brightness only instead of full RGB. Handy for black-and-white content.
- `max_growth_iter` – Safety cap on how many expansion steps the border mask can take.
- `return_border_mask` – Toggle to output the detected border mask alongside the cropped image.
- `use_gpu` – When `True`, temporary tensors jump to CUDA. Disable on tight VRAM budgets.

---

## Outputs
- `image` – The cropped photo with even width and height where possible.
- `left`, `top`, `width`, `height` – Crop rectangle relative to the original image.
- `border_mask` – Binary mask of removed borders (zeroed out when the mask output is disabled).

---

## Where It Fits

Use AutoCropBorders on scanned headshots or production stills before resizing or running diffusion. It quickly removes black bars from footage exports so downstream nodes don’t waste cycles on empty pixels.

---

## Tuning Tips

- Switch to `percent` mode with a low `fuzz_percent` when dealing with perfectly solid borders (e.g., exact RGB black).
- Bump `edge_margin_px` for thicker frames so the sampler grabs enough context.
- Disable `use_gpu` if you are processing long batches on a shared card; the CPU path is slower but reliable.

---

## Troubleshooting

- **Border remains** – Increase `fuzz_percent` (in percent mode) or `adaptive_k` (in adaptive mode) so minor noise doesn’t block detection.
- **Image cropped too tightly** – Raise `pad_px` or reduce the fuzz tolerance so fewer pixels are marked as border.
- **Unexpected slivers left behind** – Check that `max_growth_iter` isn’t too low for high-resolution images; raising it lets the mask reach the true content edge.

---

Screenshot: `docs/screenshots/auto_crop_borders.png`
