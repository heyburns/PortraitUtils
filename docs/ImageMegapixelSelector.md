# ImageMegapixelSelector

`ImageMegapixelSelector` resizes images to hit a target megapixel budget. It’s a quick way to keep batches inside memory limits or align images with downstream models that expect a specific pixel count.

---

## Inputs
- `image` – The photo to resize.
- `target_megapixels` – Desired image size measured in megapixels (e.g., `1.5` equals 1.5 million pixels).
- `clamp_min_megapixels` – Prevent downsizing below this value; useful when you only want to shrink oversize images.
- `round_dimensions` – Choose how the node rounds width and height:
  - `Even` keeps both dimensions even.
  - `Multiple of 64` suits latent models that require divisible sizes.
  - `None` keeps the exact float result.
- `keep_aspect` – When `True`, scale both sides evenly; when `False`, allow separate width/height scaling to hit the target precisely.

---

## Outputs
- `image` – Resized image tensor.
- `width`, `height` – New dimensions in pixels.
- `scale_factor` – Multiplier applied to the original size.

---

## Where It Fits

Use the node before memory-heavy diffusion or upscaling steps when you need to guarantee each frame stays under a certain size. It’s also handy for standardising mixed-source reference sets so later processing behaves consistently.

---

## Tuning Tips

- Enable `keep_aspect` to avoid distortion; only disable it if you plan to crop afterward.
- Choose `Multiple of 64` rounding for Stable Diffusion checkpoints, or `Even` for general-purpose pipelines.
- Set `clamp_min_megapixels` to the same value as `target_megapixels` when you only ever want to downscale oversized inputs.

---

## Troubleshooting

- **Image still too big** – Lower the target megapixels or check that `keep_aspect` isn’t preventing the exact resize you need.
- **Image looks stretched** – Make sure `keep_aspect` is enabled unless non-uniform scaling is intentional.
- **Model rejects dimensions** – Adjust the rounding mode to satisfy the model’s width/height requirements.

---

Screenshot: `docs/screenshots/image_megapixel_selector.png`
