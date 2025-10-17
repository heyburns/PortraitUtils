# StitchByMask

`StitchByMask` blends a foreground and background image using a mask with optional feathering and opacity control. It’s the quick way to composite retouched faces, outfit swaps, or background replacements inside ComfyUI.

---

## Inputs
- `foreground` – Image that should appear wherever the mask is white.
- `background` – Image that fills in when the mask is black. Use the same resolution as the foreground.
- `mask` – Greyscale mask controlling the blend. White reveals the foreground, black reveals the background.
- `invert_mask` – Flip mask interpretation without editing the mask itself.
- `feather_px` – Apply additional smoothing to the mask edge in pixels.
- `opacity` – Blend strength from 0.0 (only background) to 1.0 (full foreground).
- `preserve_metadata` – Pass through metadata from the foreground when `True`; otherwise copy from the background.

---

## Outputs
- `image` – Composited result.
- `mask` – The mask after any feathering or inversion (useful for debugging or saving).

---

## Where It Fits

Use StitchByMask when weaving together portrait fixes: slipping a cleaned-up face back into the original plate, combining outfit variations, or dropping a new background behind a subject. It also pairs with outpaint padding to smooth transitions.

---

## Tuning Tips

- Keep `opacity` just under 1.0 (e.g., 0.95) to allow subtle bleed from the background and avoid hard seams.
- Apply feathering sparingly; you can always stack another blur if the edge still feels sharp.
- Invert the mask via `invert_mask` rather than regenerating the mask upstream—fewer steps, same result.

---

## Troubleshooting

- **Harsh seam visible** – Increase `feather_px` slightly or ensure the mask resolution matches the images exactly.
- **Blend looks washed out** – Check `opacity`; if it’s below 1.0, the background will bleed through more strongly.
- **Wrong area reveals** – Flip `invert_mask` or verify the mask isn’t inverted before it reaches the node.

---

Screenshot: `docs/screenshots/stitch_by_mask.png`
