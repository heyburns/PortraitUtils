# StripBottomBanner

`StripBottomBanner` trims scan-style metadata slugs or broadcast tickers that cling to the bottom edge of portrait photos. It looks for an extended, very dark band and removes it while reporting how many rows were cut.

---

## Inputs
- `image` – RGB tensor to inspect; batch-friendly.
- `pixel_dark_threshold` – Luma cutoff that marks a pixel as “dark”. Raise it for very soft grey banners, lower it for deep black tickers.
- `dark_fraction_threshold` – Minimum proportion of dark pixels required in a row before it is considered banner material.
- `bright_pixel_threshold` / `bright_fraction_threshold` – Define what counts as a “bright” pixel and how many must appear to confirm the strip was truly a banner with overlaid text or graphics.
- `min_band_percent` – Minimum vertical footprint (as percent of image height) that the candidate band has to cover before removal.
- `max_scan_percent` – How far up from the bottom to inspect. Reduce it if you only expect very short banners.
- `extra_trim_px` – Additional rows removed after the detected band, useful for shaving residual glow or aliasing.
- `require_bright_rows` – When enabled, demands at least one bright row inside the dark run. Disable to remove pure-black bars with no overlays.

---

## Outputs
- `image` – Cropped tensor with the banner removed (batch preserved).
- `trimmed_rows` – Number of rows that were shaved off the bottom.
- `detected` – Boolean flag indicating a banner was found and cropped. If the detector rejected the band (e.g., it was inconsistent between batch items) the original image is passed through and this flag is `False`.

---

## Usage Notes
- Prefer feeding single images or batches where every item shares the same banner height. The node aborts if detections disagree within the batch.
- Tune `max_scan_percent` so the search region only covers the likely banner area; that keeps large base images from producing partial matches.
- If you work with monochrome archives, disable `require_bright_rows` so the darker strip is still eligible even without UI highlights.

---

## Troubleshooting
- **Banner remains** – Increase `pixel_dark_threshold` or relax `dark_fraction_threshold` so noisy borders still count as “dark enough”. Raising `max_scan_percent` also allows taller banners to be considered.
- **Too aggressive** – Lower `max_scan_percent` or raise `min_band_percent` to keep the detector from consuming legitimate image content.
- **Mixed batch failure** – The node currently requires identical detections per batch entry. Split the batch or run twice with filtered groups when banners vary in height.
