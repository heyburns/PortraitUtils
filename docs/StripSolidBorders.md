# StripSolidBorders

`StripSolidBorders` repeatedly trims thick, almost-flat colour frames that surround archive scans and scrapbook imports. The node keeps shaving edges while they stay uniform, making it ideal for cleaning large batches of legacy material.

---

## Inputs
- `image` – Single RGB tensor; batching is not yet supported.
- `max_border_percent` – Hard stop for how much of the width/height can be removed on any edge. Lower it to protect content. Expressed as percent of the current dimension.
- `variance_threshold` – Maximum per-edge colour variance allowed before the algorithm stops stripping. Drop it when borders are extremely flat; raise it for noisier frames.
- `colour_delta_threshold` – Largest per-channel deviation (0–1 range) compared to the running edge mean. Increase when borders include gentle gradients.
- `min_band_px` – Minimum consecutive pixels per pass that must qualify as border before they are removed.
- `passes` – Number of full sweeps. Extra passes help chew through multi-stage mats or alternating light/dark stripes.
- `mean_follow` – Smoothing factor for the running colour estimate. Higher values let the baseline colour drift slowly as the strip progresses inward.
- `min_uniform_fraction` – Required portion of pixels in each sampled row/column that fall within the threshold. Relax this for grainy scans.
- `dominant_bins` – Histogram granularity (per channel) used for the fallback dominance test. Raising the count tolerates more subtle variations.
- `edge_retry_allowance` – Number of “outlier” rows/columns allowed before the pass halts. Increase when a few bright scratches interrupt the border.

---

## Outputs
- `image` – Cropped tensor with the surrounding frame removed.
- `trim_left`, `trim_top`, `trim_right`, `trim_bottom` – Total pixel counts removed from each edge.
- `detected` – `True` when at least one edge was stripped; otherwise the input image is returned unchanged and counts are zero.

---

## Usage Notes
- Feed the node individual images or pre-sliced batches—the implementation currently expects `B=1`.
- Start with the defaults. If borders remain on one side, increase `passes` or relax both `variance_threshold` and `colour_delta_threshold`.
- Because edges are evaluated independently, you can deliberately stack it after `StripBottomBanner` to remove bottom tickers first and then clean the remaining sides.

---

## Troubleshooting
- **Nothing trimmed** – Either the edge noise exceeded `variance_threshold`, or the border exceeded `max_border_percent`. Try raising both thresholds or lowering `min_uniform_fraction`.
- **Content lost** – Reduce `passes` or tighten `max_border_percent` so the crop stops earlier. Inspect the returned trim counts to see which edge over-trimmed.
- **Alternating stripes survive** – Borders with multiple colours may need a higher `edge_retry_allowance` and `mean_follow` so the detector can adapt between stripes.
