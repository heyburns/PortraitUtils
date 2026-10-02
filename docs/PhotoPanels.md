# Detect & Split Photo Panels

Node ID: `PortraitPhotoPanelSplitV2`.

This node takes **one composite IMAGE**, detects rectangular photographic panels,
and extracts the original source pixels. It does not identify people, generate
masks, deskew prints, pad images, or resize them to editor buckets. Each panel
keeps its own width, height, dtype, alpha, and device.

## Selection and wiring

Use the existing [Image Chooser Classic](https://github.com/heyburns/image-chooser-classic)
for selection. Its installed implementation accepts a list of differently sized
images, so no padded/resized batch or latent conversion is needed.

```text
Loader IMAGE
    → Detect & Split Photo Panels.image
         panels → Image Chooser Classic.images
                      images → AutoCrop → subject/protected masks → Photo Prepare
```

Select **one thumbnail**, with the chooser's `count` set to **1**, to pass one
ordinary IMAGE to the rest of a single-photo workflow. **Only pause if batch**
automatically proceeds for ordinary single photos and asks when multiple panels
are found. **Always pause** asks every time. The chooser retains its existing
selection UI, hotkeys, cancellation, and repeat modes; its code is not modified.

The new splitter has only two outputs:

- `panels`: an IMAGE **list**, not a same-size tensor batch. Wire this directly
  into the chooser's `images` input. Connecting it directly to an ordinary
  processing node will generally process every detected panel in turn.
- `analysis`: JSON text containing panel count, source coordinates/dimensions,
  divider evidence, and warnings. Panel `number` is one-based for thumbnail
  labels; `index` is zero-based for the chooser's selection string.

The splitter itself has no chooser dependency and works with other list-aware
selectors. It adds no JavaScript, web routes, model dependency, or latent work.
Place it before mask generation so masks are made for the selected photo's grid.
Keep editor/upscaler routing unchanged.

## Detection controls

- **Gutters only** (default): look for nearly uniform divider strips with
  corroborated image boundaries on both sides. Supports black, white, gray,
  and colored gutters. Detection recurses inside panels to handle horizontal,
  vertical, grid, and guillotine-style mixed layouts with unequal panel sizes.
- **Gutters + strong seams**: additionally consider abrupt, well-supported
  straight boundaries between touching panels. This is opt-in because a scene
  boundary can resemble a collage seam. Always verify the thumbnails.
- **Collage (partial seams)**: opt-in detection for touching panels with thin,
  interrupted dividers. It requires a locally prominent seam supported near both
  ends of its parent region, then recurses so each column can have different row
  cuts. Strong scene edges can still be mistaken for panel borders.
- **Manual splits**: require at least one explicit split position below.
- **Sensitivity**: Conservative requires stronger divider evidence; Sensitive
  accepts more variation and weaker boundaries. Balanced is the default.
- **min_panel_area_percent**: minimum automatic panel area as a fraction of the
  whole input. Default 3%. Lower it for small thumbnails; manual splits bypass it.
- **max_panels**: default 16, maximum 64. Automatic detection keeps remaining
  areas unsplit at the limit rather than discarding them. Excessive manual
  splits produce a helpful error rather than truncating the output.

Only analysis samples transfer to CPU. Every position along a candidate divider
axis is examined; sampling is perpendicular to that axis, preserving narrow,
even one-pixel, gutters. Final crops are literal slices of the input tensor.
Analysis uses float RGB (RGBA composited on white only for detection); actual
source channels, alpha, and precision remain unchanged.

This is a divider detector, **not semantic scene segmentation**. Large blank
gaps, text-heavy layouts, non-rectangular/rotated/overlapping prints, and subtle
borderless collages may require manual splits. An ordinary single photograph
is retained whole when there is no supported divider. Uniform empty grid cells
may be omitted after automatic detection. Outer print borders are not removed;
use AutoCrop or the scanned-photo tools afterward as appropriate. Divider scores
are heuristic evidence values, not calibrated probabilities.

## Manual correction

Any non-empty `vertical_splits`, `horizontal_splits`, or
`column_horizontal_splits` **replaces automatic detection** with explicit
source-coordinate splits, regardless of the selected detection mode.

- `vertical_splits = 38.9%`: a vertical line at approximately 38.9% of source width.
- `vertical_splits = 357:364`: omit source columns 357 through 363 as a gutter.
- `vertical_splits = 33%, 66%`: create three columns.
- Add `horizontal_splits = 50%` to split every column into two equal rows.
- For staggered rows, use `vertical_splits = 150,430` and
  `column_horizontal_splits = 258,520 | 205,581 | 258,520`. Each `|` section
  supplies y cuts for one column, left to right; an empty section leaves that
  column unsplit. Leave `horizontal_splits` empty in this mode.

Single positions retain every source pixel. A `start:end` range explicitly
omits that gutter; endpoints are half-open. Pixel positions are whole numbers;
percentages can be fractional. Output panels are ordered top-to-bottom, then
left-to-right. Leave all three fields empty to return to automatic detection.

## Verification

CPU regressions cover unequal panels, colored/noisy/JPEG gutters, one-pixel
dividers, grids, interrupted nine-panel collages with staggered rows, per-column
manual corrections, outer-border preservation, single-photo fallbacks, count
limits, precision/alpha preservation, and the installed chooser's actual
single-selection tensor bundling method. Browser interaction and real-world
detection quality still need workflow testing.
