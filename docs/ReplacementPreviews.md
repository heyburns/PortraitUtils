# Production replacements for retired IDs

The former replacement previews are now the production nodes. Their historical
Python IDs still contain `Preview` from the testing period, but ComfyUI
displays the proper names below. Superseded IDs are no longer registered.
No workflow JSON is bundled in this repository. The historical local v5.3
workflow uses the production replacements but still needs current editor target
labels and gate socket wiring; v5.2 additionally uses removed node IDs.
See the [current editor wiring](SmartPhotoPrepare.md).

## Migrating a saved workflow

Restart the ComfyUI server and refresh the browser. Reconnect matching sockets
and copy old widget settings when replacing a node manually; changing a node
type does not migrate its widget values automatically. Keep a copy of the old
workflow for comparison.

| Existing node | Replacement display name | Replacement ID |
| --- | --- | --- |
| Stitch by Mask (`StitchByMaskV2`) | Stitch by Mask | `PortraitStitchPreview` |
| Simple Image Saver | Simple Image Saver | `PortraitImageSaverPreview` |
| Process Scanned Photo | Process Scanned Photo | `PortraitScannedPhotoPreview` |
| Paired Image Loader | Paired Image Loader | `PortraitPairedLoaderPreview` |
| Crop by Margins (Image) | Crop by Margins (Image) | `PortraitCropImageMarginsPreview` |
| Crop by Margins (Mask) | Crop by Margins (Mask) | `PortraitCropMaskMarginsPreview` |

Existing required/optional/hidden socket names, existing widget order/types,
defaults, and output types/order are mostly retained. The saver and paired
loader add optional widgets. Reconnect old graphs by socket name and copy the
intended settings; similarly named controls need not have identical behavior.

## Stitch by Mask

Same inputs as `StitchByMaskV2`: `image_a`, `image_b`, the existing
`PORTRAIT_EDIT_CONFIG_V2` bundle, `opacity_source`, inversion, bypass,
feather radius, and optional force-size dimensions; optional `mask`.
Same outputs: `stitched` IMAGE, `processed_mask` MASK.

Changes:

- Feathering operates on coverage first. Configured opacity is applied **once,
  last**, so the effective mask cannot exceed the selected blend/stitch opacity.
  Earlier versions could turn a 75% blend into nearly 100% after feathering.
- Feathering is an outward Gaussian halo, bounded by `feather_radius`, with
  original soft coverage retained. There is no binary threshold or disk-dilation
  pass. Retune the radius if comparing
  with results saved by earlier versions.
- Constant masks retain their coverage at image edges. CPU feathering uses
  OpenCV's efficient separable blur; non-CPU tensors use separable PyTorch
  filtering on their current device. No shaders/models are involved.
- `processed_mask` is standard ComfyUI **BHW**.
  HW and BHW1 inputs are accepted too; any 3D mask is interpreted as BHW,
  including one-pixel-wide masks. Wrap HWC1 as BHW1 explicitly.
- The source A batch is authoritative. B and mask may each have one member or
  match A's count. A single member broadcasts; other counts are rejected.
- RGBA is retained and blended using premultiplied color weights, preventing
  transparent hidden RGB from tinting visible pixels. RGB/grayscale can be
  promoted when paired with an RGBA image. Alpha-aware image resizing avoids
  hidden-color bleed. RGB-only blends still use the legacy sRGB interpolation.
- Source dimensions, dtype, and device are retained unless force-size is on.
  Half inputs are processed in float32; double inputs remain double. Images and
  masks must be finite, normalized 0–1. Mismatches produce actionable errors.

With force-size on, output actually changes to the target dimensions; this is
intentional compositing alignment, not the analysis-only option in white balance.
Images use antialiased bilinear resizing; masks use nearest-neighbor resizing
before feathering. With force-size off, all spatial dimensions must match.

## Simple Image Saver

Same existing controls: `images`, `output_path`, `filename`, `suffix`,
`file_format` (PNG/JPG), `jpeg_quality`, `include_metadata`, and
`unique_filenames`. This remains an output node with **no graph output sockets**;
its result is the usual ComfyUI saved-image UI preview.

New optional controls:

- `png_bit_depth`: **8-bit** (default, same output precision as before) or
  **16-bit**, preserving more of the working float image's precision. PNG16
  supports grayscale, RGB, and RGBA, including workflow metadata. It cannot
  restore detail already lost earlier in the pipeline.
- `jpeg_alpha_background`: **White** (default) or **Black**. PNG retains alpha;
  JPEG composites RGBA onto this matte rather than silently discarding alpha.

Changes:

- Encode a temporary file in the destination directory and publish only a
  complete, synced image. Encoding/publication failures clean up the temporary
  file. Existing destination files remain intact when encoding fails.
- Unique mode atomically publishes through a no-overwrite hard link and retries
  numbered collisions. This requires a destination filesystem supporting hard
  links. Failure is explicit; it never silently switches to overwriting.
- `unique_filenames=false` is an explicit overwrite choice and uses atomic
  replacement after successful encoding. Batch filenames keep their index.
- Relative output paths stay under ComfyUI/output, including symlink checks.
  Explicit absolute paths remain supported. Necessary directories are created.
- Filename/suffix components cannot create subdirectories, contain control
  characters, or use reserved device names. Excessively long names are rejected.
- Grayscale works; float16/bfloat16 and other float inputs are safely converted
  for encoding. NaN/infinite inputs are rejected. Finite out-of-range pixels
  are clamped for display-format export, as in the old saver.
- PNG workflow text supports Unicode. Non-finite numbers in workflow metadata
  are written as JSON `null` with a warning; the live workflow is not changed.
  Other invalid workflow JSON produces an error before image writes. Disabling
  metadata bypasses it; the global ComfyUI setting is respected. Oversized JPEG
  comments are omitted with a warning; use PNG for a large workflow.
- External-directory UI previews copy already encoded bytes, preserving JPEG
  quality and PNG16 precision instead of re-encoding. If creating this preview
  fails after a successful save, the error reports the real saved path.

Atomicity applies to each file, not an entire batch. Successfully published
earlier batch files remain if a later file fails. The saver does not infer or
convert a tensor's color profile; use a color-managed loader for your working
photos. It does not copy original EXIF or invent a source ICC profile.

PNG16 uses [OpenCV's documented unsigned-16-bit PNG encoder](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html),
with workflow text written as standard [PNG text chunks](https://www.w3.org/TR/png-3/).
JPEG workflow comments use [Pillow's JPEG comment support](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg).

## Process Scanned Photo

The original five controls remain: `image`, `straighten`, `crop_mode`, `padding`,
and `threshold`. One IMAGE output. This is still a narrow scanned-print utility,
not a subject cropper or perspective correction node.

- **Scanner Bed Only** detects a dominant rectangular print against a bright
  scanner background and crops around its outer physical boundary. A distinct
  print border is retained, including a Polaroid's thicker lower border.
- **Inner Photo Frame** starts from that print boundary, then seeks a broadly
  supported rectangular image/paper seam to remove the actual print frame too.
  It can fall back to conservative uniform-border detection for bright photos
  that do not qualify for the threshold-based inner detector.
- **None** skips cropping; straightening can still operate if enabled.

Straightening uses the dominant rectangular print, not the convex hull of all
dark marks on the scanner bed. Isolated dust does not extend that rectangle.
Confidence checks require bright matching corner background, a substantial
rectangular region, and room around the print. Uncertain angle/boundary estimates
are not forced. Outer detection uses contrast with the scanner bed rather than
assuming the print's own white paper is the same thing as the scanner background.

`threshold` is the minimum scanner-background brightness for outer detection
and the bright-paper/darker-image divider for the inner threshold detector.
It also influences the fallback uniform-border tolerance. It is no longer a
single blind threshold for every stage. Adjust it to your scanner/print.

`padding` is applied once around the final crop: positive retains extra context,
negative crops further inward. Negative padding that removes the detected image
raises an explanatory error. Scanner Bed Only can conservatively retain a thin
scanner-bed fringe to avoid clipping physical print edges.

Analysis is bounded to a 2048-pixel longest side, but **the actual pixels are
never converted to 8-bit**. Crop-only processing is an exact slice of the original
floating-point data. Rotation necessarily resamples the original full-resolution
image once; dimensions expand to avoid clipping. Grayscale/RGBA, source dtype,
and source device are preserved; geometry analysis/rotation are CPU/OpenCV work.
RGBA rotation uses premultiplied colors. Differently sized batch results are
padded white at bottom/right, not resized to each other.

Limitations: one dominant print per scan; no perspective correction, semantic
paper model, or guaranteed recovery of white-on-identical-white boundaries.
Bright/textured frames, extreme dust, multiple prints, very thin low-contrast
paper edges, or prints touching canvas edges can remain ambiguous. Inspect real
scans before trusting a large batch. No settings mean “remove the subject.”

## Paired Image Loader

The original inputs remain: `source_dir`, `output_dir`, `reverse`, and
`strip_trailing_numbers`. Outputs remain **output_image, source_image, filename**
in that order. Relative directory paths retain the old behavior: relative to
the current process directory. Absolute paths are least ambiguous.

Pairs match case-insensitive basenames, not file extensions. Natural sorting
puts photo2 before photo10. A trailing `(n)` copy marker is ignored for matching
only when requested; ordinary filename digits are never stripped. `filename`
remains the actual source stem, matching the old output contract.

Changes:

- Each node instance has independent, bounded cursor state. Separate nodes do
  not affect one another even when using the same folders or hidden node ID.
- Positions track the last successfully loaded basename, not an unstable numeric
  index. Adding an earlier file does not shift the cursor. Switching folder pairs
  preserves each folder pair's position. Removing the previous pair restarts
  from the first (or last in reverse).
- Both files decode before the cursor advances. A broken second file retries
  the same pair until fixed; it does not silently skip or misalign images.
- Every queued execution reloads/advances; cycling back to an earlier pair does
  not reuse a cached old image. Reverse starts from the last pair and wraps.
- Ambiguous name collisions are errors unless there is exactly one common-
  extension pairing. Unmatched basenames are reported once per changed list.
- Uses the tested Photo Loader preview decoder: EXIF orientation once, preserved
  16-bit samples, optional 8-bit ICC conversion, and explicit alpha compositing.

New optional controls: `color_management` (Convert to sRGB / Keep encoded values)
and `alpha_background` (White / Black), with the same defaults and limitations
as [Photo Loader](PhotoLoaderPreview.md), including explicit errors for 16-bit
ICC conversions that would otherwise lose precision. Animated/multipage files
use their first image. No resize aligns the two images automatically.

## Crop by Margins: Image and Mask

Same controls: `left_px`, `top_px`, `right_px`, `bottom_px`, and `snap_multiple`.
The image preview has `image` input/output; the mask preview has `mask`
input/output. Use identical margin values for both.

The two previews share one crop-box calculation. They preserve source pixels,
precision, channels, and device without clamping, resampling, or color changes.
MASK accepts HW/BHW/BHW1 and returns standard BHW, including multi-image batches.
Every 3D MASK is BHW; use a batch dimension for HWC1 masks.

Nonnegative margins trim inward. `snap_multiple` rounds the remaining width and
height **down**, keeping the top-left fixed; any extra trim comes from right and
bottom. There is no protection-mask or semantic framing logic in these manual
nodes. Margins that remove the entire image, or a requested multiple larger than
the remaining crop, raise an explanation instead of producing a hidden
one-pixel result that does not honor the settings. No negative-margin padding,
fill-color, or enforce-even widgets exist.
