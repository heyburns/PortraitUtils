# Intelligent AutoCrop

Node ID: `PortraitAutoCropPreview`. Category: `PortraitUtils/Transform`.

This production node removes solid photographic borders, including black image-hosting borders, and optionally dark text-bearing
hosting footers. Alamy-style borders/footers are an explicit use case; detection
does not depend on the provider's name, a specific font, or OCR.

The old `IntelligentAutoCropV2` ID is removed. This node defaults to a
conservative policy and also offers literal configured tolerances.

## Wiring

Restart ComfyUI and add **Intelligent AutoCrop**. Connect:

- `image`: the loader image (or whichever image the old AutoCrop processed).
- `crop_config`: the existing **Input & Crop Config** bundle or its Get node,
  using the unchanged `PORTRAIT_CROP_CONFIG_V2` socket type.

Reconnect the image output to the same downstream consumer. There are still six
outputs, in the original order: `image`, `trim_left`, `trim_top`, `trim_right`,
`trim_bottom`, `detected`. If you only used the image output, keep doing that.
The v5.3 workflow already uses this node.

Use it before subject-mask generation, just like the old border-removal stage.
Smart Photo Prepare still handles subject framing, protected masks, aspect
ratios, and native editor resolutions. AutoCrop does none of those operations.

## Configuration

The preview reads only the first, **AUTOCROP · SOLID BORDER REMOVAL** section of
your existing configuration bundle:

- `autocrop_detect_borders`: Enable independent solid-border scans at all edges.
- `autocrop_strip_bottom_banner`: Enable text-bearing dark-footer detection.
  Turn this **on** for Alamy-style branding strips. Your saved v5.2 crop config
  currently has it **off**; the preview does not override that setting.
- `autocrop_fuzz_tolerance`: RGB color variation tolerated around the fixed edge
  color, on the normalized 0–1 scale.
- `autocrop_edge_uniformity`: Required matching-pixel fraction. Footer lettering
  is treated separately from its dark background rather than as a solid border.
- `autocrop_pad_px`: Pixels of detected border/footer to retain. Applied once to
  each final margin, after any nested frames have been identified.

One local dropdown, `border_policy`, distinguishes the safety choices without
expanding the central config or adding output sockets:

- **Conservative** (default): Requires at least 98.5% matching coverage in a solid
  border line, caps the configured color tolerance at 0.12, and stops at strong
  outliers even if they are tiny. The edge color must remain stable, and the
  proposed inner rectangle must have broad, contrasting photo-boundary evidence.
- **Configured tolerances**: Uses your configured color/matching thresholds
  literally, without the conservative outlier/stability/rectangle safeguards.
  It still requires a bounded edge-connected run and a contrasting boundary.
  Useful for comparison or unusually noisy borders; inspect the result carefully.

Both policies stop at the first nonmatching content rather than jumping across
arbitrary gaps in the photo. Up to four passes handle nested frames and footers
initially hidden behind an outer border. Only one text footer is stripped.

## Footer detection

A footer must touch the bottom edge, have a consistent dark background, contain
small aligned bright components consistent with lettering, and have a contrasting
upper boundary. Its total height must be less than the bottom 20% search region.
The detector checks the whole proposed footer and rejects a boundary that cuts
through bright lettering; it does not add extra crop rows inside the photograph.

A plain black border does not need text or the footer switch: solid-border
detection handles it. Disabling footer detection may still remove empty solid
padding below its lettering, but it will not deliberately jump through the text.
Large artwork, scattered bright photo details, and a lone bright rectangle are
not sufficient text evidence by themselves.

These are image-structure heuristics, not proof of where a photograph ends.
Identical solid photo backgrounds and added borders can be visually ambiguous.
Conservative mode favors leaving uncertain pixels rather than consuming content;
some low-contrast, damaged, densely branded, or unusual layouts may remain.
Transparent RGBA images are passed through; composite their background explicitly
in Photo Loader first. This is not a general in-photo watermark remover.

## Pixel and batch behavior

This node only takes a rectangular slice. It never resizes, interpolates, clamps,
recolors, transfers the output to another device, or pads a smaller result with
black pixels. Output channel count and floating-point precision are preserved.
With both switches off, it returns the original tensor unchanged.

Detection analyzes CPU float32 pixels, sharing memory for ordinary CPU float32
loader outputs. Other input devices/precisions require an analysis copy. The
returned image always stays on its original device and uses its original samples.
It does not alter PyTorch thread settings, load models, or use GLSL.

For a genuine tensor batch, every member is analyzed and the minimum safe trim
on each edge is used for all members. The four scalar outputs therefore describe
the whole batch consistently. A borderless member can prevent cropping; no
per-image black padding or first-image-only trim metadata is introduced.

Trim values are original source pixels. `detected` means a nonzero crop was
actually applied after padding and batch safety checks. If you already have a
mask, crop it using the same margins; this node does not output or transform
masks. Invalid image shapes, unnormalized/nonfinite pixels, and mismatched config
bundles produce actionable errors instead of silently changing data.

Automated tests cover synthetic Alamy-style footers, solid and nested borders,
JPEG/noise cases, tiny foreground tips, smooth gradients, studio-background
ambiguity, padding, batch consistency, and precision preservation. Validate it
against your real hosting images before retiring the original node.
