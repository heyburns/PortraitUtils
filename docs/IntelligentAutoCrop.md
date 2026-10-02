# Intelligent AutoCrop

The production [Intelligent AutoCrop](AutoCropPreview.md) node
(`PortraitAutoCropPreview`) removes uniform borders before subject framing.
It consumes an image and Input & Crop Config. The old
`IntelligentAutoCrop` and `IntelligentAutoCropV2` IDs have been removed.

Removing solid black image-hosting borders, including Alamy-style borders, is
an explicit purpose of this stage. The production node offers a conservative
policy or literal configured tolerances and a shared safe crop rectangle for
image batches.

## Configuration

Settings are in the first, **AUTOCROP · SOLID BORDER REMOVAL** section of the
configuration node:

- `autocrop_strip_bottom_banner`: Detect a dark strip with bright content touching
  the bottom edge and remove it before border detection.
- `autocrop_detect_borders`: Enable uniform-border scanning on all four edges.
- `autocrop_fuzz_tolerance`: Allow color variation in a border.
- `autocrop_edge_uniformity`: Fraction of a line that must match its border color.
- `autocrop_pad_px`: Retain this many pixels of a detected border.

Smart Crop and resolution/aspect controls belong to the separate second section
and are consumed by Smart Photo Prepare, not by this node.

## Outputs

`image`, `trim_left`, `trim_top`, `trim_right`, `trim_bottom`, and `detected`.
Trim values are source-image pixels. This node does not output a transformed mask;
generate segmentation from the cropped image or crop existing masks consistently.

Detection analyzes a CPU copy where necessary; the output stays on the input
tensor's device without image resampling. Border and banner detection remain
heuristic, so inspect the result when tuning tolerances.

See [Typed Workflow Configuration](ConfigurationV2.md).
