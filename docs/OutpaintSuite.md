# Outpaint Configuration & Padding

**Outpaint Config** (`PortraitOutpaintConfigV2`) stores settings in one typed
bundle. **Outpaint Padding Compute** (`OutpaintPaddingComputeV2`) consumes that
bundle and an image, then emits resolved pixel padding.

## Configuration

- `mode`: Percent or Pixels.
- `gravity`: Position of the existing image within the expanded canvas.
- `horizontal_percent`, `vertical_percent`: Total expansion as a percentage of
  source width/height, used in Percent mode.
- `left_px`, `right_px`, `top_px`, `bottom_px`: Explicit expansion in Pixels mode.
- `outpaint_prompt`: Read locally by **Outpaint Prompt from Config** for an editor.

In Percent mode, center gravity splits expansion across opposite edges. Left
gravity keeps the source at the left and adds horizontal pixels on the right;
top gravity adds vertical pixels below. Corner choices combine those anchors.
Pixels mode uses the explicit values and ignores gravity.

## Wiring and outputs

Connect the image and the configuration bundle to Outpaint Padding Compute.
Its four outputs are `left`, `top`, `right`, and `bottom`, in that order.
The computation adds one pixel on the right/bottom if needed to make the final
canvas dimensions even. This is not an editor-native resolution guarantee.

Connect those outputs to a canvas-expansion node. Padding Compute calculates
geometry only; it does not generate new image content or feather a mask.

The old standalone Outpaint Config and unconfigured Padding Compute IDs have
been removed. See [Retired Nodes](RetiredNodes.md) and
[Typed Workflow Configuration](ConfigurationV2.md).
