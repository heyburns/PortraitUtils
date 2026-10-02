# Stitch by Mask

`PortraitStitchPreview` blends two images with a mask and a typed Edit & Composite Config.
The old unconfigured `StitchByMask` interface has been removed.

## Inputs

- `image_a`: Base/background image, retained where the mask is black.
- `image_b`: Replacement image, selected where the mask is white.
- `edit_config`: Required `PORTRAIT_EDIT_CONFIG_V2` bundle.
- `opacity_source`: Use `blend_opacity` or `stitch_opacity` from the bundle.
- `mask`: Required unless `bypass_mask` is enabled.
- `invert_mask`: Reverse mask coverage.
- `bypass_mask`: Blend the whole image using the selected opacity.
- `feather_radius`: Expand/feather mask coverage in processing pixels.
- `force_size`: Resize both images and the mask to the target width/height.
- `target_width`, `target_height`: Used only when force_size is enabled.

With force_size off, the images must share dimensions and the mask must match.
No metadata input/output is provided by this node.

## Outputs

- `stitched`: Composited image.
- `processed_mask`: Mask used for the blend, for inspection/downstream use.

The production node uses the corrected 2.0 compositor: it applies opacity once
after feathering, returns a standard BHW mask, and handles RGBA safely. The
`StitchByMask` and `StitchByMaskV2` IDs have been removed.
See the [replacement details](ReplacementPreviews.md#stitch-by-mask).

See [Typed Workflow Configuration](ConfigurationV2.md) for the edit bundle.
