# Crop by Margins: Image / Mask

Production IDs: `PortraitCropImageMarginsPreview` and
`PortraitCropMaskMarginsPreview`. The old IDs are removed. Both use the 2.0
margin engine. [Details](ReplacementPreviews.md#crop-by-margins-image-and-mask).

Both nodes have the corresponding `image` or `mask` input and four nonnegative
integer widgets: `left_px`, `top_px`, `right_px`, and `bottom_px`. They crop inward,
not outward. `snap_multiple` rounds the remaining width and height down to a
multiple, keeping the top-left fixed. There are no fill-color, negative-padding,
or enforce-even controls; previous documentation describing them was inaccurate.

Each has one output of the corresponding IMAGE/MASK type. Both use validated
geometry, preserve original pixels/dtype, support standard batched BHW masks,
and explain impossible crops rather than silently changing them.
