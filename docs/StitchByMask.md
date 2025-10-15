# StitchByMask


## Overview
`StitchByMask` composites two RGB images using a soft mask, with optional mask inversion, bypass blending, feathering, and enforced output dimensions. It is geared toward portrait editing workflows where foreground/background swaps or stitched passes need precise control.

## Inputs
- `image_a` (`IMAGE`): Base layer tensor.
- `image_b` (`IMAGE`): Layer to composite over `image_a`.
- `mask` (`MASK`, default `None`): Blend mask. Required unless `bypass_mask` is `True`. Accepted in common mask layouts (`[B,H,W]`, `[B,H,W,1]`, `[H,W]`, `[H,W,1]`).
- `invert_mask` (`BOOLEAN`, default `False`): Flips the mask before blending.
- `bypass_mask` (`BOOLEAN`, default `False`): When enabled, ignores the supplied mask and uses a constant opacity value.
- `opacity` (`FLOAT`, default `1.0`): Global blend strength (0 preserves `image_a`, 1 favours `image_b`). Applied after optional inversion.
- `feather_radius` (`INT`, default `5`): Radius (in pixels) for disk dilation before Gaussian feathering. Set to `0` to disable feathering.
- `force_size` (`BOOLEAN`, default `False`): If `True`, both images (and the mask) are resized to `target_width` × `target_height` before blending.
- `target_width` (`INT`, default `1344`): Width used when `force_size` is `True`.
- `target_height` (`INT`, default `768`): Height used when `force_size` is `True`.

## Outputs
- `stitched` (`IMAGE`): Composited RGB image, clamped to `[0, 1]`.
- `processed_mask` (`MASK`): The mask actually used for blending (after inversion, opacity scaling, feathering, and resizing).

## Processing Notes
- Input tensors are converted to BHWC float32 arrays. Alpha channels, if present, are truncated to RGB.
- When `force_size` is enabled, image tensors are resized with bilinear sampling and masks with nearest-neighbour sampling to avoid edge blurring before feathering.
- Feathering expands the binary core of the mask via convolution with a disk kernel, then applies a Gaussian blur (`sigma ≈ radius/2`) and takes the element-wise maximum with the original mask to avoid revealing previously masked pixels.
- In bypass mode the node builds a uniform mask at the requested `opacity`, allowing linear interpolation between `image_a` and `image_b` without supplying a mask.

## Tips
- Set `feather_radius` to `0` when blending masks that already contain feathered edges.
- Use bypass mode to create simple crossfades or to output the second pass at reduced opacity without worrying about masks.
- When enforcing a specific resolution (e.g., 1344×768 for Flux), ensure both inputs already match the target aspect to avoid distortion.*** End Patch
