# OtherConfigNode
![Screenshot](screenshots/other_config_node.png)


## Overview
`OtherConfigNode` is a lightweight configuration hub that emits a handful of numeric and boolean values commonly reused across portrait workflows. Unlike `WorkflowConfig`, it does not interact with presets or metadata—its role is to provide a simple socket interface that keeps related parameters grouped.

## Inputs
- `initial_upscale_megapixels` (`FLOAT`, default `1.0`): Megapixel target for the initial upscale stage.
- `final_upscale_megapixels` (`FLOAT`, default `4.0`): Megapixel target for the final upscale stage.
- `tight_crop` (`BOOLEAN`, default `False`)
- `blend_factor` (`FLOAT`, default `1.0`): Blend strength (0–1) for downstream compositing nodes.
- `stitch_bypass_mask` (`BOOLEAN`, default `False`)
- `input_dir` (`STRING`, default empty): Directory hint propagated to loaders or savers.

## Outputs
- `initial_upscale (megapixels)` (`FLOAT`)
- `final_upscale (megapixels)` (`FLOAT`)
- `tight_crop` (`BOOLEAN`)
- `blend_factor` (`FLOAT`)
- `stitch_bypass_mask` (`BOOLEAN`)
- `input_dir` (`STRING`)

## Processing Notes
- The node clamps upscale megapixels to a minimum of `0.1` and blend factors to the `[0, 1]` range.
- Whitespace is preserved in `input_dir`; trim it upstream if you require strict paths.

## Tips
- Use this node as a compact fan-out hub when you don’t need the full preset system from `WorkflowConfig`.
- Route `blend_factor` into `StitchByMask.opacity` or similar nodes to keep mix strength consistent across passes.*** End Patch
