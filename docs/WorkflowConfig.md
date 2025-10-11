# WorkflowConfig

## Overview
`WorkflowConfig` centralises high-level settings (upscale amounts, cropping policy, stitch opacity, seed control, notes) and exposes them both as individual sockets and as a JSON bundle. It supports loading/saving presets, honours server-side random seed generation, and attempts to keep workflow metadata in sync when special seed values are used.

## Inputs
Required parameters:
- `template_name` (`STRING`): Built-in preset to load (currently `"Default"`). Templates provide fallback values for all outputs.
- `saved_preset` (`STRING`): Filename stem of a saved preset in the workspace (`<none>` to disable).
- `load_saved_preset` (`BOOLEAN`, default `False`): When `True`, merges the chosen `saved_preset` over the template defaults.
- `initial_upscale` (`FLOAT`, default `1.0`): Upscale factor applied before the main pass.
- `final_upscale` (`FLOAT`, default `4.0`): Target upscale factor for the final output.
- `tight_crop` (`BOOLEAN`, default `False`): Indicates whether downstream nodes should prefer tight subject crops.
- `stitch_bypass_mask` (`BOOLEAN`, default `False`): Signals that stitched blends should ignore masks.
- `stitch_opacity` (`FLOAT`, default `1.0`): Preferred opacity for stitch operations (0–1).
- `crop_fuzz_percent` (`FLOAT`, default `5.0`): Suggested fuzziness allowance for auto-cropping nodes.
- `crop_pad_px` (`INT`, default `0`): Additional padding to apply around detected crops.
- `seed` (`INT`, default `0`): Primary seed value. Special values `-1`, `-2`, `-3` trigger server-side random generation (increment/decrement where possible).
- `job_notes` (`STRING`, default empty): Arbitrary notes stored alongside the configuration.

Optional parameters:
- `save_preset_as` (`STRING`): Filename stem to use when persisting the current configuration.
- `write_preset` (`BOOLEAN`, default `False`): When `True`, writes `save_preset_as` to disk in the preset folder.

Hidden inputs (managed by ComfyUI):
- `prompt` (`PROMPT`): Used to persist generated seeds back into the executing graph.
- `extra_pnginfo` (`EXTRA_PNGINFO`): Allows embedding the resolved seed into workflow metadata.
- `unique_id` (`UNIQUE_ID`): Required to update workflow widgets when special seeds are consumed.

## Outputs
- `initial_upscale` (`FLOAT`)
- `final_upscale` (`FLOAT`)
- `tight_crop` (`BOOLEAN`)
- `stitch_bypass_mask` (`BOOLEAN`)
- `stitch_opacity` (`FLOAT`)
- `crop_fuzz_percent` (`FLOAT`)
- `crop_pad_px` (`INT`)
- `seed` (`INT`): The resolved seed after handling special values.
- `job_notes` (`STRING`)
- `config_bundle` (`STRING`): JSON string containing all of the above values.

## Processing Notes
- Templates and saved presets are merged in order: template defaults → saved preset (if loaded) → current widget values.
- When `seed` is `-1`, `-2`, or `-3`, the node generates a random seed server-side. Because the client-side UI cannot receive the update automatically, `WorkflowConfig` attempts to write the new seed back into both the workflow metadata and the executing prompt using the hidden inputs.
- Presets are stored under `ComfyUI/user/default/workflows/workflow_config_presets/` relative to the install. Files are JSON-formatted and written only when `write_preset` is explicitly toggled.

## Tips
- Drive other nodes (e.g., `AutoCropBorders`, `StitchByMask`, custom scripts) using the JSON bundle to keep all config values in sync.
- Use `job_notes` as a quick logging field—its contents travel alongside the bundle and can be embedded into output metadata downstream.
- Save presets for different portrait styles (tight headshots vs. wide environmental shots) and swap them via the dropdown without rearranging your workflow.*** End Patch
