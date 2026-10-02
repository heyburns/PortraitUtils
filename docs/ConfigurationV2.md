# Typed Workflow Configuration

The current configuration nodes replace one socket and one Set/Get variable per setting with four typed, stage-scoped configuration buses. Their V2 IDs and socket types remain unchanged; display names no longer carry a V2 suffix.

| Producer | Socket type | Settings |
| --- | --- | --- |
| Input & Crop Config | `PORTRAIT_CROP_CONFIG_V2` | AutoCrop border removal first; Smart Crop framing, native resolution policy, forced aspect ratio, subject prompt, and padding fill second |
| Edit & Composite Config | `PORTRAIT_EDIT_CONFIG_V2` | edit prompt, blend/stitch opacity, three short stitch prompts |
| Outpaint Config | `PORTRAIT_OUTPAINT_CONFIG_V2` | geometry mode, gravity, percent/pixel expansion, prompt |
| Finish & Run Config | `PORTRAIT_FINISH_CONFIG_V2` | scale factors, automatic color controls, flip, seed |

Each producer has one output. A single KJ Set node publishes each bus across the workflow. PortraitUtils V2 processing nodes consume the appropriate bus directly. Small reader nodes expose one primitive only where a third-party node or subgraph input requires it.

This keeps expansion local to the consumer instead of placing dozens of primitive sockets in the configuration center.

**Edit & Composite Config** has three single-line stitch prompts in place of the
former multiline `stitch_mask` field. Use three **Edit Prompt from Config**
readers, selecting `stitch_prompt_1`, `stitch_prompt_2`, and `stitch_prompt_3`,
to drive three SAM3 prompt inputs. The config producer still has one output.

Saved reader nodes selecting `stitch_mask` must be changed to one of the new
fields. If an existing config node does not show all three widgets after
refreshing ComfyUI, recreate that node and reconnect its single config output.

## Configured processing nodes

- `Intelligent AutoCrop` reads border tolerances and padding from the crop bus.
- `Smart Photo Prepare` reads Smart Crop enablement, resolution policy, forced aspect ratio, and padding fill from the crop bus. Fine mask/framing controls remain local to the processing node. The standard four-output preparation node uses the same settings.
- `Outpaint Padding Compute` reads all outpaint geometry from the outpaint bus and outputs only the four resolved pixel paddings.
- `Auto Adjust (Photoshop-style)` reads the automatic adjustment toggles from the finish bus.
- `Stitch by Mask` selects either blend or stitch opacity from the edit bus.

These nodes call the existing processing implementations rather than maintaining duplicate algorithms.

Configuration definitions and validation now live in `core/contracts.py` and
are re-exported under the same names. Producers and consumers report malformed
fields explicitly rather than silently casting text or clamping bad values.
The config socket types remain unchanged. Edit & Composite Config's widget list
and Edit Prompt from Config's selector now contain three stitch prompts.

`padding_fill` is an appended optional dropdown on **Input & Crop Config**:
**Edge extension** (the existing default), **Black**, or **White**. It fills only
canvas pixels added by Photo Prepare; it does not affect AutoCrop border removal,
crop selection, safety margins, or the protected region. Mask padding stays black.
Existing configs that omit this setting use Edge extension. The widget is appended
after the subject prompt to avoid shifting saved widget values. No new output or
Set/Get connection is needed; Smart Photo Prepare receives it through `crop_config`.

The standard four-output preparation node adds a separate
`PORTRAIT_PREPARATION_INFO_V2` result bus—not another settings bundle—and
also retains its exact native source crop. Local readers unpack dimensions,
crop geometry, scale, or diagnostics beside the consumer.
[Preparation wiring guide](SharedEngine.md#standard-preparation-node-wiring)

The promoted [Intelligent AutoCrop](AutoCropPreview.md) accepts the same
crop-config bundle and runs the border/footer detector. It does not change
Smart Photo Prepare's framing/resolution controls. The superseded AutoCrop ID
has been removed.

## Third-party boundaries

Typed configuration objects cannot connect directly to ordinary `STRING`, `INT`, or `FLOAT` inputs. The V2 readers provide those values locally:

- Crop Prompt from Config
- Edit Prompt from Config
- Outpaint Prompt from Config
- Seed from Config
- Scale from Config

## Source metadata

`Load Image + Source Info` has two outputs: the image and one `PORTRAIT_SOURCE_INFO_V2` object. Filename and output directory are unpacked beside the saver. Width and height remain stored in the source object without occupying sockets in the configuration center.

Images, masks, color references, and face references are intentionally not placed inside configuration objects. Keeping pixel tensors separate makes previews, caching, and troubleshooting clearer.

## Compatibility

Removed IDs are listed in [Retired Nodes](RetiredNodes.md). The historical local
v5.2 workflow uses some of them and will show missing nodes.
Typed bundles work with matching PortraitUtils consumers/readers; they cannot
connect to primitive sockets without an adapter.

No workflow JSON is bundled in this repository. The historical local
`PortraitUtilsv5.3.json` uses the four-output Smart Photo Prepare, updated I/O,
and Editor Result Gate, but its editor target labels and gate socket wiring
need updating for the current editors. See the
[current editor wiring](SmartPhotoPrepare.md).
