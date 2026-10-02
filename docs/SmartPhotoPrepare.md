# Smart Photo Prepare

`Smart Photo Prepare` combines subject-aware cropping and exact editor sizing for real-photo edit workflows. It is designed for:

- FireRed Image Edit 1.1
- FLUX.2 Klein Base
- Qwen Image 2.1
- a direct editor-bypass path to SeedVR2, SUPIR, or another upscaler

The production node ID is `PortraitPhotoPrepareV2`. Connect Input & Crop Config
to its required `crop_config` socket: Smart Crop enablement, resolution policy,
and forced aspect family are set there. Fine mask/framing controls remain on
the processing node. The old unconfigured interface has been removed.

This node has four outputs: `prepared_image`, exact `native_crop`,
`prepared_mask`, and typed `prepare_info`. The previous 13-output
`SmartPhotoPrepareV2` and three-output compact ID are no longer registered.
The single production node uses the shared analysis/planning/application
engine. Lanczos resizing is floating-point; masks must match the input image grid.
[Engine contract, diagnostics, and wiring](SharedEngine.md)

Photo Prepare never routes around an editor. Use **Editor Result Gate** after
the selected editor's decoded output. It rejects a missing selected result;
enable `allow_direct` only for an intentional no-editor run. Do not connect
the prepared image directly to the upscaler on an editing run.

## Recommended layout

Put one small `Editor Target Beacon` inside each mutually exclusive editor group. Set the three beacons to FireRed Image Edit 1.1, Klein, and Qwen Image 2.1 respectively. New beacons default to Qwen Image 2.1. Connect them to one `Active Editor Target` outside the groups, then connect that target to one shared `Smart Photo Prepare` outside the groups.

```text
FireRed group beacon --\
Klein group beacon ----> Active Editor Target -> Smart Photo Prepare
Qwen 2.1 beacon ------/                             |
                                                     v
source photo -> optional subject mask -> Photo Prepare -> active editor
                                                     -> Editor Result Gate -> upscaler
```

`Active Editor Target` deliberately raises an error when more than one connected beacon is live. If every editor is muted, its default fallback is `Direct / No Editor`; this preserves the source resolution unless smart cropping is enabled. If you want Fast Groups Muter to enforce exclusivity, use a dedicated muter that matches only your editor groups and set that muter's restriction to “always one.” Do **not** apply “always one” to the existing v5 muter that also controls SmartCrop, upscaling, saving, and other processing groups.

## Resolution behavior

`Auto (preserve detail)` is the recommended policy. After selecting the crop/aspect family, it chooses the smallest available automatic tier that avoids downscaling. If the source exceeds all automatic tiers, it uses the largest: approximately 1 MP for FireRed, 1.5 MP for Klein, or 2K (~4 MP) for Qwen 2.1. A nearly identical crop may win when it preserves more detail; `framing_tolerance_percent` controls that allowance.

`Auto (closest scale)` instead chooses the smallest resize. FireRed's two Auto policies both stay near 1 MP. Force 1 MP and Force 1.5 MP remain available for every editor, but FireRed's Force 1.5 MP is an explicit experimental larger output, not a documented native tier. Force 2K (~4 MP) is available only for Qwen 2.1 and can use substantially more time and memory.

FireRed's Auto aspect profile uses the 17 approximately 1 MP dimensions from its [official 1.1 ComfyUI workflow](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/blob/main/firered-image-edit-1.1.json): 672×1568, 688×1504, 720×1456, 752×1392, 800×1328, 832×1248, 880×1184, 944×1104, 1024×1024, and their landscape counterparts. These are the presets used by [ComfyUI's FluxKontextImageScale](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_flux.py); their use in the example does not establish an exclusive mandatory FireRed size table. Force 1.5 MP scales these families to larger 16-aligned sizes.

Klein's two tiers use dynamic 16-aligned dimensions to follow the photo or smart subject frame with minimal cropping. Qwen Image 2.1 similarly uses dynamic 32-aligned dimensions near 1 MP, 1.5 MP, or 2048² pixels. Qwen 2.1's 2K recommendation is a pixel budget, not a single mandatory width/height pair; the 1.5 MP tier is a suite convenience tier. [Qwen 2.1 documentation](https://github.com/QwenLM/Qwen-Image-2.1)

The `prepared_image` dimensions are exact. `native_crop` contains the selected original source pixels before padding or resizing, so it remains useful for inspecting how much source detail was retained.

## Forced aspect ratios and protected regions

`forced_aspect_ratio` can be left at `Auto` or set to an exact `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, or `9:16` family. FireRed and Klein receive mathematically exact, 16-aligned dimensions; Qwen 2.1 receives mathematically exact, 32-aligned dimensions. Exact forced ratios can differ from approximate presets in model examples.

The subject mask guides placement in both Smart Crop modes, for Auto and forced aspects alike. With `smart_crop` off, the node chooses the largest compatible rectangle that retains every marked subject. With it on, the node seeks tighter framing around those same subjects with the configured margins. If no crop can retain the subjects and their available safety margins, the node compares limited cropping against padding as described below.

A connected protected mask is an absolute constraint in every crop mode, including Smart Crop off with aspect `Auto`. Each editor-native aspect candidate is placed inside the source with the protected region intact. The protected bounds receive a hard pad equal to 1% of the source's shorter dimension (at least two pixels). A further safety margin, set by `headroom_ratio` vertically or `side_margin_ratio` horizontally, is preserved toward the protected region's nearest source edge. This works when the head is at the bottom or side as well as at the top. If that margin is physically impossible, the crop retains as much as the source allows.

Safe crops retain the subject mask and protected region with their available margins. The fallback may trim only the small amount permitted by `crop_tolerance_px`; the separate protected-region mask remains a hard boundary. When the full subject can fit, the configured framing margins guide a tighter crop. With Smart Crop off, the largest safe rectangle is preferred.

The protected input follows normal ComfyUI mask semantics: white is the region that must be retained and black is unprotected.

A separate SAM3 node can supply this mask with the editable prompt `head`, as in the historical `PortraitUtilsv5.3.json` layout. Its output is a lazy input: when connected, SAM3 runs this additional segmentation whenever Smart Photo Prepare executes, regardless of the Smart Crop toggle or aspect setting. Change the prompt to describe another critical region for a non-portrait image. That saved workflow requires updated editor targets and gate sockets before use with this revision.

## Smart crop controls

With `smart_crop` off, the node retains the largest possible part of the photograph while using the subject and protected masks to position it. With it on, the node frames more tightly around the subject mask. The Direct / No Editor path keeps the entire source when Smart Crop is off and aspect is Auto.

The subject prompt is interpreted by the upstream segmentation node. Smart Photo Prepare frames all meaningful regions in its resulting mask: one person, multiple people, objects, or other non-portrait subjects. Quantile controls can include faint mask edges, but cannot exclude meaningful disconnected subjects or thin extremities. Remove unwanted mask noise upstream; the cropper does not choose a single largest component.

Headroom, footroom, and side margin constrain safe crop placement in both Smart Crop modes. Margins are fractions of the crop dimensions, rounded up to whole pixels. Horizontal gravity, bottom priority, and protected-region anchoring choose positions only within those bounds; they cannot discard available clearance during safe cropping. Bottom priority distributes excess vertical space after both top and bottom margins are satisfied. If a source edge already lacks the requested clearance, only that edge is limited to the available source pixels.

If the subject mask is missing or empty, the node uses the source frame as its starting point; a connected protected mask still guides placement. Uniform masks are not automatically inverted. The `debug` output reports the masks used and the selected crop.

## Crop or pad selection

Selection follows this order within the requested editor target, forced aspect, and resolution tier:

1. Prefer a crop that retains the whole subject and its available safety margins. This takes priority even if padding would change fewer pixels. Existing framing and resolution policies select among safe crops.
2. Only when no safe crop fits, consider crops whose intrusion stays within `crop_tolerance_px`, alongside padded alternatives. Choose the smallest total number of source pixels removed plus pixels added, before resizing. If a candidate both crops and pads, both operations count. An exact pixel-count tie prefers cropping.
3. Crops that intrude beyond the tolerance are excluded even if they remove fewer pixels. The separate protected-region mask, including its hard bounding-box pad, is never clipped.

`crop_tolerance_px` is a local optional control on Smart Photo Prepare. It defaults to **4 source pixels per edge**; set it to **0** to require strict subject and available-margin retention. Intrusion is measured from the available safety boundary, so existing clearance missing at an original image edge does not count as newly removed pixels. A limited crop can reach the subject itself only where its margin is smaller than the permitted intrusion. Each axis minimizes the worst intrusion before applying placement preferences.

For example, if a safe crop removes 82,000 pixels and padding adds 27,000, the safe crop still wins. When no safe crop fits, an allowed crop removing 800 pixels loses to padding that adds only 300. A crop requiring more than the permitted intrusion cannot win either comparison.

The resolution policy and `framing_tolerance_percent` continue to govern safe crops. In the fallback, they break ties between equal pixel-change counts; they cannot override a smaller total change. Forced aspect and tier choices remain in effect.

Choose `padding_fill` on **Input & Crop Config**: **Edge extension** repeats the
nearest image edge (the default), **Black** adds solid black, and **White** adds
solid white. The choice applies to the production Smart Photo Prepare node,
only when its crop/protection logic requires padding. It does not
request extra padding or change crop selection, safety margins, aspect ratios,
or native resolution buckets. The corresponding added area of `prepared_mask`
is always black. Solid fills are opaque on RGBA sources; original source alpha
is preserved. Padding is deterministic, not generated outpainting. Final resize
can blend a few pixels at the source/padding boundary according to the resampler.

`native_crop` remains the unpadded source crop. `crop_x`, `crop_y`, `crop_width`, and `crop_height` always refer to that rectangle in the original input image. `prepared_image` and `prepared_mask` include padding and the final resize. Diagnostics report `removed_px`, `added_px`, and their sum `changed_px`, plus either padding or limited-crop margin intrusion as `L/R/T/B` in source pixels. `scale_factor` describes resizing the padded canvas, and `crop_loss_percent` counts only discarded source pixels. Direct mode follows the same comparison for a forced aspect and does not resize it.

## FireRed Image Edit 1.1 wiring

Set the FireRed group's beacon to `FireRed Image Edit 1.1`. Follow the [official 1.1 workflow](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/blob/main/firered-image-edit-1.1.json) for the model, Qwen2.5-VL text encoder, Qwen image VAE, and sampler setup, with Photo Prepare supplying the finished frame:

1. Connect `prepared_image` directly to VAEEncode and use that latent for sampling.
2. Connect the same `prepared_image` to `image1` on both positive and negative `TextEncodeQwenImageEditPlus` nodes, with the VAE connected to each.
3. Send those conditioning outputs to the sampler.
4. Connect the decoded FireRed result to `firered_image` on Editor Result Gate; connect the gate output to the upscaler.

Bypass or remove the example workflow's `FluxKontextImageScale` after Photo Prepare. It would center-crop and resize the frame again, potentially undoing the protected region, forced aspect ratio, and chosen output size. Do not add another total-pixel scaler between preparation and these editor inputs.

The [stock Plus encoder](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_qwen.py) independently resizes the vision-language view to about 384² pixels and the conditioning VAE reference to about 1024² pixels. These preserve approximate aspect, not square dimensions. This does not change the separate sampling latent size. A Force 1.5 MP output therefore does not automatically provide 1.5 MP of source detail through the conditioning reference. No native-reference replacement helper is required.

Qwen 2511 has been retired. The gate's former `qwen_image` input has been replaced by `firered_image`, with no legacy alias. Restart ComfyUI, refresh the browser, and update the beacon selection and gate socket wiring in your workflow. Saved workflows and model files are not modified automatically.

## Qwen Image 2.1 wiring

Connect `prepared_image` to `image_1` in the native Qwen Image 2.1 edit subgraph and connect that group's decoded IMAGE to `qwen21_image` on Editor Result Gate. Keep the edit subgraph's `custom_size` off and its `resolution` control at `0`: ComfyUI then follows `image_1`'s dimensions, rounded to a multiple of 32. Photo Prepare already outputs 32-aligned dimensions, so no second resize is needed. A nonzero `resolution` value would rescale the edit canvas again; a mismatched custom size can shift the edit. FireRed uses the separate `firered_image` gate input and the older Plus encoder described above. [ComfyUI's Qwen 2.1 edit guide](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)

## Outputs

- `prepared_image`: exact editor-ready image, or source-scale crop/padded canvas for the direct path
- `native_crop`: selected source crop before padding or resizing
- `prepared_mask`: input mask with matching crop/resize and black padding
- dimensions and crop box: available through the typed `prepare_info` readers
- `prepare_info`: selected family, policy, crop, scale direction, and mask notes; readers can extract the resize scale and crop-loss percentage
