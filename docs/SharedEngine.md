# Shared engines and photo preparation

The shared engines power the production nodes. The historical
`PortraitUtilsv5.3.json` layout requires current editor targets and gate sockets;
v5.2 additionally uses retired node IDs. Saved workflows and model files are
not modified by this update. Historical Python IDs ending in `Preview` remain
the production IDs; ComfyUI displays their proper names.

## Processing boundaries

Node files describe ComfyUI interfaces and adapt execution to `core/` engines.
The engines do not import ComfyUI, `folder_paths`, registered node modules, or
model management. File exports receive an explicit output/temp environment.

| Internal module | Responsibility |
| --- | --- |
| `contracts.py`, `validation.py` | Immutable stage configs and shared, actionable validation |
| `tensors.py`, `resampling.py` | IMAGE/MASK layouts, explicit alignment, alpha-aware float resizing |
| `editor_profiles.py` | Editor-specific sizing policies, aligned dimensions, and bounded resolution cache |
| `geometry.py` | Source-coordinate framing, protection, crop/padding candidate selection |
| `panels.py` | Composite-photo divider detection and original-pixel panel extraction |
| `preparation.py` | Mask analysis, immutable plans, plan validation/application, result info |
| `borders.py`, `margins.py`, `outpaint.py` | Border removal, literal margin crops, pure outpaint geometry |
| `adjustments.py`, `white_balance.py`, `composite.py` | Automatic corrections, color matching, masked composition |
| `scans.py` | The distinct scanner-bed/print-border use case |
| `photo_io.py`, `pairing.py`, `state.py`, `export.py` | Decoding, matching, transactional cursors, atomic exports |

Configuration classes are still re-exported from `workflow_config_v2.py`, and
existing helper import locations remain compatibility facades. External code
does not need to change its config-class imports.

## Preparation pipeline

```text
IMAGE + subject MASK + protected MASK
                 |
            analyze_masks
                 |
             plan_photo        (no IMAGE processing)
                 |
       immutable PreparationPlan
                 |
             apply_plan        (literal crop -> optional pad -> one final resize)
                 |
       IMAGE + native_crop + MASK + info
```

The plan records source size, crop rectangle, padding, output size/profile,
framing controls, subject/protected bounds, mask status, and selection reason.
It contains no pixel tensors. Before application it checks source dimensions,
crop bounds, the native editor profile/tier, and protected pixels and outward
safety margins. Protection remains mandatory with Smart Crop off. Configured
crop tolerance may consume a small safety margin; it cannot remove protected
pixels. Invalid plans are rejected before crop/resize execution.

FireRed Image Edit 1.1 replaces the retired Qwen 2511 profile. Both Auto
policies use the 17 approximately 1 MP sizes selected by the official FireRed
workflow's FluxKontextImageScale node. Force 1.5 MP is an explicit experimental
larger output, not a documented native tier; FireRed does not expose a 2K tier.
Forced FireRed ratios use mathematically exact 16-aligned dimensions and can
differ from the example workflow's approximate aspect presets.

Flux Klein retains dynamic 16-aligned 1/1.5 MP sizes. Qwen Image 2.1 uses
dynamic 32-aligned 1/1.5 MP and 2K (~4 MP) sizes; forced aspect families
are mathematically exact. Auto (preserve detail) may select 2K for Qwen 2.1
when that avoids discarding source resolution.
[Profile details and source references](SmartPhotoPrepare.md#resolution-behavior)

Preparation plans also record `padding_fill`: Edge extension (default), Black,
or White, supplied by Input & Crop Config. It affects only the canvas application,
not analysis or crop/resolution selection. Added mask pixels always remain zero;
solid image fills are opaque on RGBA inputs, and `native_crop` remains an exact,
unpadded source slice. The optional config dropdown is appended to preserve the
position of existing widgets and adds no sockets to Smart Photo Prepare.

## Standard preparation node: wiring

Use **Smart Photo Prepare** (`PortraitPhotoPrepareV2`) for all new workflows. It
takes Input & Crop Config, Active Editor Target, an optional subject mask and
protected-region mask, and local fine framing controls. It has four outputs:

- `prepared_image`: native editor-resolution image for the selected editor,
  or a deliberate direct-upscaler run.
- `native_crop`: exact, unpadded source-pixel crop. It is always available.
- `prepared_mask`: aligned subject mask after crop/pad/resize.
- `prepare_info`: one typed `PORTRAIT_PREPARATION_INFO_V2` result. Carry the
  bundle to a consumer and unpack it locally with the readers below.

The old 13-output `SmartPhotoPrepareV2` and 3-output
`PortraitPhotoPrepareCompactV2` IDs are removed. Connected protected masks
retain the same lazy evaluation behavior in every mode.

Use local info readers only where needed: **Dimensions from Prepare Info**
(width/height), **Crop Box from Prepare Info** (source x/y/width/height),
**Scale from Prepare Info** (scale, x/y scale, crop loss), and **Summary from
Prepare Info** (debug, resolution, selection reason, or JSON).

## Editor result routing

Active Editor Target selects a model profile for preparation; it does not
route image pixels. **Editor Result Gate** receives that typed target
and the decoded FireRed, Klein, or Qwen 2.1 images through `firered_image`,
`klein_image`, or `qwen21_image`. Only the selected image is requested.
A missing selected editor result raises an error instead of falling back to a
pre-editor image. Connect the gate output directly to the upscaler.
New Editor Target Beacons default to Qwen Image 2.1; set the FireRed group's
beacon explicitly to FireRed Image Edit 1.1.

For a run that deliberately bypasses every editor, connect the direct source
branch and enable `allow_direct`. Turn it off when editing is mandatory, so
an all-editors-muted run fails explicitly. The older v5.3 workflow still
contains retired Qwen 2511 and Kontext targets/sockets and is not a current
wiring example. Rewire the replacement FireRed branch to `firered_image` and
the Qwen 2.1 branch to `qwen21_image` in your current workflow. The old
`qwen_image` input has no compatibility alias. Keep the gate and Active Editor
Target outside all Fast Groups Muter-controlled groups.

FireRed uses `prepared_image` directly for its sampling VAEEncode and
`image1` on both TextEncodeQwenImageEditPlus nodes. Bypass the example workflow's
FluxKontextImageScale after Photo Prepare: its center crop can undo protection
and exact framing. The stock Plus encoder still resizes its conditioning VAE
reference to about 1 MP, even for a larger sampling output.
[FireRed wiring and sources](SmartPhotoPrepare.md#firered-image-edit-11-wiring)

## Image and mask contract

- IMAGE is floating-point BHWC with 1, 3, or 4 channels; MASK is floating-point
  BHW. HW and BHW1 masks are accepted explicitly. A 3D MASK is always BHW,
  including one-pixel-wide images; HWC1 is not guessed.
- Preparation requires B=1. Other engines retain their documented batch rules.
- Preparation, color and composition inputs are finite, normalized 0–1 values.
  Literal margin crops do not clip HDR values. Disabled border removal remains
  a pixel-preserving bypass. Saving deliberately clips at the export boundary.
- IMAGE/MASK dtype and device are preserved by preparation. Low-precision math
  uses float32 work buffers; float64 inputs retain float64 work precision.
  RGBA resizing uses premultiplied color to prevent transparent-color halos.
- Native crops are exact tensor slices, with no color conversion or quantization.
  Bicubic/Lanczos reconstruction overshoot is clipped only after resampling.
- Lanczos now uses an antialiased, separable floating-point filter. It does not
  use ComfyUI's PIL helper, which converts through uint8. It supports grayscale,
  RGB and RGBA, on the current tensor device, without a retained image cache.
- Masks must match the current IMAGE grid. Preparation reports mismatches
  instead of silently stretching a subject/protected mask. Generate masks after
  border removal, or apply identical border trims to both image and masks.
- Missing subject masks no longer allocate and resize a source-sized zero mask.
  The zero output mask is created directly at output size.
- Legacy `enforce_image_format` retains its explicit float32/clamping/RGB
  conversion policy for legacy consumers; new engines use the strict contract.

## State, performance and diagnostics

The promoted loaders use per-instance `TransactionalCursor` behavior:
selection does not advance until all outputs decode successfully; each instance
retains at most 32 folder/listing positions. Photo-loader change checks use weak
node references and streamed content fingerprints. Paired-loader auto-advance
continues to force a fresh execution. Deprecated IDs retain their own input
contracts for saved graphs.

Resolution profiles use a 128-entry immutable-value cache; white-balance color
constants retain a 16-entry cache. No unbounded image, mask, model or shader cache
is introduced. Color statistics remain sampled/tiled where already supported;
the full-resolution source receives the correction. Auto Adjust now retains
dtype/alpha and skips conversion/copying when no correction is requested.

Set Smart Photo Prepare's optional `profile` switch to record host milliseconds
for validation/alignment, mask analysis, planning, application, and total time.
The default does not collect timings. Profiling does not call CUDA synchronize,
so these are host timings, not GPU-kernel completion measurements. Preparation
uses normal `PortraitUtils` logging; it never changes global logging settings,
Torch thread counts, model placement, VRAM/offload state, or shader contexts.

JSON diagnostics include the plan, selection reason, changed-pixel counts,
padding, margin intrusion, warnings, and pixel-center coordinate transforms.
`source_to_output()` and `output_to_source()` are inverse coordinate maps;
padding coordinates may not refer to real source pixels. Coordinates begin at
the IMAGE supplied to preparation, after any upstream border removal.

Config producers/readers/consumers now validate fields consistently. Text is
not silently converted into numbers, invalid modes/ratios are not silently
replaced, and out-of-range values are not silently clamped. Errors name the
offending setting and suggest the matching reader/config socket. Forced-aspect
errors suggest restoring Auto if the selection was accidental.

## Verification and deferred work

The CPU tests cover frozen existing interfaces, engine imports without ComfyUI,
native plans/replay, absolute protection/safety, compact readers, immutable
metadata, image/mask precision, float Lanczos, profiling, loader transactions,
export environments, and existing border/color/scan/compositing behavior.
GPU execution and the live editor/upscaler workflow still need user validation.

Packaging and dependency upgrades remain separate work. GPU editor/upscaler
behavior still needs validation in the user's live installation. Restart
ComfyUI and refresh the browser to load the new node definitions, then update
the retired beacon selections and Editor Result Gate sockets in your workflow.
