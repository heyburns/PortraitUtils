# Auto White-Balance + Color Match

Node ID: `PortraitWhiteBalancePreview`. This is the production node.
The old `AutoWBColorMatch` ID is removed. It adds `preserve_luminance`.

## Wiring

Restart ComfyUI and add **Auto White-Balance + Color Match** from
`PortraitUtils/Analysis`. Connect `image` and `reference`, then send its
single `IMAGE` output to the next stage. The v5.3 workflow already uses it.

The original nine inputs retain their names, types, order, method choices, and
defaults. One additional control, `preserve_luminance`, is enabled by default.
The `reference` socket is retained as a required connection for compatibility;
WB-only methods ignore its pixels. For WB-only use, the source image itself can
also be connected to `reference`.

## Methods

| Method | What it does | Uses reference? |
| --- | --- | --- |
| `wb_grayworld` | Equalizes the source's alpha-weighted average linear RGB channels. | No |
| `wb_highlight` | Equalizes channels in the source's brightest eligible pixels. | No |
| `reinhard_lab` | Matches per-image mean and standard deviation in D65 Lab. | Yes |
| `lab_l_only` | Matches Lab lightness statistics without changing Lab a/b. | Yes |
| `wb_highlight+reinhard` | Applies source highlight WB, then matches its corrected Lab statistics to the reference. | Yes |

White balance uses independent channel gains in linear-light RGB. By default,
the selected pixels are neutralized to their existing average linear luminance:
neutral highlights are not automatically brightened to white. This preserves
the sampled average luminance, not every pixel's brightness. Reference matching
can still change exposure and color even with `preserve_luminance` enabled.

Disable `preserve_luminance` to target 0.95 sRGB in highlight WB, or the arithmetic
mean of the linear RGB channels in gray-world WB. This is an explicit brightness
choice, not a reproduction of the legacy node's faulty calculation.

## Controls

- `percentile`: highlight selection, default 95. Only used by highlight methods.
  Selection excludes black pixels and pixels with any sRGB channel at or above
  0.995. A 2,048-bin linear-luminance histogram estimates the percentile without
  sorting the full image. Higher settings select fewer, brighter pixels.
- `strength`: blend from unchanged source (0) to full correction (1), in sRGB.
- `clip_gamut`: clamp the final blended RGB to 0–1. No intermediate clipping.
  Keep enabled for ordinary photo/editor/upscaler workflows. Disabled output can
  contain negative or greater-than-one RGB; this node expects normalized inputs,
  so such output must be gamut-mapped before feeding it into another instance.
- `force_size`: limit the images used to estimate statistics. This **never
  resizes the output or replaces original pixels with an enlarged thumbnail**.
- `target_width` / `target_height`: analysis bounding box when `force_size` is
  enabled. Aspect ratio is preserved, and small inputs are never enlarged.
- `preserve_luminance`: the WB brightness behavior described above.

For faster analysis of large scans, enable `force_size` and start with the
default 1440×1080 box. A smaller sample may change the estimated correction,
particularly when neutral highlights are tiny. The fitted color transform is
always applied to the original full-resolution pixels.

## Safeguards and limitations

- Input is floating-point sRGB in `[batch, height, width, channels]`, normalized
  to 0–1. RGB, RGBA, and single-channel grayscale are supported. Active grayscale
  correction produces RGB; strength 0 returns the original tensor unchanged.
- Output retains source dimensions, batch size, floating-point dtype, and device.
  Half-precision inputs are processed in float32 and converted back at the end.
- Alpha is unchanged. Statistics are alpha-weighted, fully transparent RGB is
  ignored and preserved, and analysis downsampling uses premultiplied alpha.
- A single reference serves the whole source batch; an equal-sized reference
  batch matches pairwise. Other batch counts are rejected. Reference spatial
  dimensions can differ from the source. Only small fitted statistics cross
  devices if the reference and source reside on different devices.
- WB gains are limited to 0.25–4 in linear light. Missing color channels or no
  eligible highlights yield identity WB instead of unsafe division.
- Lab standard-deviation ratios are limited to 0.125–8. Channels with source
  **or reference** standard deviation below 0.05 Lab units use a mean shift only:
  flat images do not amplify tiny noise or erase source texture. These guards
  mean extreme references will not be matched exactly.
- Fully transparent reference images cause an explanatory error for matching
  methods. Fully transparent source pixels remain unchanged.
- Processing uses bounded row tiles rather than full-image Lab intermediates.
  The node does not load models, use GLSL, or change PyTorch's thread settings.

Neither gray-world nor highlight WB can identify a true neutral patch from color
statistics alone. Colored backgrounds, theatrical lighting, skin highlights,
and intended warm/cool grading can mislead them. Reference matching transfers
global color statistics, not a physically measured illuminant or local lighting.
Compare at modest strength before applying aggressive correction to skin tones.

## Color-math reference

The sRGB transfer curves and correctly oriented RGB/XYZ matrices follow the
[W3C CSS Color 4 conversion examples](https://www.w3.org/TR/css-color-4/#color-conversion-code).
Working Lab uses D65 consistently for both images; it is not CSS's D50 `lab()`
space. Conversion tests include neutral white/black, known primaries, round
trips, independent-channel WB, reference transforms, alpha, and detail retention.
