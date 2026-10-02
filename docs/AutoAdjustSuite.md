# Auto Adjust Suite

`AutoAdjustV2` (displayed as Auto Adjust (Photoshop-style)) provides conservative, Photoshop-style automatic contrast,
tone, and color correction. Adobe's exact selection heuristics are proprietary,
so the results are designed to follow the documented behavior rather than be a
bit-for-bit Photoshop clone.

## Choosing an adjustment

The three corrections are normally alternatives. Use `adjustment_mode` to pick
one command directly:

- **Auto Levels** applies one black/white-point stretch to all RGB channels. It
  preserves channel relationships and corresponds most closely to Photoshop's
  Auto Contrast / Enhance Monochromatic Contrast behavior.
- **Auto Tone** in `Per-channel` mode calculates black and white points for R,
  G, and B independently. This corresponds to Photoshop Auto Tone / automatic
  Levels and can remove or introduce a color cast.
- **Auto Color** identifies average dark and light colors, sets channel
  endpoints, and can neutralize a genuinely low-chroma midtone. It refuses to
  treat saturated skin, foliage, or colored lighting as a gray reference.

`Use Finish Config switches` lets the three booleans in Finish & Run Config
control execution. Enabling several switches intentionally stacks them
in this order:

```text
Auto Levels -> Auto Tone -> Auto Color
```

That is more aggressive than the normal Photoshop workflow and can increase
clipping, so use it only when the stacked result is intentional.

## Inputs

- `image` – ComfyUI image batch. RGB, grayscale, and RGBA are accepted; alpha
  is preserved and fully transparent pixels are excluded from statistics.
- `finish_config` – Required `PORTRAIT_FINISH_CONFIG_V2` bundle carrying
  `auto_levels`, `auto_tone`, `auto_color`, and `flip_horizontal`.
- `precision` – `Histogram (fast)` uses 256 bins and is recommended for normal
  photographic work, especially after upscaling. `Exact` sorts all pixels for
  exact percentile measurements.
- `levels_shadow_clip_pct` / `levels_highlight_clip_pct` – Extreme pixels
  ignored when setting the shared Levels endpoints. These settings are also
  used by Auto Color. Photoshop's traditional default is `0.1%`.
- `levels_gamma_normalize` – Optional custom midtone normalization. This is a
  Levels-style convenience, not part of Photoshop Auto Contrast.
- `tone_mode` – `Per-channel` matches Auto Tone. `Monochromatic` uses the same
  color-preserving method as Auto Levels.
- `tone_shadow_clip_pct` / `tone_highlight_clip_pct` – Clipping percentages for
  Auto Tone.
- `snap_neutral_midtones` – Used by Auto Color. Candidate pixels must be in the
  midtone range and have low chroma before they can influence color balance.
- `flip_horizontal` (in Finish & Run Config) – Mirrors RGB and alpha after adjustment.
- `adjustment_mode` – Select one Photoshop-style command exclusively, or use
  the switches supplied by Finish & Run Config.
- `strength` – Blends the corrected image with the original from 0 to 100%,
  comparable to lowering an adjustment layer's opacity.

Flat or nearly flat channels are left unchanged rather than being expanded into
noise or mapped to black. A clip value of zero is supported and uses the true
minimum or maximum.

## Suggested starting points

- Color already looks correct, but contrast is weak: select Auto Levels.
- Each RGB channel occupies a noticeably different range: select Auto Tone in
  `Per-channel` mode.
- The image has a plausible neutral object but a global color cast: select Auto
  Color with `snap_neutral_midtones` on.
- Portrait dominated by intentional warm light: prefer Auto Levels; inspect
  Auto Color carefully before using it.

## Shared configuration

`PortraitFinishConfigV2` shares the four workflow switches `auto_levels`,
`auto_tone`, `auto_color`, and `flip_horizontal` in one typed bundle. Connect it
directly to each adjustment node's `finish_config` socket. The old standalone
Auto Adjust and AutoColor Config interfaces have been removed.
