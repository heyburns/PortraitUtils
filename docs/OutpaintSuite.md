# Outpaint Suite

The Outpaint Suite captures your outpaint preferences in one place and converts them into pixel padding on demand. `OutpaintConfigNode` stores the knobs; `OutpaintPaddingComputeNode` turns those knobs into edges you can wire elsewhere.

---

## OutpaintConfigNode

### Inputs
- `preset_name` – Label for the current configuration. Handy when saving/loading workflows.
- `padding_percent_top`, `padding_percent_bottom`, `padding_percent_left`, `padding_percent_right` – Desired padding on each edge expressed as a percentage of image height or width.
- `feather_percent` – Soft transition amount applied during blending.
- `blend_mode` – Text tag you can pass downstream (“normal”, “add”, etc.). No enforcement, but useful for scripting nodes.
- `notes` – Free-form text field to document the intent of this preset.

### Outputs
- Mirrors every input so you can drive pins directly.
- `config_bundle` – JSON string containing the entire preset, useful for scripts or custom nodes.

---

## OutpaintPaddingComputeNode

### Inputs
- `width`, `height` – Current image dimensions.
- `padding_percent_*` – Percentages from `OutpaintConfigNode`.
- `minimum_padding_px` – Guarantees at least this many pixels of padding even if the percentage is tiny.

### Outputs
- `padding_top`, `padding_bottom`, `padding_left`, `padding_right` – Pixel counts for each edge.
- `feather_px` – Feather amount converted to pixels for quick drop-ins.

---

## Where It Fits

Use the suite when building outpaint templates where the padding amount changes by project or client. Instead of hardcoding numbers across the graph, store them once in the config node and let the compute node feed every mask generator, cropper, or stitcher downstream.

---

## Tuning Tips

- Keep percentages modest (5–15%) for natural extensions; higher values produce large blank borders that may need manual cleanup.
- Set `minimum_padding_px` to a safety value when working with very small or vertical crops so the padding never collapses to zero.
- Log or overlay `config_bundle` in your UI to track which preset is active during runs.

---

## Troubleshooting

- **Padding feels uneven** – Ensure the image dimensions reaching `OutpaintPaddingComputeNode` are correct; wrong width/height leads to mismatched pixels.
- **Feather looks harsh** – Raise `feather_percent` or adjust downstream blend nodes to respect the provided feather value.
- **Config changes don’t propagate** – Confirm you wired outputs from `OutpaintConfigNode` directly; copying values manually into multiple places defeats the point of the suite.

---

Screenshot: `docs/screenshots/outpaint_config_node.png`
