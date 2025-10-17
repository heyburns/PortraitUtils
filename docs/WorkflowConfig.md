# WorkflowConfig

`WorkflowConfig` centralises high-level settings for a portrait pipeline—upscale amounts, crop preferences, stitch options, seeds, notes—and pushes them to the rest of your graph. It also saves and loads presets so you can flip between looks without rewiring nodes.

---

## Inputs
- `template_name` – Name of the built-in preset to load first (e.g., `Default`). Provides fallback values.
- `saved_preset` – Filename for a preset stored on disk. Leave as `<none>` to skip loading.
- `load_saved_preset` – When `True`, merge the saved preset over the template.
- `initial_upscale` – Pre-upscale amount before your main render pass.
- `final_upscale` – Target upscale factor for the finished output.
- `tight_crop` – Flag indicating whether downstream nodes should prefer tight framing.
- `stitch_bypass_mask` – Signal to skip mask usage in stitch nodes.
- `stitch_opacity` – Default blend strength for composite steps.
- `crop_fuzz_percent` – Suggested tolerance for auto-cropping tools.
- `crop_pad_px` – Extra padding around detected crops.
- `seed` – Main seed value. Special numbers (`-1`, `-2`, `-3`) trigger ComfyUI’s random seed behaviour.
- `job_notes` – Freeform text that travels with the config.
- `save_preset_as` – Name to use when writing a new preset to disk.
- `write_preset` – When `True`, save the current configuration using `save_preset_as`.

Hidden sockets (`prompt`, `extra_pnginfo`, `unique_id`) are handled by ComfyUI automatically to keep seeds and metadata in sync.

---

## Outputs
- Individual sockets mirroring key settings (`initial_upscale`, `tight_crop`, etc.).
- `seed` – Resolved seed after handling special values.
- `job_notes` – Echo of the notes field.
- `config_bundle` – JSON string containing the entire configuration, perfect for custom script nodes.

---

## Where It Fits

Drop WorkflowConfig near the top of your graph and wire its outputs wherever you normally hardcode constants. It keeps branches aligned—croppers, stitchers, savers—and makes it easy to swap between “tight headshot” and “environmental portrait” setups with a single preset change.

---

## Tuning Tips

- Use presets to store different client looks. Toggle `load_saved_preset` to bring a saved set of numbers back instantly.
- Parse `config_bundle` in custom Python nodes to avoid running extra wires across the canvas.
- Keep `job_notes` updated with the context of your run (lighting notes, prompt tweaks) for easy reference later.

---

## Troubleshooting

- **Preset won’t load** – Confirm the file exists in ComfyUI’s preset directory and that `load_saved_preset` is enabled.
- **Seed doesn’t change** – Remember that special seeds require the node to execute; run the graph once to force a refresh.
- **Outputs look stale** – Make sure downstream nodes receive their values directly from WorkflowConfig instead of cached copies.

---

Screenshot: `docs/screenshots/workflow_config.png`
