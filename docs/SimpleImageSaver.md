# SimpleImageSaver

`SimpleImageSaver` writes images to disk with a few essential controls: choose PNG or JPG, decide whether to embed metadata, and add a custom suffix to the filename.

---

## Inputs
- `image` – The photo to save.
- `filename` – Base name for the output file. Combine with `FilenameAppendSuffix` if you need version tags.
- `format` – `PNG` for lossless output or `JPG` for smaller files.
- `quality` – JPEG quality (ignored for PNG). Use values between 1–100.
- `embed_metadata` – When `True`, include ComfyUI metadata in the file.
- `suffix` – Optional text appended before the extension.
- `output_dir` – Destination folder. Leave blank to use ComfyUI’s default save path.

---

## Outputs
- `filepath` – The path of the saved file for logging or downstream reference.
- `image` – Pass-through of the original image so you can continue processing if needed.

---

## Where It Fits

Use SimpleImageSaver for quick exports during iteration: saving contact sheets, before/after comparisons, or intermediate passes without configuring the full ComfyUI save stack.

---

## Tuning Tips

- Stick with PNG when you plan to revisit the file; choose JPG with quality around 90 for client previews or web use.
- Set a default `output_dir` to keep test renders separate from final deliveries.
- Pair the node with `FilenameAppendSuffix` or `WorkflowConfig` to generate consistent naming.

---

## Troubleshooting

- **File not appearing** – Confirm the `output_dir` exists and that ComfyUI has write access. The node won’t create missing directories.
- **Metadata missing** – Make sure `embed_metadata` is toggled on and that downstream tools read PNG/JPG EXIF data.
- **Suffix duplicated** – Leave the node’s `suffix` empty when you already modified the filename upstream.

---

Screenshot: `docs/screenshots/simple_image_saver.png`
