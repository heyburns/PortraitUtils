# LoadImageCombined

`LoadImageCombined` is a flexible image loader that handles single files, folders, and auto-advancing batches in one node. It keeps multi-shot reviews moving without juggling separate loaders.

---

## Inputs
- `image_path` – File path to a single image. Leave blank if you want to use folder mode.
- `folder_path` – Directory to scan for images. Works with recursive or flat structures.
- `recursive` – Search subfolders when set to `True`.
- `auto_next` – When `True`, the node advances to the next image each time it executes.
- `loop` – Restart from the beginning when the folder list is exhausted.
- `shuffle` – Randomise the order for variety during reviews.
- `max_batch` – Number of images to load per run. Set to `1` for single-image mode.

---

## Outputs
- `image` – Loaded image tensor or batch.
- `mask` – Optional alpha/mask channel when the file provides one; otherwise emits zeros.
- `filename` – Name of the file currently loaded.
- `next_index` – Position of the next image the node will load (useful for sequencing).

---

## Where It Fits

Use LoadImageCombined at the front of portrait pipelines when you bounce between single reference frames and whole folders of stills. It’s also handy for QA passes where you want to step through before/after pairs without reconfiguring the graph.

---

## Tuning Tips

- Enable `auto_next` with `loop` for unattended slideshows or automated batch processing.
- Keep `shuffle` off when the order matters (e.g., matching filenames later in the workflow).
- Set `max_batch` higher than `1` only when downstream nodes can handle true batches; many portrait tools expect single images.

---

## Troubleshooting

- **Node repeats the same image** – Confirm `auto_next` is enabled and `max_batch` is `1`. In batch mode it stays on the same set until the next execution.
- **Images appear in strange order** – Disable `shuffle` and ensure `recursive` is set according to your folder structure.
- **Mask output is empty** – The source file likely lacks an alpha channel; this is normal. Supply a separate mask if needed.

---

Screenshot: `docs/screenshots/load_image_combined.png`
