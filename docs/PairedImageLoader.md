# PairedImageLoader

`PairedImageLoader` pulls matching source/processed image pairs from aligned folders. It’s ideal for before/after reviews or comparison pipelines where both streams must advance together.

---

## Inputs
- `source_folder` – Directory containing the “before” images.
- `target_folder` – Directory containing the “after” images.
- `pattern` – Optional filename template (e.g., `{name}_after.png`) to align variations when the names differ.
- `auto_next` – Step to the next pair on every execution.
- `loop` – Restart from the top when you reach the end of the folder.
- `shuffle` – Randomise the order of the pairs.
- `strict_matching` – When `True`, only emits pairs where both files exist. Disable to allow singletons.

---

## Outputs
- `source_image` – The “before” image tensor.
- `target_image` – The matching “after” image tensor.
- `source_filename`, `target_filename` – Filenames currently loaded.
- `index` – Position in the list for logging or syncing with other nodes.

---

## Where It Fits

Use PairedImageLoader for A/B pipeline QA, training-data spot checks, or client review decks where each processed shot should sit next to its original capture.

---

## Tuning Tips

- Keep `strict_matching` enabled during quality checks so missing files are obvious.
- Combine `auto_next` with `ComparisonGate` and your preferred viewer for smooth slideshow-style reviews.
- Use the `pattern` field to translate naming schemes, such as turning `portrait.jpg` into `portrait_graded.png`.

---

## Troubleshooting

- **Pairs go out of sync** – Double-check folder naming and the `pattern` format. The node advances both sides together when matches are found.
- **Only one image appears** – If strict matching is on, missing companions stop the pair. Either add the file or disable strict mode to let singletons through.
- **Order unexpected** – Toggle `shuffle` off and ensure the folders list files in the same sequence.

---

Screenshot: `docs/screenshots/portraitutils_paired_image_loader.png`
