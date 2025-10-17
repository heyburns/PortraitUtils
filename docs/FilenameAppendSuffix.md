# FilenameAppendSuffix

`FilenameAppendSuffix` rewrites a filename by adding a suffix to the base name while keeping extensions and directories tidy. Use it to mark version numbers, processing passes, or other notes on the way out of a workflow.

---

## Inputs
- `filename` – Original path or file name.
- `suffix` – Text to append to the base name. Leave blank to skip adding anything.
- `separator` – Character placed between the base name and suffix (default `-`).
- `strip_all_extensions` – When `True`, remove every extension (e.g., `.tar.gz`). Set to `False` to remove only the final extension.
- `preserve_directory` – Keep the original folder structure if `True`; otherwise only the file name is returned.
- `trim_whitespace` – Clean leading and trailing spaces before processing.

---

## Output
- `filename` – The rewritten name or path.

---

## Where It Fits

Use this node ahead of savers when you want to tag output files with notes like `-graded`, `-v2`, or `-autoadjusted` without writing a custom Python block.

---

## Tuning Tips

- Swap the separator to `_` or `.` when matching existing naming schemes.
- Disable `preserve_directory` if you plan to funnel all outputs into a single folder.
- Leave `strip_all_extensions` off when working with multi-part archives you want to keep intact.

---

## Troubleshooting

- **Suffix duplicates separator** – The node avoids doubling the separator, but double-check the suffix itself doesn’t start with one.
- **Directory disappears** – Make sure `preserve_directory` is enabled if you need the original folder path.
- **Whitespace still visible** – Confirm `trim_whitespace` is on; otherwise leading/trailing spaces are kept as-is.

---

Screenshot: `docs/screenshots/filename_append_suffix.png`
