# FilenameAppendSuffix
<div align="center"><img src="screenshots/filename_append_suffix.png" alt="Screenshot" width="300" /></div>


## Overview
`FilenameAppendSuffix` builds a new filename by appending a suffix to an existing path. It supports removing multi-part extensions (e.g., `.tar.gz`), trimming whitespace, and optionally preserving the original directory component.

## Inputs
- `filename` (`STRING`, default empty): Original filename or path.
- `suffix` (`STRING`, default `"supir"`): Text appended to the base name. Leave empty to omit the suffix altogether.
- `separator` (`STRING`, default `"-"`): Separator inserted between the base name and `suffix` when needed.
- `strip_all_extensions` (`BOOLEAN`, default `True`): When `True`, removes every trailing extension component (useful for multi-part archives). When `False`, only strips the final extension.
- `preserve_directory` (`BOOLEAN`, default `True`): When enabled, reattaches the original directory path to the generated filename.
- `trim_whitespace` (`BOOLEAN`, default `True`): Trims leading/trailing whitespace from the filename, suffix, and separator before processing.

## Outputs
- `filename` (`STRING`): The rewritten filename/path.

## Processing Notes
- If the base name already ends with the separator, the suffix is appended without duplicating it.
- When `suffix` is empty, the base name is returned as-is (subject to extension stripping).
- Directory handling uses `os.path.join`, so platform-specific separators are respected.

## Tips
- Chain this node after `LoadImageCombined` or `PortraitUtils_PairedImageLoader` to build save paths that mirror the input file names.
- Disable `preserve_directory` when you want to relocate files into a new output folder without carrying over their original path.*** End Patch
