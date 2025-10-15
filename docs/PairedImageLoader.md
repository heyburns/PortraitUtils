# PairedImageLoader


## Overview
`PairedImageLoader` pairs images from two directories (typically "source" and "output") and emits the next matching duo on every run. It keeps its position per node instance, supports forward/backward stepping, and reports the active filename for UI display or logging.

## Inputs
- `source_dir` (`STRING`): Directory containing the reference/source images. Relative paths resolve against the current working directory.
- `output_dir` (`STRING`): Directory containing the processed images to compare with the source set.
- `reverse` (`BOOLEAN`, default `False`): When enabled, stepping moves backwards through the matched list.
- `strip_trailing_numbers` (`BOOLEAN`, default `False`): Strips trailing ` (1)`, ` (2)`, etc., before matching filenames, useful when OS-level copy suffixes are present.
- `unique_id` (`UNIQUE_ID`, hidden): Injected by ComfyUI so the node can preserve its index across runs and avoid collisions when duplicated.

## Outputs
- `source_image` (`IMAGE`)
- `output_image` (`IMAGE`)
- `filename` (`STRING`): The source filename associated with the current pair (base name only).

## Matching & Caching Behaviour
- Filenames are matched case-insensitively by base name (optionally after stripping trailing numbers). Extensions may differ; the loader picks the first intersection and logs any extension mismatches.
- Directories are rescanned on each execution. The node maintains an index keyed by a signature derived from file names, sizes, and modification times so it can resume where it left off unless the listing changes.
- One-time warnings are printed when files exist only in one directory or when name collisions arise after stripping suffixes.
- Because `NOT_IDEMPOTENT = True`, ComfyUI will not cache outputs—the node always reloads the active pair, so edits on disk are visible immediately.

## Tips
- Use the `reverse` toggle to walk backwards when you overshoot a comparison target.
- Connect the `filename` output to a `ShowText`-style widget (or metadata saver) to document which pair was reviewed.
- Keep directory sizes manageable; the loader reads both folders on every run to stay in sync.*** End Patch
