"""Deterministic basename pairing; no workflow state or node dependencies."""
from pathlib import Path
import re
from .photo_io import _natural_key, EXTENSIONS as _IMAGE_EXTENSIONS

def _directory(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Paired Loader preview: {label} is required; enter a directory path.")
    path = Path(value.strip()).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Paired Loader preview: {label} does not exist: {path}")
    return path

def _entries(directory, strip):
    result = {}
    for path in sorted(directory.iterdir(), key=lambda p: _natural_key(p.name)):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        key = re.sub(r"\s*\([0-9]+\)$", "", path.stem).strip() if strip else path.stem
        result.setdefault(key.casefold(), []).append(path)
    return result

def _pairs(source_dir, output_dir, strip):
    source, output = _entries(source_dir, strip), _entries(output_dir, strip)
    pairs = []
    for key in source.keys() & output.keys():
        left, right = source[key], output[key]
        if len(left) == len(right) == 1:
            pair = left[0], right[0]
        else:
            candidates = [(s, o) for s in left for o in right if s.suffix.lower() == o.suffix.lower()]
            if len(candidates) != 1:
                names = ", ".join(p.name for p in left + right)
                raise ValueError(f"Paired Loader preview: ambiguous basename {key!r}: {names}. "
                                 "Make names unique or disable strip_trailing_numbers.")
            pair = candidates[0]
        pairs.append((key, *pair))
    if not pairs:
        raise ValueError("Paired Loader preview: no matching basenames in the selected folders. "
                         "Extensions may differ; filenames must match apart from optional (n) copy markers.")
    pairs.sort(key=lambda pair: _natural_key(pair[1].stem))
    unmatched = (tuple(sorted(source.keys() - output.keys())),
                 tuple(sorted(output.keys() - source.keys())))
    return pairs, unmatched
