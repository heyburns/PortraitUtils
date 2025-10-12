from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

import node_helpers

_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

_TRAILING_NUMBER_PATTERN = re.compile(r"\s*\(\d+\)$")
_NATURAL_KEY_PATTERN = re.compile(r"(\d+)")


def _natural_sort_key(value: str) -> List[object]:
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in _NATURAL_KEY_PATTERN.split(value)
    ]


def _normalize_base(stem: str, strip_numbers: bool) -> str:
    base = stem
    if strip_numbers:
        base = _TRAILING_NUMBER_PATTERN.sub("", base)
    return base.lower()


def _resolve_directory(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


@dataclass(frozen=True)
class _FileEntry:
    path: Path
    name: str
    stem: str
    normalized_key: str
    ext: str
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class _Pair:
    key: str
    display_name: str
    source: _FileEntry
    output: _FileEntry


@dataclass
class _NodeState:
    pairs: List[_Pair] = field(default_factory=list)
    index: int = -1
    signature: Tuple | None = None
    warn_signature: Tuple | None = None


_STATE: Dict[str, _NodeState] = {}
_SIGNATURE_INDEX: Dict[Tuple, int] = {}


def _coerce_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _coerce_str(value[0])
    return str(value)


class PortraitUtils_PairedImageLoader:
    CATEGORY = "PortraitUtils"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    FUNCTION = "load_next_pair"
    RETURN_NAMES = ("output_image", "source_image", "filename")
    NOT_IDEMPOTENT = True

    def __init__(self):
        self._state_key = uuid.uuid4().hex

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_dir": ("STRING", {"default": "", "multiline": False}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "reverse": ("BOOLEAN", {"default": False}),
                "strip_trailing_numbers": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Strip trailing \" (n)\" suffixes before matching.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def load_next_pair(
        self,
        source_dir,
        output_dir,
        reverse=False,
        strip_trailing_numbers=False,
        unique_id=None,
    ):
        if isinstance(source_dir, list):
            source_dir = source_dir[0]
        if isinstance(output_dir, list):
            output_dir = output_dir[0]
        if isinstance(reverse, list):
            reverse = reverse[0]
        if isinstance(strip_trailing_numbers, list):
            strip_trailing_numbers = strip_trailing_numbers[0]
        if isinstance(unique_id, list) and unique_id:
            unique_id = unique_id[0]

        reverse = bool(reverse)
        strip_trailing_numbers = bool(strip_trailing_numbers)

        current_key = getattr(self, "_state_key", None)
        if unique_id is not None:
            state_key = str(unique_id)
        else:
            if current_key is None:
                current_key = uuid.uuid4().hex
                self._state_key = current_key
            state_key = current_key

        migrated_state = None
        if current_key is not None and state_key != current_key:
            migrated_state = _STATE.pop(current_key, None)

        self._state_key = state_key

        if not source_dir:
            raise ValueError("PortraitUtils_PairedImageLoader: source_dir is required.")
        if not output_dir:
            raise ValueError("PortraitUtils_PairedImageLoader: output_dir is required.")

        src_path = _resolve_directory(source_dir)
        out_path = _resolve_directory(output_dir)

        if not src_path.exists() or not src_path.is_dir():
            raise ValueError(
                f"PortraitUtils_PairedImageLoader: source_dir not found: {src_path}"
            )
        if not out_path.exists() or not out_path.is_dir():
            raise ValueError(
                f"PortraitUtils_PairedImageLoader: output_dir not found: {out_path}"
            )

        state = _STATE.setdefault(state_key, migrated_state or _NodeState())

        pairs, signature, warnings = self._scan_directories(
            src_path, out_path, strip_trailing_numbers
        )

        state.pairs = pairs
        if signature != state.signature:
            state.signature = signature
            if state.index >= len(pairs):
                state.index = len(pairs) - 1

        if signature != state.warn_signature and any(warnings.values()):
            self._emit_warnings(warnings, src_path, out_path)

        state.warn_signature = signature

        if not pairs:
            raise RuntimeError(
                "PortraitUtils_PairedImageLoader: no matching images found "
                f"between {src_path} and {out_path}."
            )

        if state.index < 0:
            next_index = len(pairs) - 1 if reverse else 0
        else:
            next_index = (state.index - 1) % len(pairs) if reverse else (
                (state.index + 1) % len(pairs)
            )

        state.index = next_index
        pair = pairs[next_index]

        source_tensor = self._load_image(pair.source.path)
        output_tensor = self._load_image(pair.output.path)

        _SIGNATURE_INDEX[signature] = state.index

        print(
            "[PortraitUtils_PairedImageLoader] "
            f"{pair.display_name} ({next_index + 1}/{len(pairs)})"
        )

        return output_tensor, source_tensor, pair.source.name

    @classmethod
    def _scan_directories(
        cls,
        source_dir: Path,
        output_dir: Path,
        strip_trailing_numbers: bool,
    ) -> Tuple[List[_Pair], Tuple, Dict[str, object]]:
        source_entries = cls._gather_entries(source_dir, strip_trailing_numbers)
        output_entries = cls._gather_entries(output_dir, strip_trailing_numbers)

        source_map: Dict[str, List[_FileEntry]] = {}
        output_map: Dict[str, List[_FileEntry]] = {}
        collisions: Dict[str, List[str]] = {}

        for entry in source_entries:
            source_map.setdefault(entry.normalized_key, []).append(entry)
        for entry in output_entries:
            output_map.setdefault(entry.normalized_key, []).append(entry)

        if strip_trailing_numbers:
            for mapping in (source_map, output_map):
                for key, entries in mapping.items():
                    if len(entries) > 1:
                        existing = collisions.setdefault(key, [])
                        existing.extend(e.name for e in entries)

        for mapping in (source_map, output_map):
            for key in mapping:
                mapping[key].sort(key=lambda e: _natural_sort_key(e.name))

        shared_keys = sorted(
            set(source_map).intersection(output_map),
            key=lambda k: _natural_sort_key(
                source_map.get(k, output_map[k])[0].stem
            ),
        )

        pairs: List[_Pair] = []
        extension_mismatch: List[str] = []

        for key in shared_keys:
            src_candidates = source_map[key]
            out_candidates = output_map[key]
            src_entry, out_entry, matched_ext = cls._select_pair(
                src_candidates, out_candidates
            )
            if matched_ext is None:
                extension_mismatch.append(src_entry.stem)
            display_name = src_entry.stem or out_entry.stem
            pairs.append(_Pair(key, display_name, src_entry, out_entry))

        source_only = cls._flatten_unmatched(source_map, shared_keys)
        output_only = cls._flatten_unmatched(output_map, shared_keys)

        signature = (
            strip_trailing_numbers,
            str(source_dir),
            str(output_dir),
            tuple(sorted((e.name.lower(), e.mtime_ns, e.size) for e in source_entries)),
            tuple(sorted((e.name.lower(), e.mtime_ns, e.size) for e in output_entries)),
        )

        warnings = {
            "source_only": source_only,
            "output_only": output_only,
            "collisions": collisions,
            "extension_mismatch": extension_mismatch,
        }

        return pairs, signature, warnings

    @staticmethod
    def _gather_entries(
        directory: Path,
        strip_trailing_numbers: bool,
    ) -> List[_FileEntry]:
        entries: List[_FileEntry] = []
        with os.scandir(directory) as iterator:
            for item in iterator:
                if not item.is_file():
                    continue
                name = item.name
                ext = Path(name).suffix.lower()
                if ext not in _IMAGE_EXTENSIONS:
                    continue
                stat = item.stat()
                stem = Path(name).stem
                normalized_key = _normalize_base(stem, strip_trailing_numbers)
                entry = _FileEntry(
                    path=directory / name,
                    name=name,
                    stem=stem,
                    normalized_key=normalized_key,
                    ext=ext,
                    mtime_ns=stat.st_mtime_ns,
                    size=stat.st_size,
                )
                entries.append(entry)
        return entries

    @staticmethod
    def _select_pair(
        source_candidates: List[_FileEntry],
        output_candidates: List[_FileEntry],
    ) -> Tuple[_FileEntry, _FileEntry, str | None]:
        source_by_ext: Dict[str, List[_FileEntry]] = {}
        output_by_ext: Dict[str, List[_FileEntry]] = {}

        for entry in source_candidates:
            source_by_ext.setdefault(entry.ext, []).append(entry)
        for entry in output_candidates:
            output_by_ext.setdefault(entry.ext, []).append(entry)

        for entries in source_by_ext.values():
            entries.sort(key=lambda e: _natural_sort_key(e.name))
        for entries in output_by_ext.values():
            entries.sort(key=lambda e: _natural_sort_key(e.name))

        for ext in sorted(output_by_ext):
            if ext in source_by_ext:
                return source_by_ext[ext][0], output_by_ext[ext][0], ext

        return (
            source_candidates[0],
            output_candidates[0],
            None,
        )

    @staticmethod
    def _flatten_unmatched(
        mapping: Dict[str, List[_FileEntry]],
        matched_keys: Iterable[str],
    ) -> List[str]:
        matched = set(matched_keys)
        leftovers: List[str] = []
        for key, entries in mapping.items():
            if key in matched:
                continue
        leftovers.extend(entry.name for entry in entries)
        leftovers.sort(key=_natural_sort_key)
        return leftovers

    @staticmethod
    def _emit_warnings(
        warnings: Dict[str, object],
        source_dir: Path,
        output_dir: Path,
    ) -> None:
        source_only = warnings.get("source_only", [])
        output_only = warnings.get("output_only", [])
        collisions = warnings.get("collisions", {})
        ext_mismatch = warnings.get("extension_mismatch", [])

        def _summarize(names: List[str]) -> str:
            if len(names) <= 5:
                return ", ".join(names)
            head = ", ".join(names[:5])
            return f"{head}, … (+{len(names) - 5} more)"

        if source_only:
            print(
                "[PortraitUtils_PairedImageLoader] Unmatched source files "
                f"({len(source_only)}) in {source_dir}: {_summarize(source_only)}"
            )
        if output_only:
            print(
                "[PortraitUtils_PairedImageLoader] Unmatched output files "
                f"({len(output_only)}) in {output_dir}: {_summarize(output_only)}"
            )
        if collisions:
            for key, files in collisions.items():
                print(
                    "[PortraitUtils_PairedImageLoader] Normalised name collision "
                    f"for '{key}': {_summarize(sorted(set(files), key=_natural_sort_key))}"
                )
        if ext_mismatch:
            print(
                "[PortraitUtils_PairedImageLoader] Chosen fallback matches with differing "
                f"extensions ({len(ext_mismatch)}): {_summarize(ext_mismatch)}"
            )

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        img = node_helpers.pillow(Image.open, path)
        try:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                if "A" in img.getbands():
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")
            if img.mode == "RGBA":
                img = img.convert("RGB")
            array = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(array)[None, ...]
            return tensor
        finally:
            img.close()

    @classmethod
    def IS_CHANGED(
        cls,
        source_dir,
        output_dir,
        reverse=False,
        strip_trailing_numbers=False,
        unique_id=None,
    ):
        source_dir = _coerce_str(source_dir).strip()
        output_dir = _coerce_str(output_dir).strip()
        if isinstance(reverse, list):
            reverse = reverse[0]
        if isinstance(strip_trailing_numbers, list):
            strip_trailing_numbers = strip_trailing_numbers[0]

        reverse = bool(reverse)
        strip_trailing_numbers = bool(strip_trailing_numbers)

        digest = hashlib.sha256()
        digest.update(f"reverse={reverse}".encode("utf-8"))

        if not source_dir or not output_dir:
            digest.update(f"|missing_dirs|{source_dir}|{output_dir}|".encode("utf-8"))
            return digest.hexdigest()

        try:
            src_path = _resolve_directory(source_dir)
            out_path = _resolve_directory(output_dir)
        except Exception as exc:
            digest.update(f"|resolve_error|{exc}|".encode("utf-8"))
            return digest.hexdigest()

        if not src_path.exists() or not src_path.is_dir():
            digest.update(f"|src_missing|{src_path}|".encode("utf-8"))
            return digest.hexdigest()
        if not out_path.exists() or not out_path.is_dir():
            digest.update(f"|out_missing|{out_path}|".encode("utf-8"))
            return digest.hexdigest()

        pairs, signature, _ = cls._scan_directories(
            src_path, out_path, strip_trailing_numbers
        )

        digest.update(str(signature).encode("utf-8"))
        digest.update(f"|pairs={len(pairs)}|".encode("utf-8"))
        last_index = _SIGNATURE_INDEX.get(signature, -1)
        digest.update(f"|last_index={last_index}|".encode("utf-8"))
        return digest.hexdigest()


NODE_CLASS_MAPPINGS = {
    "PortraitUtils_PairedImageLoader": PortraitUtils_PairedImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitUtils_PairedImageLoader": "Paired Image Loader",
}
