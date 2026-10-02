"""Independent, transactional before/after folder sequencing for 2.0."""

from copy import deepcopy
import logging

from .paired_image_loader import PairedImageLoader
from .core.photo_io import _decode, COLOR_POLICIES, ALPHA_BACKGROUNDS

from .core.pairing import (_directory, _entries, _pairs)
from .core.state import TransactionalCursor
from .core.validation import boolean


_LOG = logging.getLogger(__name__)


class PortraitPairedLoaderPreview:
    RETURN_TYPES = PairedImageLoader.RETURN_TYPES
    RETURN_NAMES = PairedImageLoader.RETURN_NAMES
    FUNCTION = "load_next_pair"
    CATEGORY = "PortraitUtils/IO"
    NOT_IDEMPOTENT = True

    def __init__(self):
        self._cursor = TransactionalCursor()
        self._positions = self._cursor.positions
        self._lock = self._cursor.lock

    @classmethod
    def INPUT_TYPES(cls):
        schema = deepcopy(PairedImageLoader.INPUT_TYPES())
        schema["optional"] = {
            "color_management": (COLOR_POLICIES.copy(), {"default": "Convert to sRGB"}),
            "alpha_background": (ALPHA_BACKGROUNDS.copy(), {"default": "White"}),
        }
        return schema

    def load_next_pair(self, source_dir, output_dir, reverse=False, strip_trailing_numbers=False,
                       unique_id=None, color_management="Convert to sRGB", alpha_background="White"):
        boolean(reverse, "Paired Loader.reverse")
        boolean(strip_trailing_numbers, "Paired Loader.strip_trailing_numbers")
        source_dir = _directory(source_dir, "source_dir")
        output_dir = _directory(output_dir, "output_dir")
        selection = (str(source_dir), str(output_dir), bool(strip_trailing_numbers))
        with self._lock:
            pairs, unmatched = _pairs(source_dir, output_dir, strip_trailing_numbers)
            keys = [pair[0] for pair in pairs]
            previous, old_unmatched = self._positions.get(selection, (None, None))
            selected = self._cursor.next_item(keys, previous, reverse=reverse)
            index = keys.index(selected)
            key, source, output = pairs[index]
            # Decode both before committing. A failed second image must not skip
            # the pair or silently deliver a before/after from different files.
            source_image = _decode(str(source), color_management, alpha_background)
            output_image = _decode(str(output), color_management, alpha_background)
            if unmatched != old_unmatched and any(unmatched):
                _LOG.warning("Paired Loader preview: unmatched source/output basenames: %s / %s",
                             unmatched[0][:10], unmatched[1][:10])
            self._cursor.commit(selection, (key, unmatched))
            _LOG.info("Paired Loader preview: %s (%d/%d)", source.name, index + 1, len(pairs))
            return output_image, source_image, source.stem

    @classmethod
    def IS_CHANGED(cls, source_dir, output_dir, reverse=False, strip_trailing_numbers=False,
                   unique_id=None, color_management="Convert to sRGB", alpha_background="White"):
        # Auto-advance must execute on every queue, even after cycling back to a
        # previous pair. No shared signature cursor and no timestamp-only cache.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, source_dir=None, output_dir=None):
        try:
            for label, value in (("source_dir", source_dir), ("output_dir", output_dir)):
                if value is not None:
                    _directory(value, label)
            return True
        except (OSError, ValueError) as exc:
            return str(exc)


NODE_CLASS_MAPPINGS = {"PortraitPairedLoaderPreview": PortraitPairedLoaderPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitPairedLoaderPreview": "Paired Image Loader"}
