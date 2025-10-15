"""
Gate a source image and up to three comparison inputs before forwarding them.

ComfyUI passes ``None`` through empty sockets. This node inspects every input,
selects an appropriate pair of populated images, and only emits them when both
slots are ready. Otherwise, it mirrors the behaviour of an unplugged viewer
socket by returning ``None`` for both outputs.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Any


def _collect_present(values: Iterable[Any]) -> List[Any]:
    """
    Return the sequence of values that look like real images.

    ComfyUI represents disconnected sockets as ``None`` and occasionally wraps
    outputs in small containers. We filter out anything that is clearly empty so
    the gate never forwards placeholders into downstream nodes.
    """
    present: List[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            if all(item is None for item in value):
                continue
        present.append(value)
    return present


class ComparisonGate:
    """
    Emit two images only when at least two inputs are populated.

    Designed for comparison viewers: the first populated comparison becomes the
    "final image", while the designated source socket feeds the secondary output
    (with a fallback to the next comparison if the source is missing). If fewer
    than two images are available the node returns ``None`` for both sockets, so
    downstream widgets remain blank.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Mark every socket as optional so that workflows can omit connections
        # without triggering validation errors. ComfyUI fills unspecified inputs
        # with ``None`` which this node explicitly understands.
        return {
            "optional": {
                "source_image": ("IMAGE", {"default": None}),
                "image_a": ("IMAGE", {"default": None}),
                "image_b": ("IMAGE", {"default": None}),
                "image_c": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "source_image")
    FUNCTION = "forward_images"
    CATEGORY = "PortraitUtils"

    def forward_images(
        self,
        source_image=None,
        image_a=None,
        image_b=None,
        image_c=None,
    ) -> Tuple[Any, Any]:
        # Gather the inputs in priority order to ensure predictable selection.
        total_inputs = (source_image, image_a, image_b, image_c)
        populated = _collect_present(total_inputs)
        if len(populated) < 2:
            return None, None

        # The first comparison with data becomes the final output. The source
        # socket remains reserved for the secondary output to keep viewer labels
        # aligned with the user's expectations.
        candidate_finals = _collect_present((image_a, image_b, image_c))
        final_image = candidate_finals[0] if candidate_finals else None
        source_output = source_image

        # If the source socket is empty but multiple comparisons are available,
        # fall back to the next comparison so we still deliver two images.
        if source_output is None and len(candidate_finals) > 1:
            source_output = candidate_finals[1]

        # Avoid emitting half-populated pairs; downstream nodes expect either a
        # full tuple of tensors or pure ``None`` values.
        if final_image is None or source_output is None:
            return None, None

        return final_image, source_output


NODE_CLASS_MAPPINGS = {
    "ComparisonGate": ComparisonGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComparisonGate": "Comparison Gate",
}
