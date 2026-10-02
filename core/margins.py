"""Literal pixel crops; shared geometry for IMAGE and MASK."""
from .tensors import image_tensor, integer, mask_tensor

def _crop_box(height, width, left, top, right, bottom, multiple):
    for name, value in (("left_px", left), ("top_px", top),
                        ("right_px", right), ("bottom_px", bottom)):
        integer(value, f"Margin Crop: {name}", 0, 16384)
    integer(multiple, "Margin Crop: snap_multiple", 1, 512)
    remaining_w, remaining_h = width - left - right, height - top - bottom
    if remaining_w < 1 or remaining_h < 1:
        raise ValueError(f"Margin Crop: margins remove the entire {width}×{height} input. "
                         "Reduce left/right or top/bottom margins.")
    snapped_w = remaining_w // multiple * multiple
    snapped_h = remaining_h // multiple * multiple
    if snapped_w < 1 or snapped_h < 1:
        raise ValueError(f"Margin Crop: the remaining {remaining_w}×{remaining_h} crop is smaller "
                         f"than snap_multiple={multiple}. Use a smaller multiple or smaller margins.")
    return left, top, left + snapped_w, top + snapped_h


class ImageMarginEngine:
    def crop(self, image, left_px, top_px, right_px, bottom_px, snap_multiple=1):
        image = image_tensor(image, "Margin Crop: image")
        left, top, right, bottom = _crop_box(image.shape[1], image.shape[2],
            left_px, top_px, right_px, bottom_px, snap_multiple)
        return (image[:, top:bottom, left:right, :],)


class MaskMarginEngine:
    def crop(self, mask, left_px, top_px, right_px, bottom_px, snap_multiple=1):
        mask = mask_tensor(mask, "Margin Crop: mask")
        left, top, right, bottom = _crop_box(mask.shape[1], mask.shape[2],
            left_px, top_px, right_px, bottom_px, snap_multiple)
        return (mask[:, top:bottom, left:right],)
