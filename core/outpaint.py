"""Pure outpaint geometry; no image conversion or ComfyUI dependencies."""
from .contracts import OutpaintConfigV2, _expect
from .validation import integer


def compute_padding(width, height, config):
    config = _expect(config, OutpaintConfigV2, "outpaint_config")
    integer(width, "source_width", 1, 2**31 - 1)
    integer(height, "source_height", 1, 2**31 - 1)
    if config.mode == "Pixels":
        left, top, right, bottom = config.left_px, config.top_px, config.right_px, config.bottom_px
    else:
        horizontal = round(width * config.horizontal_percent / 100)
        vertical = round(height * config.vertical_percent / 100)
        if config.gravity in ("left", "top left", "bottom left"):
            left, right = 0, horizontal
        elif config.gravity in ("right", "top right", "bottom right"):
            left, right = horizontal, 0
        else:
            left, right = horizontal // 2, horizontal - horizontal // 2
        if config.gravity in ("top", "top left", "top right"):
            top, bottom = 0, vertical
        elif config.gravity in ("bottom", "bottom left", "bottom right"):
            top, bottom = vertical, 0
        else:
            top, bottom = vertical // 2, vertical - vertical // 2
    # Preserve the existing even-canvas rule and L/T/R/B socket ordering.
    right += (width + left + right) % 2
    bottom += (height + top + bottom) % 2
    return left, top, right, bottom
