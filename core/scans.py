"""Scanner-bed and print-frame processing, distinct from general border removal."""
import logging
import math
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from .borders import _bounds
from .tensors import image_tensor, normalized, finite_float, integer
from .validation import boolean

_LOG = logging.getLogger(__name__)

_ANALYSIS_SIDE = 2048

def _rgb(array):
    if array.shape[-1] == 1:
        return np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        return array[..., :3] * array[..., 3:4] + 1 - array[..., 3:4]
    return array

def _sample(rgb):
    height, width = rgb.shape[:2]
    scale = min(1, _ANALYSIS_SIDE / max(height, width))
    if scale == 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)

def _print_rectangle(rgb, threshold):
    """Require a dominant, rectangular component against a bright corner bed."""
    sample = _sample(rgb)
    height, width = sample.shape[:2]
    if min(height, width) < 8:
        return None
    corner = max(1, min(32, round(min(height, width) * 0.035)))
    corners = np.concatenate([sample[:corner, :corner].reshape(-1, 3),
                              sample[:corner, -corner:].reshape(-1, 3),
                              sample[-corner:, :corner].reshape(-1, 3),
                              sample[-corner:, -corner:].reshape(-1, 3)])
    background = np.median(corners, axis=0)
    if float(background @ np.array([0.2126, 0.7152, 0.0722])) < threshold:
        return None
    noise = float(np.median(np.max(np.abs(corners - background), axis=-1)))
    tolerance = max(2 / 255, noise * 6)
    if np.mean(np.max(np.abs(corners - background), axis=-1) <= tolerance) < 0.85:
        return None
    foreground = (np.max(np.abs(sample - background), axis=-1) > tolerance).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < max(20, height * width * 0.01):
        return None
    center, size, _ = rectangle = cv2.minAreaRect(contour)
    if min(size) < 3 or area / max(1, size[0] * size[1]) < 0.88 or max(size) / min(size) > 8:
        return None
    box = cv2.boxPoints(rectangle)
    if np.any(box.min(0) < 1) or np.any(box.max(0) > np.array([width - 2, height - 2])):
        return None  # No surrounding bed / clipped print: do not guess the angle.
    # Expand half a sample pixel to include the physical boundary, then map
    # sample pixel centers back to original coordinates.
    vectors = box - np.array(center)
    box += vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1) * 0.5
    factors = np.array([rgb.shape[1] / width, rgb.shape[0] / height])
    box = (box + 0.5) * factors - 0.5
    edges = np.roll(box, -1, axis=0) - box
    edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    angle = math.degrees(math.atan2(edge[1], edge[0]))
    angle -= round(angle / 90) * 90
    return box, angle

def _rotation(array, angle):
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1)
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transformed = corners @ matrix[:, :2].T + matrix[:, 2]
    minimum, maximum = transformed.min(0), transformed.max(0)
    matrix[:, 2] -= minimum
    out_width, out_height = np.ceil(maximum - minimum + 1).astype(int)
    if array.shape[-1] == 4:
        array = np.concatenate((array[..., :3] * array[..., 3:4], array[..., 3:4]), -1)
    rotated = cv2.warpAffine(array, matrix, (int(out_width), int(out_height)),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(1, 1, 1, 1))
    if rotated.ndim == 2:
        rotated = rotated[..., None]
    if array.shape[-1] == 4:
        alpha = rotated[..., 3:4]
        rgb = np.divide(rotated[..., :3], alpha, out=np.zeros_like(rotated[..., :3]), where=alpha > 1e-8)
        rotated = np.concatenate((rgb, alpha), -1)
    return np.clip(rotated, 0, 1), matrix

def _box(points, width, height):
    # Box points are pixel-center coordinates. Inclusive maxima retain the edge.
    lower = np.floor(points.min(0)).astype(int)
    upper = np.ceil(points.max(0) + 1).astype(int)
    return max(0, lower[0]), max(0, lower[1]), min(width, upper[0]), min(height, upper[1])

def _inner_frame(rgb, threshold):
    """Corroborate a rectangular inner photograph despite rotated-edge noise.

    Merely finding a dark subject is insufficient: require a nearly filled
    rectangle, bright surrounding paper, and a broad seam on all four sides.
    """
    sample = _sample(rgb)
    height, width = sample.shape[:2]
    if min(height, width) < 8:
        return None
    luminance = sample @ np.array([0.2126, 0.7152, 0.0722], dtype=sample.dtype)
    dark = (luminance < threshold).astype(np.uint8)
    radius = max(1, min(7, round(min(height, width) * 0.01)))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((2 * radius + 1,) * 2, np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    left, top, cw, ch = cv2.boundingRect(contour)
    right, bottom = left + cw, top + ch
    if (left < 1 or top < 1 or right >= width or bottom >= height or
            min(cw, ch) < 4 or cv2.contourArea(contour) / (cw * ch) < 0.88):
        return None
    outside = (sample[top:bottom, left - 1], sample[top - 1, left:right],
               sample[top:bottom, right], sample[bottom, left:right])
    inside = (sample[top:bottom, left:min(right, left + 3)].swapaxes(0, 1),
              sample[top:min(bottom, top + 3), left:right],
              sample[top:bottom, max(left, right - 3):right].swapaxes(0, 1),
              sample[max(top, bottom - 3):bottom, left:right])
    for paper, edge in zip(outside, inside):
        bright = paper @ np.array([0.2126, 0.7152, 0.0722], dtype=sample.dtype)
        seam = np.max(np.abs(edge - paper[None]), axis=(0, 2))
        if np.mean(bright >= threshold) < 0.90 or np.mean(seam > 0.035) < 0.65:
            return None
    sx, sy = rgb.shape[1] / width, rgb.shape[0] / height
    return math.floor(left * sx), math.floor(top * sy), math.ceil(right * sx), math.ceil(bottom * sy)


class ScannedPhotoEngine:
    @torch.no_grad()
    def process(self, image, straighten=True, crop_mode="Inner Photo Frame", padding=0, threshold=0.8):
        boolean(straighten, "Scanned Photo preview.straighten")
        image = normalized(image_tensor(image, "Scanned Photo preview: image"), "Scanned Photo preview: image")
        if crop_mode not in ("Inner Photo Frame", "Scanner Bed Only", "None"):
            raise ValueError("Scanned Photo preview: choose Inner Photo Frame, Scanner Bed Only, or None.")
        integer(padding, "Scanned Photo preview: padding", -500, 500)
        threshold = finite_float(threshold, "Scanned Photo preview: threshold", 0, 1)
        if not straighten and crop_mode == "None":
            return (image,)
        frames = []
        for frame in image:
            dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
            array = frame.detach().to(device="cpu", dtype=dtype).numpy()
            detection = _print_rectangle(_rgb(array), threshold)
            matrix = None
            if straighten and detection is not None and abs(detection[1]) >= 0.05:
                array, matrix = _rotation(array, detection[1])
            height, width = array.shape[:2]
            left, top, right, bottom = 0, 0, width, height
            if crop_mode != "None" and detection is not None:
                points = detection[0]
                if matrix is not None:
                    points = points @ matrix[:, :2].T + matrix[:, 2]
                left, top, right, bottom = _box(points, width, height)
            if crop_mode == "Inner Photo Frame":
                rgb = _rgb(array[top:bottom, left:right])
                inner = _inner_frame(rgb, threshold)
                if inner is not None:
                    il, it, ir, ib = inner
                    left, top, right, bottom = left + il, top + it, left + ir, top + ib
                else:
                    sample = _sample(rgb)
                    config = SimpleNamespace(autocrop_strip_bottom_banner=False, autocrop_detect_borders=True,
                                             autocrop_fuzz_tolerance=min(0.05, max(0.008, 1 - threshold)),
                                             autocrop_edge_uniformity=0.995)
                    trims = _bounds(sample, config, "Conservative")
                    dl, dt, dr, db = (math.floor(trim * scale) for trim, scale in zip(trims,
                                     (rgb.shape[1] / sample.shape[1], rgb.shape[0] / sample.shape[0],
                                      rgb.shape[1] / sample.shape[1], rgb.shape[0] / sample.shape[0])))
                    left, top, right, bottom = left + dl, top + dt, right - dr, bottom - db
            if crop_mode != "None":
                found = (left, top, right, bottom) != (0, 0, width, height)
                if found:
                    left, top = max(0, left - padding), max(0, top - padding)
                    right, bottom = min(width, right + padding), min(height, bottom + padding)
                    if left >= right or top >= bottom:
                        raise ValueError("Scanned Photo preview: negative padding removes the entire detected print. "
                                         "Increase padding toward zero.")
                    array = array[top:bottom, left:right]
                else:
                    _LOG.info("Scanned Photo preview: no confident %s boundary; image retained.", crop_mode)
            frames.append(torch.from_numpy(np.ascontiguousarray(array)).to(device=image.device, dtype=image.dtype))
        height, width = max(frame.shape[0] for frame in frames), max(frame.shape[1] for frame in frames)
        if len(frames) == 1:
            return (frames[0][None],)
        # Same legacy batch policy, now explicit: white padding at bottom/right.
        output = torch.ones((len(frames), height, width, image.shape[-1]), device=image.device, dtype=image.dtype)
        for index, frame in enumerate(frames):
            output[index, :frame.shape[0], :frame.shape[1]] = frame
        return (output,)
