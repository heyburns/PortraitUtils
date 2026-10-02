"""Float Lanczos-3, antialiased and separable, without PIL/uint8 round trips."""
import math
import torch

# Target block size for temporary gathered samples. A single column can exceed
# this for extremely large inputs; no image or coefficient cache survives a run.
_GATHER_ELEMENTS = 4 * 1024 * 1024


def _axis(samples, length, axis):
    source_length = samples.shape[axis]
    if source_length == length:
        return samples
    step = source_length / length
    filter_scale = max(1.0, step)  # Widen the filter when reducing resolution.
    radius = 3 * filter_scale
    taps = math.ceil(2 * radius) + 1
    positions = (torch.arange(length, device=samples.device, dtype=samples.dtype) + .5) * step - .5
    first = torch.floor(positions - radius).to(dtype=torch.int64)
    indices = first[:, None] + torch.arange(taps, device=samples.device)[None, :]
    distance = (positions[:, None] - indices.to(dtype=samples.dtype)) / filter_scale
    weights = torch.sinc(distance) * torch.sinc(distance / 3)
    valid = (distance.abs() < 3) & (indices >= 0) & (indices < source_length)
    weights = torch.where(valid, weights, 0)
    weights = weights / weights.sum(-1, keepdim=True)
    indices = indices.clamp(0, source_length - 1)
    moved = samples.movedim(axis, -1)
    output = torch.empty(*moved.shape[:-1], length, device=samples.device, dtype=samples.dtype)
    leading = math.prod(moved.shape[:-1])
    chunk = max(1, min(length, _GATHER_ELEMENTS // max(1, leading * taps)))
    for start in range(0, length, chunk):
        stop = min(length, start + chunk)
        selected = moved.index_select(-1, indices[start:stop].reshape(-1))
        selected = selected.reshape(*moved.shape[:-1], stop - start, taps)
        output[..., start:stop] = (selected * weights[start:stop]).sum(-1)
    return output.movedim(-1, axis)


@torch.no_grad()
def lanczos_resize(samples, width, height):
    """Resize NCHW float32/float64 tensors on their current device."""
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4 or samples.dtype not in (torch.float32, torch.float64):
        raise ValueError("Lanczos: expected an NCHW float32/float64 working tensor.")
    if min(samples.shape) < 1 or isinstance(width, bool) or isinstance(height, bool) or \
            not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1:
        raise ValueError("Lanczos: source and output dimensions must be positive integers.")
    # Reduce intermediate canvas size when one dimension is being reduced.
    if height / samples.shape[-2] < width / samples.shape[-1]:
        return _axis(_axis(samples, height, -2), width, -1)
    return _axis(_axis(samples, width, -1), height, -2)
