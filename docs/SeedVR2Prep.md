# SeedVR2Prep

## Overview
`SeedVR2Prep` prepares images for SeedVR2 and other VRAM-sensitive pipelines by enforcing even dimensions and optionally converting tensors to FP16 on the GPU. It ensures data is in `[B, H, W, 3]` format, repeating channels as necessary.

## Inputs
- `image` (`IMAGE`): Input tensor. Non-tensor inputs are converted to `torch.Tensor` before processing.
- `ensure_even_dims` (`BOOLEAN`, default `True`): When enabled, trims or pads by repeating edge pixels so both width and height are even.
- `to_fp16_on_gpu` (`BOOLEAN`, default `True`): If CUDA is available, moves the tensor to GPU and converts it to float16.

## Outputs
- `image` (`IMAGE`): Sanitised tensor with even dimensions and optional FP16 GPU residency.

## Processing Notes
- Channel counts greater than 3 are truncated to RGB; single-channel inputs are repeated across RGB.
- Padding when width/height equals 1 duplicates the final column/row so dimensions never drop to zero.
- FP16 conversion only occurs when CUDA is available; otherwise the data remains on CPU in float32.

## Tips
- Place this node right before VRAM-heavy samplers to cut down on memory usage and avoid odd-dimension errors.
- Disable `to_fp16_on_gpu` if a downstream node expects CPU tensors or full precision.*** End Patch
