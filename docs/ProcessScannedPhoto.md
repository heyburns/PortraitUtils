# Process Scanned Photo

`PortraitScannedPhotoPreview` is a narrow utility for scanned physical photographs. A
scanner's white bed/background is distinct from a print's own border, such as a
Polaroid frame. This node can straighten a scan and select which boundary to crop.
It is retained separately from Intelligent AutoCrop and Smart Photo Prepare.

## Inputs

- `image`: Scanned photograph.
- `straighten`: Estimate print rotation and rotate the scan before cropping.
- `crop_mode`:
  - **Scanner Bed Only**: Target the outer print boundary, retaining the print's
    own border where the detector can distinguish it from the white bed.
  - **Inner Photo Frame**: Target the image region inside the print's border,
    removing the scanner bed and the print border.
  - **None**: Skip cropping; straightening can still be enabled.
- `padding`: Pixels to retain around the detected crop. Negative values crop
  further inward.
- `threshold`: Brightness threshold used in scan-boundary detection.

## Output

One processed `image`. No subject mask, protection mask, or editor-resolution
selection is involved.

Detection uses brightness, edges, and contours rather than a semantic print
model. A white print border that is indistinguishable from a white scanner bed
can be ambiguous; inspect the result and adjust threshold/padding.

The production node uses the 2.0 scan engine, preserving floating-point pixels
and the scanner-bed/inner-frame distinction. The old `ProcessScannedPhoto`
ID has been removed.
