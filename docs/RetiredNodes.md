# Retired Nodes

PortraitUtils now registers only 32 production IDs. The following 25 old IDs
are absent: a workflow using them will show missing nodes. No workflow JSON is
bundled in this repository. The historical local v5.2 workflow uses retired IDs;
the historical local `PortraitUtilsv5.3.json` still needs current editor target
labels and gate socket wiring. See the [current editor wiring](SmartPhotoPrepare.md).

| Removed ID | Current replacement |
| --- | --- |
| `FluxResolutionPrepare` | `PortraitPhotoPrepareV2` |
| `MQBBoxMin` | Subject-mask analysis in `PortraitPhotoPrepareV2` |
| `FitAspectHeadSafe` | Aspect/protection fitting in `PortraitPhotoPrepareV2` |
| `UniversalProjectConfig` | Stage-specific `Portrait*ConfigV2` nodes |
| `ExtractConfigValue` | Matching typed `Portrait*V2` reader |
| `AutoColorConfigNode` | `PortraitFinishConfigV2` |
| `OutpaintConfigNode` | `PortraitOutpaintConfigV2` |
| `QwenImageEditNativeReference` | Built-in VAE reference path in `TextEncodeQwenImageEditPlus` |
| `LoadImageCombined` | `PortraitPhotoLoaderPreview` |
| `IntelligentAutoCrop` | `PortraitAutoCropPreview` |
| `SmartPhotoPrepare` | `PortraitPhotoPrepareV2` |
| `OutpaintPaddingComputeNode` | `OutpaintPaddingComputeV2` |
| `AutoAdjustNode` | `AutoAdjustV2` |
| `StitchByMask` | `PortraitStitchPreview` |
| `SmartPhotoPrepareV2` | `PortraitPhotoPrepareV2` (four outputs) |
| `PortraitPhotoPrepareCompactV2` | `PortraitPhotoPrepareV2` (four outputs) |
| `LoadImageCombinedV2` | `PortraitPhotoLoaderPreview` |
| `IntelligentAutoCropV2` | `PortraitAutoCropPreview` |
| `AutoWBColorMatch` | `PortraitWhiteBalancePreview` |
| `StitchByMaskV2` | `PortraitStitchPreview` |
| `SimpleImageSaver` | `PortraitImageSaverPreview` |
| `ProcessScannedPhoto` | `PortraitScannedPhotoPreview` |
| `PairedImageLoader` | `PortraitPairedLoaderPreview` |
| `CropImageByMargins` | `PortraitCropImageMarginsPreview` |
| `CropMaskByMargins` | `PortraitCropMaskMarginsPreview` |

This is an interface cleanup, not removal of the shared processing engines.
Scanner-bed removal remains a distinct operation under
`PortraitScannedPhotoPreview`. The production IDs ending in `Preview` are
historical Python identifiers; their ComfyUI display names omit that suffix.
Do not substitute IDs blindly in old JSON: reconnect changed sockets and copy
widget values. Existing production IDs alone do not guarantee that a historical
workflow has current target labels or socket wiring.
