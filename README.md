# PortraitUtils Node Suite

PortraitUtils is a focused collection of ComfyUI nodes designed for pipelines using real potographic images of people, where image sizes are not standardized as they are in text-to-image generation. The suite streamlines prep work such as cropping, resolution targeting, metadata management, and side-by-side QA so you can concentrate on creative iteration instead of utility wiring. These nodes should work well on any type of image, but some of them are designed for and tested on images of humans, particularly the smart cropping nodes.

## Node Index

Each entry links to a short, task-oriented reference page (coming soon) for deeper usage notes and parameter breakdowns.

### Image Adjustments & Preparation
- `AutoAdjustNode` + `AutoColorConfigNode` - Global tone balancing with reusable toggle bundles. [Docs](docs/AutoAdjustSuite.md)
  <div align="center"><img src="docs/screenshots/auto_adjust_node.png" alt="AutoAdjust suite screenshot" width="500" /></div>
- `AutoWBColorMatch` - White-balances and colour matches against a reference image using several algorithms. [Docs](docs/AutoWBColorMatch.md)
  <div align="center"><img src="docs/screenshots/auto_wb_color_match.png" alt="AutoWBColorMatch screenshot" width="300" /></div>
- `FluxResolutionPrepare` - Crops and resizes to the nearest Flux-friendly resolution, with optional pre-upscale. [Docs](docs/FluxResolutionPrepare.md)
  <div align="center"><img src="docs/screenshots/flux_resolution_prepare.png" alt="FluxResolutionPrepare screenshot" width="300" /></div>
- `ImageMegapixelSelector` - Chooses image sizes by target megapixels to match downstream model limits. [Docs](docs/ImageMegapixelSelector.md)
  <div align="center"><img src="docs/screenshots/image_megapixel_selector.png" alt="ImageMegapixelSelector screenshot" width="300" /></div>

### Cropping, Framing, and Stitching
- `AutoCropBorders` - Removes uniform borders using adaptive fuzzy detection. GPU acceleration is currently broken, so leave that off for now. [Docs](docs/AutoCropBorders.md)
  <div align="center"><img src="docs/screenshots/auto_crop_borders.png" alt="AutoCropBorders screenshot" width="300" /></div>
- `CropImageByMargins` + `CropMaskByMargins` - Aligned margin croppers for RGB images and masks. [Docs](docs/CropByMarginsSuite.md)
  <div align="center"><img src="docs/screenshots/crop_image_by_margins.png" alt="Crop by margins suite screenshot" width="300" /></div>
- `FitAspectHeadSafe` + `MQBBoxMin` - Subject-aware boxing with headroom-aware aspect fitting. [Docs](docs/FitAspectSuite.md)
  <div align="center"><img src="docs/screenshots/fit_aspect_head_safe.png" alt="FitAspect suite screenshot" width="500" /></div>
- `StitchByMask` - Composites foreground and background images using mask-driven blending. [Docs](docs/StitchByMask.md)
  <div align="center"><img src="docs/screenshots/stitch_by_mask.png" alt="StitchByMask screenshot" width="300" /></div>

### Prompting & Workflow Configuration
- `MultiPromptNode` - Cycles through prompt variants with per-slot weights. [Docs](docs/MultiPromptNode.md)
  <div align="center"><img src="docs/screenshots/multi_prompt_node.png" alt="MultiPromptNode screenshot" width="300" /></div>
- `WorkflowConfig` - Centralized key/value store for workflow-wide settings. [Docs](docs/WorkflowConfig.md)
  <div align="center"><img src="docs/screenshots/workflow_config.png" alt="WorkflowConfig screenshot" width="300" /></div>
- `OutpaintConfigNode` + `OutpaintPaddingComputeNode` - Preference capture plus per-edge padding conversion. [Docs](docs/OutpaintSuite.md)
  <div align="center"><img src="docs/screenshots/outpaint_config_node.png" alt="Outpaint suite screenshot" width="500" /></div>

### I/O & Utility Helpers
- `LoadImageCombined` - Hybrid single/batch image loader with auto-advance support. [Docs](docs/LoadImageCombined.md)
  <div align="center"><img src="docs/screenshots/load_image_combined.png" alt="LoadImageCombined screenshot" width="200" /></div>
- `PairedImageLoader` - Loads synchronized source/output pairs for A/B comparison workflows. [Docs](docs/PairedImageLoader.md)
  <div align="center"><img src="docs/screenshots/portraitutils_paired_image_loader.png" alt="PairedImageLoader screenshot" width="300" /></div>
- `ComparisonGate` - Gates comparison outputs until two populated images are available. [Docs](docs/ComparisonGate.md)
- `SimpleImageSaver` - Minimal saver with PNG/JPG, metadata toggle, and filename suffix controls. [Docs](docs/SimpleImageSaver.md)
  <div align="center"><img src="docs/screenshots/simple_image_saver.png" alt="SimpleImageSaver screenshot" width="300" /></div>
- `FilenameAppendSuffix` - Adds suffixes to filenames while respecting multi-extension cases. [Docs](docs/FilenameAppendSuffix.md)
  <div align="center"><img src="docs/screenshots/filename_append_suffix.png" alt="FilenameAppendSuffix screenshot" width="300" /></div>
- `GpuSyncCacheClear` - Forces a CUDA sync and empties selected caches to avoid VRAM buildup. [Docs](docs/GpuSyncCacheClear.md)
  <div align="center"><img src="docs/screenshots/gpu_sync_cache_clear.png" alt="GpuSyncCacheClear screenshot" width="300" /></div>
- `SeedVR2Prep` - Convenience helper for SeedVR2/VRAM-heavy pipelines (cache priming & cleanup). [Docs](docs/SeedVR2Prep.md)
  <div align="center"><img src="docs/screenshots/seedvr2_prep.png" alt="SeedVR2Prep screenshot" width="300" /></div>

## License

This project is released under the [MIT License](LICENSE).
