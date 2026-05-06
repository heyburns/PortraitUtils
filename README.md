# PortraitUtils Node Suite

PortraitUtils is a suite of helper nodes for ComfyUI workflows that start with real photographs where resolutions are not standardized like they are in text-to-image generation. Nonstandard or odd resolutions may cause problems with inpainting or other workflow functions, and there may be borders or other dead space in photographs that needs to be trimmed. The suite covers cropping, fit logic, megapixel targeting, colour matching, batch I/O, and comparison QA so you can spend less time wiring utilities and more time iterating.

---

## What You Get

- **Human-aware framing** – turn noisy masks into reliable subject boxes and aspect-friendly crops.
- **Resolution helpers** – snap images to Flux whitelist sizes or match a megapixel budget without manual math.
- **Colour and tone tools** – mimic Photoshop-style auto adjustments across entire branches.
- **Workflow glue** – loaders, savers, filename utilities, and comparison gates that keep larger graphs tidy.

Every node has a matching guide under `docs/` with plain-language control explanations, quick examples, and troubleshooting tips.

---

## Install / Update

1. Copy or clone `PortraitUtils` into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI (hard refresh from the manager works too).
3. Keep your existing workflows—node settings live inside the graphs you already saved.

Updating is as simple as replacing this folder and restarting.

---

## Node Guide

### Image Adjustments & Preparation

- **Auto Adjust (Levels/Tone/Color)**  
  Balance levels, tone, and colour. Mirrors Photoshop’s Auto Levels / Auto Tone / Auto Color. [Guide](docs/AutoAdjustSuite.md)  
  <div align="center"><img src="docs/screenshots/auto_adjust_node.png" alt="Auto Adjust screenshot" width="500" /></div>

- **AutoColor Config**  
  Shares the toggle settings for Auto Adjust across your workflow. [Guide](docs/AutoAdjustSuite.md)  
  <div align="center"><img src="docs/screenshots/auto_color_config_node.png" alt="AutoColor Config screenshot" width="500" /></div>

- **Auto White-Balance + Color Match**  
  Match white balance to a reference frame using quick presets. [Guide](docs/AutoWBColorMatch.md)  
  <div align="center"><img src="docs/screenshots/auto_wb_color_match.png" alt="AutoWBColorMatch screenshot" width="500" /></div>

- **Flux Resolution Prepare**  
  Crop and resize to Flux-friendly dimensions with optional pre-upscale. [Guide](docs/FluxResolutionPrepare.md)  
  <div align="center"><img src="docs/screenshots/flux_resolution_prepare.png" alt="FluxResolutionPrepare screenshot" width="500" /></div>

### Cropping & Framing

- **Intelligent AutoCrop (GPU)**  
  Automatically snipe away ugly black letterboxing or solid borders using your graphics card. [Guide](docs/IntelligentAutoCrop.md)  
  <div align="center"><img src="docs/screenshots/PLACEHOLDER_INTELLIGENT_AUTOCROP.png" alt="Intelligent AutoCrop screenshot" width="500" /></div>

- **Process Scanned Photo**  
  Automatically detect, straighten, and crop messy scanned photographs. [Guide](docs/ProcessScannedPhoto.md)  
  <div align="center"><img src="docs/screenshots/PLACEHOLDER_PROCESS_SCANNED.png" alt="Process Scanned Photo screenshot" width="500" /></div>

- **Crop by Margins (Image)**  
  Apply precise pixel padding to your RGB images. [Guide](docs/CropByMarginsSuite.md)  
  <div align="center"><img src="docs/screenshots/crop_image_by_margins.png" alt="Crop Image by Margins screenshot" width="500" /></div>

- **Crop by Margins (Mask)**  
  Apply precise pixel padding to your masks. [Guide](docs/CropByMarginsSuite.md)  
  <div align="center"><img src="docs/screenshots/crop_image_by_margins.png" alt="Crop Mask by Margins screenshot" width="500" /></div>

- **MQ BBox (MIN)**  
  Derive subject-aware crops based on noisy masks using smart quantiles. [Guide](docs/FitAspectSuite.md)  
  <div align="center"><img src="docs/screenshots/mq_bbox_min.png" alt="MQ BBox screenshot" width="500" /></div>

- **Fit Aspect (Head-Safe) - Closest AR + Tight Cover**  
  Snap to aspect ratios while protecting headroom and footroom. [Guide](docs/FitAspectSuite.md)  
  <div align="center"><img src="docs/screenshots/fit_aspect_head_safe.png" alt="Fit Aspect screenshot" width="500" /></div>

- **Outpaint Padding Compute**  
  Translates your outpainting preferences into exact pixel padding values. [Guide](docs/OutpaintSuite.md)  
  <div align="center"><img src="docs/screenshots/outpaint_padding_compute_node.png" alt="Outpaint Compute screenshot" width="500" /></div>

- **Stitch Two Images by Mask**  
  Blend layers together using mask-guided falloffs for seamless composites. [Guide](docs/StitchByMask.md)  
  <div align="center"><img src="docs/screenshots/stitch_by_mask.png" alt="Stitch by Mask screenshot" width="500" /></div>

### Prompting & Configuration

- **Universal Project Config**  
  The ultimate master control board. Replaces messy spaghetti wiring with a single settings bundle! [Guide](docs/UniversalConfigSuite.md)  
  <div align="center"><img src="docs/screenshots/PLACEHOLDER_UNIVERSAL_CONFIG.png" alt="Universal Config screenshot" width="500" /></div>

- **Extract Config Value**  
  The wireless receiver for your Universal Config bundle. [Guide](docs/UniversalConfigSuite.md)  
  <div align="center"><img src="docs/screenshots/PLACEHOLDER_EXTRACT_CONFIG.png" alt="Extract Config screenshot" width="500" /></div>

- **Outpaint Config**  
  Store outpaint gravity and percentage preferences securely. [Guide](docs/OutpaintSuite.md)  
  <div align="center"><img src="docs/screenshots/outpaint_config_node.png" alt="Outpaint Config screenshot" width="500" /></div>

### I/O & Utilities

- **Load Image (Combined)**  
  A slick hybrid loader for grabbing single images or plowing through entire batches automatically. [Guide](docs/LoadImageCombined.md)  
  <div align="center"><img src="docs/screenshots/load_image_combined.png" alt="Load Image Combined screenshot" width="500" /></div>

- **Paired Image Loader**  
  Pull matched before/after pairs directly from your folders for quick comparison passes. [Guide](docs/PairedImageLoader.md)  
  <div align="center"><img src="docs/screenshots/portraitutils_paired_image_loader.png" alt="Paired Image Loader screenshot" width="500" /></div>

- **Simple Image Saver**  
  Save your PNGs or JPGs quickly with custom suffixes and full metadata support. [Guide](docs/SimpleImageSaver.md)  
  <div align="center"><img src="docs/screenshots/simple_image_saver.png" alt="Simple Image Saver screenshot" width="500" /></div>

- **Filename: Append Suffix**  
  Add suffixes to your filenames safely without breaking tricky multi-part extensions. [Guide](docs/FilenameAppendSuffix.md)  
  <div align="center"><img src="docs/screenshots/filename_append_suffix.png" alt="Filename Append Suffix screenshot" width="500" /></div>

- **Comparison Gate**  
  Pauses the workflow until two incoming images are ready, ensuring they pop out at the exact same time. [Guide](docs/ComparisonGate.md)  
  <div align="center"><img src="docs/screenshots/comparison_gate.png" alt="Comparison Gate screenshot" width="500" /></div>

---

## Tips Before You Wire

- **Shared toggles** – drive multiple `Auto Adjust` blocks with one `AutoColor Config` when branches need the exact same look.
- **Debug strings** – cropping utilities emit detailed notes; pipe them into log widgets while tuning.
- **JSON bundles** – `Universal Project Config` emits a single string you can unpack anywhere with `Extract Config Value` to sync settings everywhere.

---

## License

GPLv3 – see [LICENSE](LICENSE).
