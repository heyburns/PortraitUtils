"""CPU-only compatibility, precision, scan, and transactional I/O regressions."""

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import cv2
import numpy as np
from PIL import Image
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes")]

import PortraitUtils
from PortraitUtils.auto_straighten import ProcessScannedPhoto
from PortraitUtils.crop_utils import CropImageByMargins, CropMaskByMargins
from PortraitUtils.crop_margins_preview import PortraitCropImageMarginsPreview, PortraitCropMaskMarginsPreview
from PortraitUtils import image_saver_preview as saver
from PortraitUtils.paired_image_loader import PairedImageLoader
from PortraitUtils.paired_loader_preview import PortraitPairedLoaderPreview
from PortraitUtils.scanned_photo_preview import PortraitScannedPhotoPreview, _print_rectangle
from PortraitUtils.simple_image_saver import SimpleImageSaver
from PortraitUtils.stitch_preview import PortraitStitchPreview
from PortraitUtils.workflow_config_v2 import EditConfigV2, StitchByMaskV2


class CpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)


class PreviewContractTests(CpuTests):
    def test_separate_ids_and_exact_existing_interfaces(self):
        pairs = [("PortraitStitchPreview", PortraitStitchPreview, "StitchByMaskV2", StitchByMaskV2),
                 ("PortraitImageSaverPreview", saver.PortraitImageSaverPreview, "SimpleImageSaver", SimpleImageSaver),
                 ("PortraitScannedPhotoPreview", PortraitScannedPhotoPreview, "ProcessScannedPhoto", ProcessScannedPhoto),
                 ("PortraitPairedLoaderPreview", PortraitPairedLoaderPreview, "PairedImageLoader", PairedImageLoader),
                 ("PortraitCropImageMarginsPreview", PortraitCropImageMarginsPreview, "CropImageByMargins", CropImageByMargins),
                 ("PortraitCropMaskMarginsPreview", PortraitCropMaskMarginsPreview, "CropMaskByMargins", CropMaskByMargins)]
        for node_id, new, old_id, old in pairs:
            with self.subTest(node_id=node_id):
                self.assertIs(PortraitUtils.NODE_CLASS_MAPPINGS[node_id], new)
                self.assertNotIn(old_id, PortraitUtils.NODE_CLASS_MAPPINGS)
                self.assertNotEqual(new, old)
                old_schema, new_schema = old.INPUT_TYPES(), new.INPUT_TYPES()
                for group, fields in old_schema.items():
                    self.assertEqual(list(fields), [name for name in new_schema[group] if name in fields])
                    for name, specification in fields.items():
                        self.assertEqual(new_schema[group][name], specification)
                self.assertEqual(new.RETURN_TYPES, old.RETURN_TYPES)
                if hasattr(old, "RETURN_NAMES"):
                    self.assertEqual(new.RETURN_NAMES, old.RETURN_NAMES)

    def test_schemas_are_independent_copies(self):
        schema = PortraitStitchPreview.INPUT_TYPES()
        schema["required"]["force_size"][1]["default"] = True
        self.assertFalse(StitchByMaskV2.INPUT_TYPES()["required"]["force_size"][1]["default"])


class MarginPreviewTests(CpuTests):
    def setUp(self):
        self.image = torch.linspace(-0.1, 1.1, 2 * 25 * 30 * 4, dtype=torch.float64).reshape(2, 25, 30, 4)
        self.mask = self.image[..., 0]
        self.image_node = PortraitCropImageMarginsPreview()
        self.mask_node = PortraitCropMaskMarginsPreview()

    def test_pixel_exact_crop_keeps_dtype_alpha_and_device(self):
        result = self.image_node.crop(self.image, 2, 3, 4, 5)[0]
        torch.testing.assert_close(result, self.image[:, 3:20, 2:26], atol=0, rtol=0)
        self.assertEqual(result.dtype, self.image.dtype)
        self.assertEqual(result.device, self.image.device)
        self.assertEqual(result.shape[-1], 4)

    def test_crop_does_not_clip_or_regrade_pixels(self):
        result = self.image_node.crop(self.image, 0, 0, 0, 0)[0]
        torch.testing.assert_close(result, self.image, atol=0, rtol=0)
        self.assertLess(result.min().item(), 0)
        self.assertGreater(result.max().item(), 1)

    def test_image_and_batch_mask_use_identical_snap_geometry(self):
        image = self.image_node.crop(self.image, 2, 3, 4, 5, 4)[0]
        mask = self.mask_node.crop(self.mask, 2, 3, 4, 5, 4)[0]
        self.assertEqual(image.shape[1:3], (16, 24))
        self.assertEqual(mask.shape, (2, 16, 24))
        torch.testing.assert_close(image[..., 0], mask, atol=0, rtol=0)

    def test_mask_accepts_hw_and_bhw1(self):
        for mask in (self.mask[0], self.mask[..., None]):
            result = self.mask_node.crop(mask, 1, 2, 3, 4)[0]
            self.assertEqual(result.ndim, 3)
            self.assertEqual(result.shape[1:3], (19, 26))

    def test_single_column_mask_is_batched_not_hwc(self):
        mask = torch.arange(10.).reshape(2, 5, 1)
        result = self.mask_node.crop(mask, 0, 1, 0, 1)[0]
        self.assertEqual(result.shape, (2, 3, 1))
        torch.testing.assert_close(result, mask[:, 1:4], atol=0, rtol=0)

    def test_excessive_margins_raise_instead_of_hidden_one_pixel_crop(self):
        for node, image in ((self.image_node, self.image), (self.mask_node, self.mask)):
            with self.assertRaisesRegex(ValueError, "entire"):
                node.crop(image, 16, 0, 15, 0)

    def test_impossible_multiple_is_reported(self):
        with self.assertRaisesRegex(ValueError, "smaller than snap_multiple"):
            self.image_node.crop(self.image, 0, 0, 0, 0, 32)

    def test_negative_float_or_invalid_multiple_is_rejected(self):
        for values in ((-1, 0, 0, 0, 1), (1.2, 0, 0, 0, 1), (0, 0, 0, 0, 0)):
            with self.assertRaisesRegex(ValueError, "integer"):
                self.image_node.crop(self.image, *values)

    def test_invalid_image_and_mask_layouts_are_explained(self):
        with self.assertRaisesRegex(ValueError, "IMAGE layout"):
            self.image_node.crop(torch.zeros(1, 8, 8, 2), 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "MASK layout"):
            self.mask_node.crop(torch.zeros(1, 8, 8, 3), 0, 0, 0, 0)


class StitchPreviewTests(CpuTests):
    def setUp(self):
        self.node = PortraitStitchPreview()
        self.a, self.b = torch.zeros(2, 32, 40, 3), torch.ones(1, 32, 40, 3)
        self.mask = torch.zeros(1, 32, 40)
        self.mask[:, 8:24, 10:30] = 1
        self.config = EditConfigV2("", 0.4, 0.75, "")

    def blend(self, **changes):
        values = dict(image_a=self.a, image_b=self.b, edit_config=self.config,
                      opacity_source="stitch_opacity", mask=self.mask)
        values.update(changes)
        return self.node.blend(**values)

    def test_feather_never_overrides_opacity(self):
        output, mask = self.blend(feather_radius=5)
        self.assertEqual(mask.shape, (2, 32, 40))
        self.assertLessEqual(mask.max().item(), 0.75)
        self.assertEqual(mask[0, 16, 20].item(), 0.75)
        torch.testing.assert_close(output[..., 0], mask, atol=0, rtol=0)

    def test_feather_is_independent_of_selected_opacity(self):
        _, high = self.blend(opacity_source="stitch_opacity")
        _, low = self.blend(opacity_source="blend_opacity")
        torch.testing.assert_close(high / 0.75, low / 0.4)

    def test_zero_opacity_is_exact_identity_even_with_feather(self):
        config = EditConfigV2("", 0, 0, "")
        output, mask = self.blend(edit_config=config, feather_radius=10)
        torch.testing.assert_close(output, self.a, atol=0, rtol=0)
        self.assertEqual(mask.max().item(), 0)

    def test_soft_mask_is_not_thresholded_or_shrunk(self):
        coverage = self.mask * 0.3
        _, effective = self.blend(mask=coverage, feather_radius=5)
        self.assertGreaterEqual(effective[0, 16, 20].item(), 0.3 * 0.75 - 1e-7)
        self.assertGreater(effective[0, 7, 20].item(), 0)

    def test_zero_radius_is_literal_soft_blend(self):
        coverage = torch.linspace(0, 1, 32 * 40).reshape(1, 32, 40)
        output, mask = self.blend(mask=coverage, feather_radius=0)
        torch.testing.assert_close(mask, (coverage * 0.75).expand(2, -1, -1))
        torch.testing.assert_close(output[..., 0], mask)

    def test_inversion_happens_before_feather_and_opacity(self):
        _, inverted = self.blend(invert_mask=True, feather_radius=0)
        torch.testing.assert_close(inverted, ((1 - self.mask) * 0.75).expand(2, -1, -1))

    def test_global_bypass_ignores_mask_and_feather(self):
        output, mask = self.blend(mask=None, bypass_mask=True, feather_radius=100)
        torch.testing.assert_close(output, torch.full_like(output, 0.75))
        torch.testing.assert_close(mask, torch.full_like(mask, 0.75))

    def test_mask_2d_and_bhw1_normalize_to_standard_mask(self):
        for mask in (self.mask[0], self.mask[..., None]):
            _, result = self.blend(mask=mask)
            self.assertEqual(result.shape, (2, 32, 40))

    def test_full_masks_have_no_dark_edges(self):
        _, result = self.blend(mask=torch.ones_like(self.mask), feather_radius=100)
        torch.testing.assert_close(result, torch.full_like(result, 0.75))

    def test_empty_mask_remains_empty(self):
        output, mask = self.blend(mask=torch.zeros_like(self.mask), feather_radius=100)
        torch.testing.assert_close(output, self.a, atol=0, rtol=0)
        self.assertEqual(mask.sum().item(), 0)

    def test_force_size_resizes_images_and_mask(self):
        output, mask = self.blend(force_size=True, target_width=48, target_height=24)
        self.assertEqual(output.shape, (2, 24, 48, 3))
        self.assertEqual(mask.shape, (2, 24, 48))

    def test_rgba_hidden_colors_do_not_bleed(self):
        a = torch.tensor([1., 0., 0., 0.]).expand(1, 16, 16, 4)
        b = torch.tensor([0., 0., 1., 1.]).expand(1, 16, 16, 4)
        output, _ = self.blend(image_a=a, image_b=b, bypass_mask=True)
        torch.testing.assert_close(output[0, 0, 0], torch.tensor([0., 0., 1., 0.75]))

    def test_rgba_zero_and_full_coverage_preserve_exact_pixels(self):
        a = torch.tensor([0.7, 0.2, 0.1, 0.25], dtype=torch.float64).expand(1, 16, 16, 4)
        b = torch.tensor([0.1, 0.8, 0.3, 0.6], dtype=torch.float64).expand(1, 16, 16, 4)
        for opacity, expected in ((0, a), (1, b)):
            output, _ = self.blend(image_a=a, image_b=b, edit_config=EditConfigV2("", opacity, opacity, ""), bypass_mask=True)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_rgb_and_rgba_can_mix_without_dropping_alpha(self):
        a = torch.zeros(1, 16, 16, 3)
        b = torch.ones(1, 16, 16, 4)
        output, _ = self.blend(image_a=a, image_b=b, bypass_mask=True)
        self.assertEqual(output.shape[-1], 4)
        self.assertEqual(output[..., 3].min().item(), 1)

    def test_half_double_dtype_and_device_are_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            output, mask = self.blend(image_a=self.a.to(dtype))
            self.assertEqual(output.dtype, dtype)
            self.assertEqual(mask.dtype, dtype)
            self.assertEqual(output.device, self.a.device)

    def test_bad_config_and_opacity_source_report_errors(self):
        with self.assertRaisesRegex(TypeError, "matching PortraitUtils V2 config"):
            self.blend(edit_config=None)
        with self.assertRaisesRegex(ValueError, "opacity_source"):
            self.blend(opacity_source="inpaint_prompt")
        with self.assertRaisesRegex(ValueError, "opacity"):
            self.blend(edit_config=EditConfigV2("", 0.5, 1.2, ""))

    def test_mismatched_layout_size_batch_and_missing_mask_report_errors(self):
        for values, message in ((dict(mask=None), "connect mask"),
                                (dict(image_b=torch.ones(1, 16, 16, 3)), "size mismatch"),
                                (dict(mask=torch.ones(1, 16, 16)), "mask size"),
                                (dict(image_b=torch.ones(3, 32, 40, 3)), "image_b batch"),
                                (dict(mask=torch.ones(3, 32, 40)), "mask batch"),
                                (dict(mask=torch.full_like(self.mask, float("nan"))), "NaN")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.blend(**values)


class SaverPreviewTests(CpuTests):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="portrait-saver-preview-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.temp = self.root / "temp"
        self.output.mkdir()
        self.temp.mkdir()
        for target, value in (("folder_paths.get_output_directory", str(self.output)),
                              ("folder_paths.get_temp_directory", str(self.temp)),
                              ("PortraitUtils.image_saver_preview.args.disable_metadata", False)):
            context = patch(target, return_value=value) if target.startswith("folder_paths") else patch(target, value)
            context.start()
            self.addCleanup(context.stop)
        self.node = saver.PortraitImageSaverPreview()
        self.image = torch.tensor([0.2, 0.4, 0.6]).expand(1, 8, 10, 3)

    def save(self, **changes):
        values = dict(images=self.image, filename="photo")
        values.update(changes)
        return self.node.save(**values)

    def test_png_rounding_and_ui_descriptor(self):
        result = self.save()
        self.assertEqual(result, {"ui": {"images": [{"filename": "photo.png", "subfolder": "", "type": "output"}]}})
        with Image.open(self.output / "photo.png") as image:
            np.testing.assert_array_equal(np.array(image)[0, 0], [51, 102, 153])

    def test_png_metadata_unicode_roundtrip(self):
        prompt = {"1": {"inputs": {"prompt": "café 東京"}}}
        workflow = {"nodes": [{"id": 1}]}
        for depth in ("8-bit", "16-bit"):
            self.save(filename=depth, png_bit_depth=depth, prompt=prompt, extra_pnginfo={"workflow": workflow})
            with Image.open(self.output / f"{depth}.png") as image:
                self.assertEqual(json.loads(image.info["prompt"]), prompt)
                self.assertEqual(json.loads(image.info["workflow"]), workflow)
                image.verify()  # Validate every PNG chunk CRC, including injected text.

    def test_nonfinite_workflow_metadata_is_saved_as_null(self):
        prompt = {"1": {"inputs": {"value": float("nan"),
                                   "limits": [float("inf"), -float("inf")]}}}
        workflow = {"nodes": [{"widgets_values": [float("nan"), {"scale": float("inf")}]}]}
        for depth in ("8-bit", "16-bit"):
            with self.assertLogs("PortraitUtils.image_saver_preview", level="WARNING") as captured:
                self.save(filename=depth, png_bit_depth=depth, prompt=prompt,
                          extra_pnginfo={"workflow": workflow})
            self.assertIn("replaced 5 non-finite", captured.output[0])
            with Image.open(self.output / f"{depth}.png") as image:
                saved_prompt = json.loads(image.info["prompt"])
                saved_workflow = json.loads(image.info["workflow"])
                self.assertIsNone(saved_prompt["1"]["inputs"]["value"])
                self.assertEqual(saved_prompt["1"]["inputs"]["limits"], [None, None])
                self.assertEqual(saved_workflow["nodes"][0]["widgets_values"], [None, {"scale": None}])
                self.assertNotIn("NaN", image.info["workflow"])
                image.verify()

        with self.assertLogs("PortraitUtils.image_saver_preview", level="WARNING"):
            self.save(filename="nonfinite", file_format="JPG", prompt=prompt,
                      extra_pnginfo={"workflow": workflow})
        with Image.open(self.output / "nonfinite.jpg") as image:
            payload = json.loads(image.info["comment"])
            self.assertIsNone(payload["prompt"]["1"]["inputs"]["value"])
            self.assertIsNone(payload["extra_pnginfo"]["workflow"]["nodes"][0]["widgets_values"][0])
        self.assertTrue(math.isnan(prompt["1"]["inputs"]["value"]))
        self.assertTrue(math.isinf(workflow["nodes"][0]["widgets_values"][1]["scale"]))

    def test_metadata_disabled_and_global_override(self):
        self.save(filename="disabled", include_metadata=False, prompt={"private": "text"})
        with patch("PortraitUtils.image_saver_preview.args.disable_metadata", True):
            self.save(filename="global", prompt={"private": "text"})
        for name in ("disabled.png", "global.png"):
            with Image.open(self.output / name) as image:
                self.assertNotIn("prompt", image.info)

    def test_invalid_metadata_leaves_no_files(self):
        circular = {}; circular["self"] = circular
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.save(prompt=circular)
        self.assertFalse(list(self.output.iterdir()))

    def test_png16_keeps_low_order_rgb_samples(self):
        pixels = np.array([[[12345, 23456, 34567], [12346, 23457, 34568]]], dtype=np.uint16)
        image = torch.from_numpy(pixels.astype(np.float64) / 65535)[None]
        self.save(images=image, png_bit_depth="16-bit")
        decoded = cv2.imread(str(self.output / "photo.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.dtype, np.uint16)
        np.testing.assert_array_equal(decoded[..., ::-1], pixels)

    def test_png16_metadata_accepts_pillow10_chunk_layout(self):
        info, _ = saver._metadata(True, {"prompt": "test"}, {})
        info.chunks = [chunk[:2] for chunk in info.chunks]
        path = self.output / "pillow10.png"
        saver._png16(path, np.full((3, 4, 3), 23456, np.uint16), info)
        with Image.open(path) as image:
            self.assertEqual(json.loads(image.info["prompt"]), {"prompt": "test"})
            image.verify()

    def test_png_rgba_alpha_for_both_depths(self):
        for depth, maximum in (("8-bit", 255), ("16-bit", 65535)):
            image = torch.tensor([0.9, 0.2, 0.6, 0.25], dtype=torch.float64).expand(1, 6, 8, 4)
            self.save(images=image, filename=depth, png_bit_depth=depth)
            decoded = cv2.imread(str(self.output / f"{depth}.png"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded[0, 0, [2, 1, 0, 3]], np.floor(np.array([.9, .2, .6, .25]) * maximum + .5))

    def test_grayscale_supported_in_both_formats_and_png_depths(self):
        image = torch.full((1, 7, 9, 1), 0.5)
        for fmt in ("PNG", "JPG"):
            for depth in ("8-bit", "16-bit"):
                self.save(images=image, filename=f"{fmt}-{depth}", file_format=fmt, png_bit_depth=depth)
        self.assertEqual(len(list(self.output.iterdir())), 4)

    def test_jpeg_composites_alpha_instead_of_discarding(self):
        image = torch.tensor([1., 0., 0., 0.]).expand(1, 8, 10, 4)
        for matte, expected in (("White", 255), ("Black", 0)):
            self.save(images=image, filename=matte, file_format="JPG", jpeg_alpha_background=matte)
            with Image.open(self.output / f"{matte}.jpg") as saved:
                np.testing.assert_array_equal(np.array(saved)[0, 0], [expected] * 3)

    def test_jpeg_comment_metadata(self):
        self.save(file_format="JPG", prompt={"1": "café"}, extra_pnginfo={"workflow": {}})
        with Image.open(self.output / "photo.jpg") as image:
            self.assertEqual(json.loads(image.info["comment"])["prompt"], {"1": "café"})

    def test_large_jpeg_metadata_is_omitted_without_losing_image(self):
        with self.assertLogs("PortraitUtils.image_saver_preview", level="WARNING"):
            self.save(file_format="JPG", prompt={"text": "x" * 70000})
        with Image.open(self.output / "photo.jpg") as image:
            self.assertNotIn("comment", image.info)

    def test_unique_names_never_replace_existing_image(self):
        self.save()
        first = (self.output / "photo.png").read_bytes()
        self.save(images=torch.ones_like(self.image))
        self.assertEqual((self.output / "photo.png").read_bytes(), first)
        self.assertTrue((self.output / "photo-0001.png").is_file())

    def test_concurrent_unique_saves_publish_only_complete_images(self):
        def write(_):
            return saver.PortraitImageSaverPreview().save(self.image, filename="parallel")
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(write, range(8)))
        names = {result["ui"]["images"][0]["filename"] for result in results}
        self.assertEqual(len(names), 8)
        for name in names:
            with Image.open(self.output / name) as image:
                image.verify()
        self.assertFalse(list(self.output.glob(".portrait-save-*")))

    def test_explicit_overwrite_is_atomic(self):
        self.save(unique_filenames=False)
        with patch.object(saver.os, "replace", wraps=saver.os.replace) as replace:
            self.save(images=torch.ones_like(self.image), unique_filenames=False)
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(len(list(self.output.iterdir())), 1)
        with Image.open(self.output / "photo.png") as image:
            self.assertEqual(np.array(image).min(), 255)

    def test_failed_encoder_leaves_old_file_and_no_placeholders(self):
        self.save()
        original = (self.output / "photo.png").read_bytes()
        for unique in (False, True):
            with patch.object(saver, "_encode", side_effect=OSError("simulated encoder failure")):
                with self.assertRaisesRegex(OSError, "encoder failure"):
                    self.save(unique_filenames=unique)
            self.assertEqual((self.output / "photo.png").read_bytes(), original)
            self.assertEqual([p.name for p in self.output.iterdir()], ["photo.png"])

    def test_failed_publish_does_not_overwrite_and_cleans_temporary(self):
        with patch.object(saver.os, "link", side_effect=OSError("link unavailable")):
            with self.assertRaisesRegex(OSError, "no existing file was overwritten"):
                self.save()
        self.assertFalse(list(self.output.iterdir()))

    def test_external_preview_copies_encoded_bytes(self):
        external = self.root / "external"
        result = self.save(output_path=str(external), file_format="JPG", prompt={"text": "test"})
        descriptor = result["ui"]["images"][0]
        self.assertEqual(descriptor["type"], "temp")
        self.assertEqual((external / "photo.jpg").read_bytes(), (self.temp / descriptor["filename"]).read_bytes())

    def test_proxy_failure_reports_successful_real_save_location(self):
        external = self.root / "external"
        with patch.object(saver.shutil, "copyfileobj", side_effect=OSError("temp unavailable")):
            with self.assertRaisesRegex(OSError, "saved successfully.*photo.png"):
                self.save(output_path=str(external))
        self.assertTrue((external / "photo.png").is_file())
        self.assertFalse(list(self.temp.iterdir()))

    def test_subfolder_suffix_and_batch_descriptors(self):
        result = self.save(images=self.image.expand(2, -1, -1, -1), output_path="neat/nested", suffix="edited")
        self.assertEqual([r["filename"] for r in result["ui"]["images"]],
                         ["photo-edited-0000.png", "photo-edited-0001.png"])
        self.assertTrue(all(r["subfolder"] == "neat/nested" for r in result["ui"]["images"]))

    def test_traversal_and_relative_symlink_escape_are_rejected(self):
        (self.output / "escape").symlink_to(self.root, target_is_directory=True)
        for path in ("../outside", "escape/outside"):
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.save(output_path=path)

    def test_filename_components_cannot_create_paths(self):
        self.save(filename="../private/name", suffix="CON")
        paths = list(self.output.iterdir())
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].is_file())
        self.assertIn("_CON", paths[0].name)

    def test_nan_invalid_format_and_bad_controls_fail_before_write(self):
        for changes, message in ((dict(images=torch.full_like(self.image, float("nan"))), "NaN"),
                                 (dict(file_format="TIFF"), "select PNG"),
                                 (dict(jpeg_quality=101), "jpeg_quality"),
                                 (dict(filename=12.3), "must be text"),
                                 (dict(filename="x" * 181), "too long")):
            with self.assertRaisesRegex(ValueError, message):
                self.save(**changes)
        self.assertFalse(list(self.output.iterdir()))

    def test_empty_input_is_noop(self):
        for image in (None, [], torch.empty(0, 8, 8, 3)):
            self.assertEqual(self.save(images=image), {"ui": {"images": []}})


class PairedPreviewTests(CpuTests):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="portrait-pairs-preview-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output = self.root / "source", self.root / "output"
        self.source.mkdir(); self.output.mkdir()
        self.node = PortraitPairedLoaderPreview()

    def photo(self, path, color=(25, 75, 125), size=(7, 5)):
        with Image.new("RGB", size, color) as image:
            image.save(path)

    def pair(self, name, extension=".png", output_extension=None):
        self.photo(self.source / (name + extension))
        self.photo(self.output / (name + (output_extension or extension)), (125, 75, 25))

    def next(self, node=None, **changes):
        values = dict(source_dir=str(self.source), output_dir=str(self.output))
        values.update(changes)
        return (node or self.node).load_next_pair(**values)

    def test_natural_sort_wrap_and_output_order(self):
        for name in ("photo10", "photo1", "photo2"):
            self.pair(name)
        self.assertEqual([self.next()[2] for _ in range(4)], ["photo1", "photo2", "photo10", "photo1"])
        output, source, _ = self.next()
        torch.testing.assert_close(source[0, 0, 0], torch.tensor([25, 75, 125]) / 255)
        torch.testing.assert_close(output[0, 0, 0], torch.tensor([125, 75, 25]) / 255)

    def test_two_node_instances_never_share_cursor_even_with_same_id(self):
        self.pair("photo1"); self.pair("photo2")
        first, second = PortraitPairedLoaderPreview(), PortraitPairedLoaderPreview()
        self.assertEqual(self.next(first, unique_id="same")[2], "photo1")
        self.assertEqual(self.next(first, unique_id="same")[2], "photo2")
        self.assertEqual(self.next(second, unique_id="same")[2], "photo1")

    def test_reverse_starts_last_and_can_change_direction(self):
        for name in ("photo1", "photo2", "photo3"):
            self.pair(name)
        self.assertEqual(self.next(reverse=True)[2], "photo3")
        self.assertEqual(self.next(reverse=True)[2], "photo2")
        self.assertEqual(self.next(reverse=False)[2], "photo3")

    def test_new_earlier_file_does_not_shift_current_position(self):
        self.pair("photo2"); self.pair("photo3")
        self.assertEqual(self.next()[2], "photo2")
        self.pair("photo1")
        self.assertEqual(self.next()[2], "photo3")

    def test_switching_folder_pairs_keeps_independent_positions(self):
        self.pair("photo1"); self.pair("photo2")
        source2, output2 = self.root / "source2", self.root / "output2"
        source2.mkdir(); output2.mkdir()
        for name in ("other1", "other2"):
            self.photo(source2 / f"{name}.png"); self.photo(output2 / f"{name}.png")
        self.assertEqual(self.next()[2], "photo1")
        self.assertEqual(self.next(source_dir=str(source2), output_dir=str(output2))[2], "other1")
        self.assertEqual(self.next()[2], "photo2")

    def test_failed_second_decode_does_not_advance(self):
        self.pair("photo1"); self.pair("photo2"); self.pair("photo3")
        self.assertEqual(self.next()[2], "photo1")
        (self.output / "photo2.png").write_bytes(b"broken image")
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "cannot load"):
                self.next()
        self.photo(self.output / "photo2.png")
        self.assertEqual(self.next()[2], "photo2")

    def test_cache_invalidation_is_always_non_idempotent(self):
        for _ in range(3):
            self.assertTrue(math.isnan(self.node.IS_CHANGED(str(self.source), str(self.output))))

    def test_pair_extensions_may_differ(self):
        self.pair("photo", output_extension=".jpg")
        self.assertEqual(self.next()[2], "photo")

    def test_normalization_is_case_insensitive_and_only_strips_copy_suffixes(self):
        self.photo(self.source / "Photo2024 (2).png")
        self.photo(self.output / "PHOTO2024.png")
        self.assertEqual(self.next(strip_trailing_numbers=True)[2], "Photo2024 (2)")

    def test_ambiguous_pairs_are_rejected_instead_of_guessed(self):
        self.pair("photo")
        self.photo(self.source / "photo (2).png")
        with self.assertRaisesRegex(ValueError, "ambiguous basename"):
            self.next(strip_trailing_numbers=True)

    def test_unique_common_extension_disambiguates(self):
        self.pair("photo")
        self.photo(self.source / "photo.jpg")
        self.assertEqual(self.next()[2], "photo")

    def test_unmatched_files_are_reported_once(self):
        self.pair("photo1"); self.pair("photo2")
        self.photo(self.source / "orphan.png")
        with self.assertLogs("PortraitUtils.paired_loader_preview", level="WARNING"):
            self.next()
        with patch("PortraitUtils.paired_loader_preview._LOG.warning") as warn:
            self.next()
        warn.assert_not_called()

    def test_transparency_and_exif_use_preview_decoder(self):
        exif = Image.Exif(); exif[274] = 6
        with Image.new("RGBA", (7, 5), (255, 0, 0, 0)) as image:
            image.save(self.source / "photo.png", exif=exif)
            image.save(self.output / "photo.png", exif=exif)
        output, source, _ = self.next()
        self.assertEqual(source.shape, (1, 7, 5, 3))
        self.assertEqual(output.min().item(), 1)

    def test_16bit_pair_preserves_precision(self):
        pixels = np.full((3, 4, 3), 23456, np.uint16)
        pixels[0, 1] += 1
        for directory in (self.source, self.output):
            cv2.imwrite(str(directory / "photo.png"), pixels)
        output, source, _ = self.next()
        self.assertGreater((source[0, 0, 1] - source[0, 0, 0]).min().item(), 0)
        torch.testing.assert_close(output, source, atol=0, rtol=0)

    def test_dead_nodes_do_not_leave_global_cursor_state(self):
        self.pair("photo")
        node = PortraitPairedLoaderPreview()
        self.next(node)
        reference = weakref.ref(node)
        del node; gc.collect()
        self.assertIsNone(reference())

    def test_missing_dirs_and_no_pairs_are_helpful(self):
        with self.assertRaisesRegex(ValueError, "source_dir is required"):
            self.next(source_dir="")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.next(output_dir=str(self.root / "missing"))
        with self.assertRaisesRegex(ValueError, "no matching basenames"):
            self.next()
        self.assertIsNot(self.node.VALIDATE_INPUTS(source_dir=""), True)


class ScannedPreviewTests(CpuTests):
    def setUp(self):
        self.node = PortraitScannedPhotoPreview()
        # White scanner bed, distinguishable white paper, dark image, asymmetric
        # lower Polaroid frame. These are separate physical boundaries.
        self.array = np.ones((240, 200, 3), np.float64)
        self.array[30:210, 25:175] = 0.94
        self.array[45:185, 40:160] = [0.24, 0.38, 0.59]
        self.image = torch.from_numpy(self.array)[None]

    def process(self, image=None, **changes):
        values = dict(image=self.image if image is None else image, straighten=False,
                      crop_mode="Scanner Bed Only", padding=0, threshold=0.8)
        values.update(changes)
        return self.node.process(**values)[0]

    def test_crop_modes_preserve_scanner_bed_vs_print_border_distinction(self):
        outer = self.process()
        inner = self.process(crop_mode="Inner Photo Frame")
        self.assertGreater(outer.shape[1], inner.shape[1])
        self.assertGreater(outer.shape[2], inner.shape[2])
        self.assertTrue(bool((outer == .94).any()))
        self.assertEqual(inner.shape, (1, 140, 120, 3))
        torch.testing.assert_close(inner, self.image[:, 45:185, 40:160], atol=0, rtol=0)

    def test_disabled_processing_is_exact_identity(self):
        result = self.process(crop_mode="None")
        self.assertIs(result, self.image)

    def test_crop_only_never_quantizes_full_precision_pixels(self):
        array = self.array.copy()
        array[45:185, 40:160, 0] += np.linspace(0, 1e-4, 140 * 120).reshape(140, 120)
        image = torch.from_numpy(array)[None]
        result = self.process(image, crop_mode="Inner Photo Frame")
        torch.testing.assert_close(result, image[:, 45:185, 40:160], atol=0, rtol=0)
        self.assertGreater(torch.unique(result[..., 0]).numel(), 1000)

    def test_bright_inner_photo_uses_seam_detection_fallback(self):
        array = self.array.copy()
        array[45:185, 40:160] = .88
        result = self.process(torch.from_numpy(array)[None], crop_mode="Inner Photo Frame")
        self.assertEqual(result.shape, (1, 140, 120, 3))
        torch.testing.assert_close(result, torch.from_numpy(array[45:185, 40:160])[None], atol=0, rtol=0)

    def test_signed_padding_expands_or_contracts_final_boundary(self):
        original = self.process(crop_mode="Inner Photo Frame")
        expanded = self.process(crop_mode="Inner Photo Frame", padding=3)
        contracted = self.process(crop_mode="Inner Photo Frame", padding=-3)
        self.assertEqual(expanded.shape[1:3], (original.shape[1] + 6, original.shape[2] + 6))
        self.assertEqual(contracted.shape[1:3], (original.shape[1] - 6, original.shape[2] - 6))

    def test_excessive_negative_padding_explains_error(self):
        with self.assertRaisesRegex(ValueError, "negative padding removes the entire"):
            self.process(crop_mode="Inner Photo Frame", padding=-100)

    def test_blank_scan_has_no_arbitrary_crop_or_rotation(self):
        image = torch.ones(1, 40, 50, 3, dtype=torch.float64)
        for mode in ("Scanner Bed Only", "Inner Photo Frame", "None"):
            result = self.process(image, straighten=True, crop_mode=mode)
            torch.testing.assert_close(result, image, atol=0, rtol=0)

    def test_isolated_speck_does_not_define_print(self):
        array = np.ones((200, 240, 3), np.float64)
        array[10:12, 10:12] = 0
        image = torch.from_numpy(array)[None]
        result = self.process(image, straighten=True)
        torch.testing.assert_close(result, image, atol=0, rtol=0)

    def test_outside_dust_does_not_expand_dominant_print(self):
        array = self.array.copy(); array[10:12, 10:12] = 0
        result = self.process(torch.from_numpy(array)[None])
        self.assertLess(result.shape[1], 190)
        self.assertLess(result.shape[2], 160)

    def test_straightening_uses_outer_print_and_preserves_orientation(self):
        array = np.ones((400, 360, 3), np.float32)
        array[80:320, 90:270] = .94
        array[100:280, 110:250] = [.24, .38, .59]
        for angle in (-13, 7, 21):
            with self.subTest(angle=angle):
                matrix = cv2.getRotationMatrix2D((180, 200), angle, 1)
                rotated = cv2.warpAffine(array, matrix, (360, 400), borderValue=(1, 1, 1))
                detection = _print_rectangle(rotated, .8)
                self.assertIsNotNone(detection)
                self.assertAlmostEqual(detection[1], -angle, delta=.6)
                result = self.process(torch.from_numpy(rotated)[None], straighten=True, crop_mode="Inner Photo Frame")
                self.assertGreater(result.shape[1], result.shape[2])
                self.assertLess(abs(result.shape[1] - 180), 12)
                self.assertLess(abs(result.shape[2] - 140), 12)

    def test_dtype_grayscale_and_alpha_layouts_are_preserved(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            for channels in (1, 3, 4):
                image = self.image.to(dtype)
                if channels == 1:
                    image = image[..., :1]
                elif channels == 4:
                    image = torch.cat((image, torch.ones_like(image[..., :1])), -1)
                result = self.process(image, crop_mode="Inner Photo Frame")
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.device, image.device)
                self.assertEqual(result.shape[-1], channels)

    def test_batch_uses_explicit_white_padding_not_color_resizing(self):
        second = np.ones_like(self.array)
        second[50:190, 40:160] = .94
        second[65:165, 55:145] = [.24, .38, .59]
        batch = torch.cat((self.image, torch.from_numpy(second)[None]))
        result = self.process(batch, crop_mode="Inner Photo Frame")
        self.assertEqual(result.shape, (2, 140, 120, 3))
        torch.testing.assert_close(result[1, :100, :90], torch.from_numpy(second[65:165, 55:145]), atol=0, rtol=0)
        self.assertEqual(result[1, 100:].min().item(), 1)

    def test_invalid_mode_threshold_padding_and_pixels_are_reported(self):
        for values, message in ((dict(crop_mode="subject"), "choose Inner"),
                                (dict(threshold=float("nan")), "threshold"),
                                (dict(padding=-501), "padding"),
                                (dict(image=self.image * 255), "normalized")):
            with self.assertRaisesRegex(ValueError, message):
                self.process(**values)


if __name__ == "__main__":
    unittest.main()
