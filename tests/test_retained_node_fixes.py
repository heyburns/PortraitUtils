"""Behavioral regressions for existing PortraitUtils node IDs after engine fixes."""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from PIL import Image
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes")]

from PortraitUtils.auto_color_match import AutoWBColorMatch, rgb_to_lab
from PortraitUtils.auto_straighten import ProcessScannedPhoto
from PortraitUtils.crop_utils import CropImageByMargins, CropMaskByMargins
from PortraitUtils.paired_image_loader import PairedImageLoader
from PortraitUtils.simple_image_saver import SimpleImageSaver
from PortraitUtils.workflow_config_v2 import (
    CropConfigV2, EditConfigV2, IntelligentAutoCropV2, LoadImageCombinedV2, StitchByMaskV2,
)


class RetainedNodeFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_white_balance_has_neutral_lab_and_highlights(self):
        white = torch.ones((1, 4, 5, 3), dtype=torch.float32)
        lab = rgb_to_lab(white)
        self.assertAlmostEqual(lab[..., 0].mean().item(), 100, places=2)
        self.assertLess(lab[..., 1:].abs().max().item(), 0.01)
        result = AutoWBColorMatch().run(white, white, method="wb_highlight")[0]
        torch.testing.assert_close(result, white, atol=1e-6, rtol=0)

    def test_configured_stitch_keeps_opacity_and_standard_mask(self):
        first = torch.zeros((1, 15, 15, 4))
        second = torch.ones_like(first)
        first[..., 3] = 1
        mask = torch.zeros((1, 15, 15))
        mask[:, 7, 7] = 1
        config = EditConfigV2("", 0.4, 0.4, "")
        image, processed = StitchByMaskV2().blend(
            first, second, config, "stitch_opacity", feather_radius=3, mask=mask,
        )
        self.assertEqual(image.shape, first.shape)
        self.assertEqual(processed.shape, mask.shape)
        self.assertLessEqual(processed.max().item(), 0.4 + 1e-6)
        self.assertEqual(image.shape[-1], 4)

    def test_configured_autocrop_uses_one_exact_box_for_batch(self):
        content = torch.linspace(0.2, 0.9, 2 * 32 * 40 * 3).reshape(2, 32, 40, 3)
        image = torch.zeros((2, 42, 54, 3))
        image[:, 5:37, 7:47] = content
        config = CropConfigV2(False, True, 0.07, 0.85, 0, False,
                              "Auto (preserve detail)", "Auto", "person")
        result = IntelligentAutoCropV2().run(image, config)
        self.assertEqual(result[1:], (7, 5, 7, 5, True))
        torch.testing.assert_close(result[0], content, atol=0, rtol=0)

    def test_combined_loader_cursor_is_per_instance_and_failed_decode_retries(self):
        with tempfile.TemporaryDirectory(prefix="retained-loader-") as directory:
            root = Path(directory)
            for name, color in (("a.png", "red"), ("b.png", "blue")):
                with Image.new("RGB", (4, 3), color) as photo:
                    photo.save(root / name)
            inputs = dict(mode="Batch", input_dir=str(root), output_dir="",
                          pattern="*.png", strip_trailing_numbers=False,
                          repeat_last=False, image="")
            first, second = LoadImageCombinedV2(), LoadImageCombinedV2()
            self.assertEqual(first.load_image_v2(**inputs)[1].filename_no_ext, "a")
            self.assertEqual(second.load_image_v2(**inputs)[1].filename_no_ext, "a")
            self.assertEqual(first.load_image_v2(**inputs)[1].filename_no_ext, "b")
            with patch.object(second, "_load_pil", side_effect=OSError("decode failed")):
                with self.assertRaisesRegex(OSError, "decode failed"):
                    second.load_image_v2(**inputs)
            self.assertEqual(second.load_image_v2(**inputs)[1].filename_no_ext, "b")
            self.assertTrue(math.isnan(LoadImageCombinedV2.IS_CHANGED(**inputs)))

    def test_combined_loader_detects_same_size_same_timestamp_replacement(self):
        with tempfile.TemporaryDirectory(prefix="retained-fingerprint-") as directory:
            path = Path(directory) / "same.bmp"
            with Image.new("RGB", (4, 3), "red") as photo:
                photo.save(path)
            before = path.stat()
            kwargs = dict(mode="Single", input_dir="", output_dir="", pattern="*",
                          strip_trailing_numbers=False, repeat_last=False, image="same.bmp")
            with patch("folder_paths.get_annotated_filepath", return_value=str(path)):
                first = LoadImageCombinedV2.IS_CHANGED(**kwargs)
                with Image.new("RGB", (4, 3), "blue") as photo:
                    photo.save(path)
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                second = LoadImageCombinedV2.IS_CHANGED(**kwargs)
            self.assertNotEqual(first, second)

    def test_paired_loader_failed_second_decode_retries_same_pair(self):
        with tempfile.TemporaryDirectory(prefix="retained-pair-") as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            source.mkdir()
            output.mkdir()
            for folder in (source, output):
                for name in ("a.png", "b.png"):
                    with Image.new("RGB", (4, 3), "red") as photo:
                        photo.save(folder / name)
            node = PairedImageLoader()
            inputs = dict(source_dir=str(source), output_dir=str(output),
                          unique_id=str(uuid.uuid4()))
            decode = node._load_image
            def fail_second(path):
                if path == output / "a.png":
                    raise OSError("second decode failed")
                return decode(path)
            with patch.object(node, "_load_image", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "second decode failed"):
                    node.load_next_pair(**inputs)
            self.assertEqual(node.load_next_pair(**inputs)[2], "a")
            self.assertTrue(math.isnan(PairedImageLoader.IS_CHANGED(**inputs)))

    def test_existing_margin_nodes_preserve_pixels_and_batch_masks(self):
        image = torch.linspace(-0.2, 1.2, 2 * 8 * 9 * 4, dtype=torch.float64).reshape(2, 8, 9, 4)
        mask = image[..., 0]
        cropped = CropImageByMargins().crop(image, 1, 2, 3, 1)[0]
        cropped_mask = CropMaskByMargins().crop(mask, 1, 2, 3, 1)[0]
        torch.testing.assert_close(cropped, image[:, 2:7, 1:6], atol=0, rtol=0)
        torch.testing.assert_close(cropped_mask, mask[:, 2:7, 1:6], atol=0, rtol=0)
        self.assertEqual(cropped_mask.shape, (2, 5, 5))

    def test_existing_scan_node_noop_keeps_full_precision(self):
        image = torch.full((1, 11, 13, 4), 0.123456789, dtype=torch.float64)
        result = ProcessScannedPhoto().process(image, False, "None", 0, 0.8)[0]
        self.assertIs(result, image)

    def test_existing_saver_does_not_publish_failed_encodes(self):
        with tempfile.TemporaryDirectory(prefix="retained-saver-") as directory:
            output = Path(directory) / "output"
            temp = Path(directory) / "temp"
            output.mkdir()
            temp.mkdir()
            inputs = dict(images=torch.full((1, 5, 7, 3), 0.5),
                          output_path="", filename="photo", suffix="",
                          file_format="PNG", jpeg_quality=95,
                          include_metadata=False, unique_filenames=True)
            with patch("folder_paths.get_output_directory", return_value=str(output)), \
                 patch("folder_paths.get_temp_directory", return_value=str(temp)), \
                 patch("PortraitUtils.core.export._encode", side_effect=OSError("encode failed")):
                with self.assertRaisesRegex(OSError, "encode failed"):
                    SimpleImageSaver().save(**inputs)
            self.assertEqual(list(output.iterdir()), [])
            with patch("folder_paths.get_output_directory", return_value=str(output)), \
                 patch("folder_paths.get_temp_directory", return_value=str(temp)):
                SimpleImageSaver().save(**inputs)
            self.assertTrue((output / "photo.png").is_file())

    def test_existing_saver_preserves_workflow_and_sanitizes_nonfinite_metadata(self):
        with tempfile.TemporaryDirectory(prefix="retained-metadata-") as directory:
            output = Path(directory) / "output"
            temp = Path(directory) / "temp"
            output.mkdir()
            temp.mkdir()
            prompt = {"12": {"inputs": {"seed": float("nan")}}}
            workflow = {"nodes": [{"id": 12, "type": "SaveImage"}]}
            inputs = dict(images=torch.full((1, 5, 7, 3), 0.5),
                          output_path="", filename="workflow", suffix="",
                          file_format="PNG", jpeg_quality=95,
                          include_metadata=True, unique_filenames=True,
                          prompt=prompt, extra_pnginfo={"workflow": workflow})
            with patch("folder_paths.get_output_directory", return_value=str(output)), \
                 patch("folder_paths.get_temp_directory", return_value=str(temp)), \
                 patch("PortraitUtils.simple_image_saver.args.disable_metadata", False):
                SimpleImageSaver().save(**inputs)
            with Image.open(output / "workflow.png") as saved:
                self.assertIsNone(json.loads(saved.info["prompt"])["12"]["inputs"]["seed"])
                self.assertEqual(json.loads(saved.info["workflow"]), workflow)
            self.assertTrue(math.isnan(prompt["12"]["inputs"]["seed"]))


if __name__ == "__main__":
    unittest.main()
