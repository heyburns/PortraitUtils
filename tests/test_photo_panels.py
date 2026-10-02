"""Panel detection, original-pixel crops, ambiguous-layout fallback, and chooser lists."""
import ast
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT.parent)]

import PortraitUtils
from PortraitUtils.core.panels import PanelBox, PanelOptions, split_photo_panels
from PortraitUtils.photo_panels import PhotoPanelSplitV2


class PhotoPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rng = np.random.default_rng(1209)

    def photo(self, height, width, low=.2, high=.8):
        return torch.from_numpy(self.rng.uniform(low, high, (1, height, width, 3)).astype(np.float32))

    def pair(self, axis="x", color=1, gap=6, height=150, widths=(100, 170)):
        first, second = self.photo(height, widths[0]), self.photo(height, widths[1])
        gutter = torch.empty(1, height, gap, 3)
        gutter[:] = torch.as_tensor(color)
        image = torch.cat((first, gutter, second), dim=2)
        if axis == "y":
            image = image.transpose(1, 2)
        return image

    def staggered_collage(self):
        image = torch.empty(1, 200, 270, 3)
        columns = (
            (0, 70, (0, 70, 130, 200), ((.15, .20, .25), (.30, .40, .50), (.20, .30, .10))),
            (70, 190, (0, 50, 150, 200), ((.70, .30, .20), (.50, .80, .20), (.80, .50, .60))),
            (190, 270, (0, 70, 130, 200), ((.10, .70, .40), (.30, .10, .80), (.85, .80, .10))),
        )
        for x0, x1, ys, colors in columns:
            for (y0, y1), color in zip(zip(ys, ys[1:]), colors):
                xx = torch.linspace(0, 1, x1 - x0)[None, :, None]
                yy = torch.linspace(0, 1, y1 - y0)[:, None, None]
                image[0, y0:y1, x0:x1] = torch.tensor(color) + .08 * xx + .06 * yy
        # Two dividers vanish or shift for roughly a quarter of their length.
        image[0, 90:135, 70] = image[0, 90:135, 69]
        image[0, 30:80, 190] = image[0, 30:80, 189]
        expected = {PanelBox(x0, y0, x1 - x0, y1 - y0)
                    for x0, x1, ys, _ in columns for y0, y1 in zip(ys, ys[1:])}
        return image, expected

    def assert_source_crops(self, image, result):
        self.assertEqual(len(result.images), len(result.boxes))
        for box, crop in zip(result.boxes, result.images):
            torch.testing.assert_close(crop, image[:, box.y:box.y + box.height, box.x:box.x + box.width], atol=0, rtol=0)
            self.assertEqual(crop.untyped_storage().data_ptr(), image.untyped_storage().data_ptr())
            self.assertEqual(crop.dtype, image.dtype)
            self.assertEqual(crop.device, image.device)

    def test_new_registry_and_two_list_aware_outputs(self):
        self.assertIs(PortraitUtils.NODE_CLASS_MAPPINGS["PortraitPhotoPanelSplitV2"], PhotoPanelSplitV2)
        self.assertEqual(PhotoPanelSplitV2.RETURN_TYPES, ("IMAGE", "STRING"))
        self.assertEqual(PhotoPanelSplitV2.OUTPUT_IS_LIST, (True, False))
        schema = PhotoPanelSplitV2.INPUT_TYPES()
        self.assertIn("Collage (partial seams)", schema["required"]["detection_mode"][0])
        images, report = PhotoPanelSplitV2().split(self.pair())
        self.assertIsInstance(images, list)
        self.assertEqual(json.loads(report)["panel_count"], 2)

    def test_unequal_side_by_side_panels_match_example_layout(self):
        image = self.pair(height=768, widths=(357, 561), gap=7)
        result = split_photo_panels(image)
        self.assertEqual(result.boxes, (PanelBox(0, 0, 357, 768), PanelBox(364, 0, 561, 768)))
        self.assert_source_crops(image, result)

    def test_black_white_gray_and_colored_divider_gutters(self):
        for color in (0, 1, .5, [.12, .30, .70]):
            with self.subTest(color=color):
                image = self.pair(color=color)
                result = split_photo_panels(image)
                self.assertEqual(result.boxes, (PanelBox(0, 0, 100, 150), PanelBox(106, 0, 170, 150)))
                self.assert_source_crops(image, result)

    def test_horizontal_panels_are_supported(self):
        image = self.pair(axis="y")
        result = split_photo_panels(image)
        self.assertEqual(result.boxes, (PanelBox(0, 0, 150, 100), PanelBox(0, 106, 150, 170)))
        self.assert_source_crops(image, result)

    def test_grid_and_reading_order(self):
        row1, row2 = self.pair(), self.pair()
        image = torch.cat((row1, torch.ones(1, 5, row1.shape[2], 3), row2), dim=1)
        result = split_photo_panels(image)
        self.assertEqual(result.boxes, (PanelBox(0, 0, 100, 150), PanelBox(106, 0, 170, 150),
                                        PanelBox(0, 155, 100, 150), PanelBox(106, 155, 170, 150)))
        self.assert_source_crops(image, result)

    def test_mixed_layout_recurses_within_panels(self):
        row = self.pair()
        image = torch.cat((row, torch.ones(1, 5, row.shape[2], 3), self.photo(90, row.shape[2])), dim=1)
        result = split_photo_panels(image)
        self.assertEqual(result.boxes, (PanelBox(0, 0, 100, 150), PanelBox(106, 0, 170, 150), PanelBox(0, 155, 276, 90)))
        self.assert_source_crops(image, result)

    def test_thin_separator_survives_large_input_analysis_sampling(self):
        image = self.pair(gap=1, height=2200, widths=(350, 510))
        result = split_photo_panels(image)
        self.assertEqual(result.boxes, (PanelBox(0, 0, 350, 2200), PanelBox(351, 0, 510, 2200)))

    def test_noisy_scan_gutter_with_small_text_interruption(self):
        image = self.pair()
        image[:, :, 100:106] = torch.from_numpy(self.rng.uniform(.965, 1, (1, 150, 6, 3)).astype(np.float32))
        image[:, :2, 102:104] = .2
        result = split_photo_panels(image)
        self.assertEqual(len(result.images), 2)
        self.assert_source_crops(image, result)

    def test_jpeg_divider_is_detected_without_changing_decoded_pixels(self):
        original = self.pair(gap=10)
        buffer = BytesIO()
        Image.fromarray((original[0].numpy() * 255).astype(np.uint8)).save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        with Image.open(buffer) as jpeg:
            image = torch.from_numpy(np.array(jpeg, dtype=np.float32) / 255).unsqueeze(0)
        result = split_photo_panels(image)
        self.assertEqual(len(result.images), 2)
        self.assert_source_crops(image, result)

    def test_single_photo_and_smooth_gradients_are_retained(self):
        gradient = torch.linspace(.2, .8, 200).reshape(1, 200, 1, 1).expand(1, 200, 180, 3)
        for image in (self.photo(160, 200), gradient, gradient.transpose(1, 2), torch.ones(1, 50, 80, 3)):
            result = split_photo_panels(image)
            self.assertEqual(result.boxes, (PanelBox(0, 0, image.shape[2], image.shape[1]),))
            self.assertIs(result.images[0], image)
            self.assertTrue(result.warnings)

    def test_outer_borders_are_not_mistaken_for_separate_photos(self):
        source = self.photo(160, 200)
        image = torch.ones(1, 170, 212, 3)
        image[:, 5:165, 6:206] = source
        result = split_photo_panels(image)
        self.assertIs(result.images[0], image)
        self.assertEqual(len(result.images), 1)
        for border in (10, 20):
            canvas = torch.ones(1, 160 + 2 * border, 200 + 2 * border, 3)
            canvas[:, border:border + 160, border:border + 200] = source
            collage = split_photo_panels(
                canvas, PanelOptions(detection_mode="Collage (partial seams)"))
            self.assertEqual(collage.boxes,
                             (PanelBox(0, 0, canvas.shape[2], canvas.shape[1]),))
            self.assertIs(collage.images[0], canvas)

    def test_borderless_hard_seams_are_opt_in(self):
        image = torch.cat((self.photo(150, 100, .15, .25), self.photo(150, 170, .75, .85)), dim=2)
        self.assertEqual(len(split_photo_panels(image).images), 1)
        result = split_photo_panels(image, PanelOptions(detection_mode="Gutters + strong seams"))
        self.assertEqual(result.boxes, (PanelBox(0, 0, 100, 150), PanelBox(100, 0, 170, 150)))
        self.assert_source_crops(image, result)
        self.assertTrue(any("scene boundaries" in w for w in result.warnings))

    def test_collage_mode_splits_staggered_borderless_panels(self):
        image, expected = self.staggered_collage()
        self.assertEqual(len(split_photo_panels(image).boxes), 1)
        self.assertEqual(len(split_photo_panels(
            image, PanelOptions(detection_mode="Gutters + strong seams")).boxes), 1)
        result = split_photo_panels(
            image, PanelOptions(detection_mode="Collage (partial seams)", min_panel_area_percent=1))
        self.assertEqual(set(result.boxes), expected)
        self.assertEqual(len(result.images), 9)
        self.assertEqual(len(result.separators), 8)
        self.assertTrue(all(s.kind == "partial seam" for s in result.separators))
        self.assert_source_crops(image, result)
        self.assertTrue(any("verify the thumbnails" in w for w in result.warnings))

    def test_collage_mode_does_not_invent_seams_in_single_photos(self):
        gradient = torch.linspace(.2, .8, 200).reshape(1, 200, 1, 1).expand(1, 200, 180, 3)
        for image in (self.photo(160, 200), gradient):
            for sensitivity in ("Conservative", "Balanced", "Sensitive"):
                with self.subTest(sensitivity=sensitivity, shape=tuple(image.shape)):
                    result = split_photo_panels(
                        image, PanelOptions(detection_mode="Collage (partial seams)",
                                            sensitivity=sensitivity))
                    self.assertEqual(result.boxes, (PanelBox(0, 0, image.shape[2], image.shape[1]),))
                    self.assertIs(result.images[0], image)

    def test_all_supported_dtypes_and_channels_are_retained(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for channels in (1, 3, 4):
                with self.subTest(dtype=dtype, channels=channels):
                    source = self.pair().to(dtype)
                    if channels == 1:
                        source = source[..., :1]
                    elif channels == 4:
                        source = torch.cat((source, torch.ones_like(source[..., :1])), -1)
                    result = split_photo_panels(source)
                    self.assertEqual(len(result.images), 2)
                    self.assert_source_crops(source, result)

    def test_alpha_is_only_composited_for_analysis(self):
        image = self.pair()
        alpha = torch.ones_like(image[..., :1])
        image[:, :, 100:106] = .3
        alpha[:, :, 100:106] = 0
        source = torch.cat((image, alpha), -1)
        result = split_photo_panels(source)
        self.assertEqual(len(result.images), 2)
        self.assert_source_crops(source, result)

    def test_max_panel_limit_keeps_remaining_source_regions(self):
        row = self.pair()
        image = torch.cat((row, torch.ones(1, 5, row.shape[2], 3), self.pair()), dim=1)
        result = split_photo_panels(image, PanelOptions(max_panels=2))
        self.assertEqual(len(result.images), 2)
        self.assertTrue(any("kept unsplit" in w for w in result.warnings))
        self.assert_source_crops(image, result)

    def test_minimum_panel_area_can_suppress_small_automatic_tiles(self):
        image = self.pair(widths=(15, 200))
        self.assertEqual(len(split_photo_panels(image, PanelOptions(min_panel_area_percent=10)).images), 1)
        self.assertEqual(len(split_photo_panels(image, PanelOptions(min_panel_area_percent=1)).images), 2)

    def test_manual_pixels_percentages_and_gutter_ranges_override_auto(self):
        image = self.photo(100, 200)
        for spec in ("50%", "100", "100,100"):
            result = split_photo_panels(image, vertical_splits=spec)
            self.assertEqual(result.boxes, (PanelBox(0, 0, 100, 100), PanelBox(100, 0, 100, 100)))
            self.assertTrue(result.manual)
            self.assert_source_crops(image, result)
        result = split_photo_panels(image, vertical_splits="98:102", horizontal_splits="50%")
        self.assertEqual(result.boxes, (PanelBox(0, 0, 98, 50), PanelBox(102, 0, 98, 50),
                                        PanelBox(0, 50, 98, 50), PanelBox(102, 50, 98, 50)))
        self.assert_source_crops(image, result)

    def test_per_column_manual_rows_match_staggered_layout(self):
        image, expected = self.staggered_collage()
        cuts = "70,130 | 50,150 | 70,130"
        result = split_photo_panels(
            image, PanelOptions(detection_mode="Manual splits"),
            vertical_splits="70,190", column_horizontal_splits=cuts)
        self.assertTrue(result.manual)
        self.assertEqual(set(result.boxes), expected)
        self.assert_source_crops(image, result)
        self.assertEqual(len([s for s in result.separators if s.axis == "y"]), 6)
        self.assertEqual({s.parent.x for s in result.separators if s.axis == "y"},
                         {0, 70, 190})
        images, report = PhotoPanelSplitV2().split(
            image, vertical_splits="70,190", column_horizontal_splits=cuts)
        self.assertEqual(len(images), 9)
        self.assertEqual(json.loads(report)["panel_count"], 9)

    def test_per_column_manual_rows_validate_specification(self):
        image, _ = self.staggered_collage()
        with self.assertRaisesRegex(ValueError, "requires vertical_splits"):
            split_photo_panels(image, column_horizontal_splits="70,130")
        with self.assertRaisesRegex(ValueError, "not both"):
            split_photo_panels(image, vertical_splits="70,190",
                               horizontal_splits="50", column_horizontal_splits="70,130 | 50,150 | 70,130")
        with self.assertRaisesRegex(ValueError, "one '\\|' section per"):
            split_photo_panels(image, vertical_splits="70,190",
                               column_horizontal_splits="70,130 | 50,150")
        with self.assertRaisesRegex(ValueError, "column 2"):
            split_photo_panels(image, vertical_splits="70,190",
                               column_horizontal_splits="70,130 | bad | 70,130")
        with self.assertRaisesRegex(ValueError, "increase max_panels"):
            split_photo_panels(image, PanelOptions(max_panels=2),
                               vertical_splits="70,190",
                               column_horizontal_splits="70,130 | 50,150 | 70,130")

    def test_invalid_manual_splits_have_useful_errors(self):
        image = self.photo(100, 200)
        for spec in ("face", "0", "100%", "101%", "-1", "nan", "5.5", "80:50", "10:20,15:30", "10:20:30", "0:200"):
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "vertical_splits|manual splits"):
                split_photo_panels(image, vertical_splits=spec)
        with self.assertRaisesRegex(ValueError, "requires vertical_splits"):
            split_photo_panels(image, PanelOptions(detection_mode="Manual splits"))
        with self.assertRaisesRegex(ValueError, "increase max_panels"):
            split_photo_panels(image, PanelOptions(max_panels=2), "25%,50%,75%")
        with self.assertRaisesRegex(ValueError, "4096"):
            split_photo_panels(image, vertical_splits="1," * 2500)

    def test_invalid_images_and_controls_are_rejected(self):
        for image in (torch.zeros(2, 10, 10, 3), torch.full((1, 10, 10, 3), float("nan")),
                      torch.zeros(1, 10, 10, 2), torch.ones(1, 10, 10, 3) * 255):
            with self.assertRaisesRegex(ValueError, "Photo Panels"):
                split_photo_panels(image)
        for options in (PanelOptions(max_panels=1), PanelOptions(sensitivity="bad"),
                        PanelOptions(min_panel_area_percent=0), PanelOptions(detection_mode="bad")):
            with self.assertRaisesRegex(ValueError, "Photo Panels"):
                split_photo_panels(self.photo(20, 20), options)

    def test_analysis_metadata_is_tensor_free_and_matches_crops(self):
        images, report = PhotoPanelSplitV2().split(self.pair())
        meta = json.loads(report)
        self.assertEqual(meta["panel_count"], len(images))
        self.assertEqual([p["index"] for p in meta["panels"]], [0, 1])
        self.assertEqual([p["number"] for p in meta["panels"]], [1, 2])
        self.assertEqual([(p["width"], p["height"]) for p in meta["panels"]], [(100, 150), (170, 150)])

    def test_existing_chooser_single_pick_accepts_ragged_list(self):
        chooser = ROOT.parent / "image-chooser-classic" / "image_chooser_preview.py"
        if not chooser.is_file():
            self.skipTest("Image Chooser Classic is not installed in this test environment")
        # Execute its real bundling method without importing ComfyUI model management
        # or starting its HTTP/message broker. No third-party file is modified.
        tree = ast.parse(chooser.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseChooser")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "tensor_bundle")
        namespace = {"torch": torch}
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
        images, _ = PhotoPanelSplitV2().split(self.pair())
        for index in (0, 1):
            selected = namespace["tensor_bundle"](None, [image[0] for image in images], [index])
            self.assertIsInstance(selected, torch.Tensor)
            torch.testing.assert_close(selected, images[index], atol=0, rtol=0)
            self.assertEqual(selected.shape, images[index].shape)


if __name__ == "__main__":
    unittest.main()
