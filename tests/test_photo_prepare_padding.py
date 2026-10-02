"""Selectable canvas fills without changing framing, protection, or native crops."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT.parent)]

from PortraitUtils.core.contracts import CropConfigV2, PADDING_FILLS, validate_config
from PortraitUtils.core.editor_profiles import TARGET_DIRECT, TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21
from PortraitUtils.core.preparation import PreparationOptions, apply_plan, prepare_photo
from PortraitUtils.core.tensors import pad_crop
from PortraitUtils.preparation_nodes_v2 import PhotoPrepareStandardV2
from PortraitUtils.workflow_config_v2 import CropConfigNodeV2


class PhotoPreparePaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.config = CropConfigV2(False, False, .07, .85, 0, False, "Force 1 MP", "1:1", "person")
        self.options = PreparationOptions(resize_method="bilinear", invert_mask="false",
                                          resolution_policy="Force 1 MP", forced_aspect_ratio="1:1",
                                          crop_tolerance_px=0)
        self.image = torch.linspace(.2, .8, 60 * 30 * 3).reshape(1, 60, 30, 3)
        self.mask = torch.ones(1, 60, 30)

    def test_optional_control_is_appended_without_moving_existing_widgets(self):
        schema = CropConfigNodeV2.INPUT_TYPES()
        self.assertEqual(list(schema["required"]), [
            "autocrop_section", "autocrop_strip_bottom_banner", "autocrop_detect_borders",
            "autocrop_fuzz_tolerance", "autocrop_edge_uniformity", "autocrop_pad_px",
            "smart_crop_section", "smart_crop", "resolution_policy", "forced_aspect_ratio", "subject_mask",
        ])
        self.assertEqual(list(schema["optional"]), ["padding_fill"])
        self.assertEqual(schema["optional"]["padding_fill"][0], list(PADDING_FILLS))
        self.assertEqual(schema["optional"]["padding_fill"][1]["default"], "Edge extension")

    def test_legacy_config_construction_and_producer_calls_keep_default(self):
        self.assertEqual(self.config.padding_fill, "Edge extension")
        args = asdict(self.config)
        del args["padding_fill"]
        config = CropConfigNodeV2().build(autocrop_section="", smart_crop_section="", **args)[0]
        self.assertEqual(config.padding_fill, "Edge extension")
        self.assertEqual(config.subject_mask, "person")
        for fill in PADDING_FILLS:
            config = CropConfigNodeV2().build(autocrop_section="", smart_crop_section="", padding_fill=fill, **args)[0]
            self.assertEqual(config.padding_fill, fill)

    def test_default_edge_extension_is_identical_to_previous_padding(self):
        image = torch.linspace(.2, .8, 4 * 5 * 3, dtype=torch.float64).reshape(1, 4, 5, 3)
        mask = torch.rand(1, 4, 5, dtype=torch.float64)
        padding = (2, 3, 1, 4)
        output, coverage = pad_crop(image, mask, padding)
        expected = torch.nn.functional.pad(image.movedim(-1, 1), padding, mode="replicate").movedim(1, -1)
        expected_mask = torch.nn.functional.pad(mask, padding, mode="constant", value=0)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(coverage, expected_mask, atol=0, rtol=0)

    def test_constant_fills_preserve_source_dtype_device_and_mask(self):
        padding = (2, 3, 1, 4)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for channels in (1, 3, 4):
                for fill, level in (("Black", 0), ("White", 1)):
                    with self.subTest(dtype=dtype, channels=channels, fill=fill):
                        image = torch.linspace(.2, .8, 2 * 4 * 5 * channels).reshape(2, 4, 5, channels).to(dtype)
                        mask = torch.rand(2, 4, 5, dtype=torch.float64)
                        original = image.clone()
                        output, coverage = pad_crop(image, mask, padding, fill)
                        expected = image.new_full((2, 9, 10, channels), level)
                        if channels == 4:
                            expected[..., 3] = 1
                        expected[:, 1:5, 2:7] = image
                        torch.testing.assert_close(output, expected, atol=0, rtol=0)
                        torch.testing.assert_close(image, original, atol=0, rtol=0)
                        torch.testing.assert_close(coverage[:, 1:5, 2:7], mask, atol=0, rtol=0)
                        self.assertFalse(coverage[:, :1].any())
                        self.assertFalse(coverage[:, 5:].any())
                        self.assertFalse(coverage[:, :, :2].any())
                        self.assertFalse(coverage[:, :, 7:].any())
                        self.assertEqual(output.dtype, dtype)
                        self.assertEqual(output.device, image.device)
                        self.assertEqual(coverage.dtype, mask.dtype)

    def test_missing_mask_and_one_pixel_source_are_supported(self):
        image = torch.tensor([[[[.2, .4, .6, .3]]]], dtype=torch.float64)
        for fill in PADDING_FILLS:
            output, mask = pad_crop(image, None, (3, 2, 4, 1), fill)
            self.assertIsNone(mask)
            self.assertEqual(output.shape, (1, 6, 6, 4))
            torch.testing.assert_close(output[:, 4:5, 3:4], image, atol=0, rtol=0)
            if fill == "Edge extension":
                torch.testing.assert_close(output, image.expand(1, 6, 6, 4), atol=0, rtol=0)

    def test_no_padding_is_a_pixel_and_storage_preserving_bypass(self):
        for fill in PADDING_FILLS:
            output, mask = pad_crop(self.image, self.mask, (0, 0, 0, 0), fill)
            self.assertIs(output, self.image)
            self.assertIs(mask, self.mask)
            result = prepare_photo(self.image, TARGET_DIRECT, replace(self.options, forced_aspect_ratio="Auto", padding_fill=fill))
            self.assertEqual(result.image.data_ptr(), self.image.data_ptr())
            torch.testing.assert_close(result.image, self.image, atol=0, rtol=0)

    def test_invalid_fill_is_actionable_even_when_no_padding_is_needed(self):
        for invalid in ("Automatic", "black", 0, None):
            with self.subTest(fill=invalid):
                with self.assertRaisesRegex(ValueError, "padding_fill"):
                    validate_config(replace(self.config, padding_fill=invalid))
                with self.assertRaisesRegex(ValueError, "padding_fill"):
                    replace(self.options, padding_fill=invalid).validate()
                with self.assertRaisesRegex(ValueError, "padding_fill"):
                    pad_crop(self.image, None, (0, 0, 0, 0), invalid)

    def test_fill_cannot_change_geometry_or_override_protection_in_any_target(self):
        for smart in (False, True):
            for target in (TARGET_DIRECT, TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21):
                expected_crop = None
                for fill in PADDING_FILLS:
                    with self.subTest(smart_crop=smart, target=target, fill=fill):
                        result = prepare_photo(self.image, target, replace(self.options, smart_crop=smart, padding_fill=fill),
                                               self.mask, self.mask)
                        crop = result.info.plan.crop
                        self.assertTrue(any(crop.padding))
                        self.assertEqual((crop.x, crop.y, crop.width, crop.height), (0, 0, 30, 60))
                        if expected_crop is None:
                            expected_crop = crop
                        self.assertEqual(crop, expected_crop)
                        torch.testing.assert_close(result.native_crop, self.image, atol=0, rtol=0)
                        self.assertEqual(result.native_crop.untyped_storage().data_ptr(), self.image.untyped_storage().data_ptr())
                        if fill != "Edge extension":
                            self.assertTrue(torch.all(result.image[:, :, 0] == (0 if fill == "Black" else 1)))
                        self.assertFalse(result.mask[:, :, 0].any())

    def test_standard_node_consumes_configured_fill(self):
        local = {key: getattr(self.options, key) for key in (
            "resize_method", "invert_mask", "q_left", "q_right", "q_top", "q_bottom", "min_span_px",
            "headroom_ratio", "footroom_ratio", "side_margin_ratio", "bottom_priority", "horiz_gravity",
            "framing_tolerance_percent", "crop_tolerance_px",
        )}
        for fill in PADDING_FILLS:
            config = replace(self.config, padding_fill=fill)
            kwargs = dict(image=self.image, editor_target=TARGET_DIRECT, crop_config=config,
                          mask=self.mask, protected_region_mask=self.mask, **local)
            standard = PhotoPrepareStandardV2().prepare(**kwargs)
            expected = prepare_photo(
                self.image, TARGET_DIRECT, replace(self.options, padding_fill=fill),
                self.mask, self.mask,
            )
            self.assertEqual(len(standard), 4)
            torch.testing.assert_close(expected.image, standard[0], atol=0, rtol=0)
            torch.testing.assert_close(expected.native_crop, standard[1], atol=0, rtol=0)
            torch.testing.assert_close(expected.mask, standard[2], atol=0, rtol=0)
            self.assertEqual(standard[3].plan.options.padding_fill, fill)

    def test_plan_replay_and_diagnostics_record_selected_fill(self):
        for fill in PADDING_FILLS:
            result = prepare_photo(self.image, TARGET_DIRECT, replace(self.options, padding_fill=fill),
                                   self.mask, self.mask)
            image, native, mask = apply_plan(self.image, self.mask, result.info.plan)
            torch.testing.assert_close(image, result.image, atol=0, rtol=0)
            torch.testing.assert_close(native, result.native_crop, atol=0, rtol=0)
            torch.testing.assert_close(mask, result.mask, atol=0, rtol=0)
            info = result.info.to_dict()
            self.assertEqual(info["transform"]["padding_fill"], fill)
            self.assertEqual(info["plan"]["options"]["padding_fill"], fill)
            self.assertTrue(any("fill: " + fill in message for message in info["warnings"]))
            self.assertIn("padding_fill=" + fill, info["selection_reason"])


if __name__ == "__main__":
    unittest.main()
