"""Regression tests for subject-aware placement at native edit resolutions."""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(COMFY_ROOT), str(CUSTOM_NODES)]

from PortraitUtils.smart_photo_prepare import (  # noqa: E402
    SmartPhotoPrepare,
    TARGET_DIRECT,
    TARGET_FIRERED,
    TARGET_KLEIN,
    TARGET_QWEN21,
)
from PortraitUtils.workflow_config_v2 import CropConfigNodeV2  # noqa: E402
from PortraitUtils.preparation_nodes_v2 import PhotoPrepareStandardV2  # noqa: E402
from PortraitUtils.core.editor_profiles import _Resolution, TIER_1MP  # noqa: E402


# Synthetic candidate canvases isolate crop-versus-padding decisions from model
# preset changes. Tests that do not use this fixture exercise real editor profiles.
GEOMETRY_RESOLUTIONS = (
    _Resolution(768, 1376, TIER_1MP, "synthetic narrow portrait"),
    _Resolution(832, 1248, TIER_1MP, "synthetic 2:3"),
    _Resolution(1024, 1024, TIER_1MP, "synthetic square"),
    _Resolution(1184, 888, TIER_1MP, "synthetic 4:3"),
)


class ProtectedCropTests(unittest.TestCase):
    def prepare(self, protected_top, subject_top=30, forced="Auto", smart_crop=True, subject_bottom=1000):
        image = torch.zeros((1, 1000, 500, 3))
        subject = torch.zeros((1, 1000, 500))
        subject[:, subject_top:subject_bottom, 100:400] = 1
        protected = torch.zeros_like(subject)
        protected[:, protected_top : protected_top + 170, 160:340] = 1
        return SmartPhotoPrepare().prepare(
            image,
            TARGET_FIRERED,
            smart_crop,
            "Force 1 MP",
            "bilinear",
            "false",
            0.005,
            0.995,
            0.005,
            0.995,
            8,
            0.12,
            0.06,
            0.08,
            0.75,
            "center",
            0.5,
            forced,
            subject,
            protected,
            crop_tolerance_px=0,
        )

    def test_auto_native_aspect_keeps_top_head_and_uses_firered_preset(self):
        result = self.prepare(protected_top=15)
        self.assertEqual(result[3:9], (672, 1568, 36, 0, 429, 1000))
        self.assertIn("protected", result[12])

    def test_auto_native_aspect_protects_inverted_head(self):
        result = self.prepare(protected_top=815)
        self.assertEqual(result[3:9], (672, 1568, 36, 0, 429, 1000))

    def test_auto_can_trim_unused_head_side_up_to_safety_margin(self):
        result = self.prepare(protected_top=180, subject_top=220, subject_bottom=900)
        crop_y, crop_height = result[6], result[8]
        padded_head_top = 180 - 5
        self.assertGreater(crop_y, 0)
        self.assertGreaterEqual(
            padded_head_top - crop_y, round(0.12 * crop_height)
        )

    def test_forced_9_16_keeps_protected_head(self):
        result = self.prepare(protected_top=15, forced="9:16")
        self.assertEqual(result[6], 0)
        self.assertEqual(result[7:9], (500, 1000))
        self.assertIn("edge padding", result[12])
        self.assertEqual(result[3:5], (720, 1280))

    def test_direct_smart_crop_keeps_separate_protected_head(self):
        image = torch.zeros((1, 1000, 500, 3))
        subject = torch.zeros((1, 1000, 500))
        subject[:, 300:650, 100:400] = 1
        protected = torch.zeros_like(subject)
        protected[:, 100:270, 160:340] = 1
        result = SmartPhotoPrepare().prepare(
            image,
            TARGET_DIRECT,
            True,
            "Auto (preserve detail)",
            "bilinear",
            "false",
            0.005,
            0.995,
            0.005,
            0.995,
            8,
            0.12,
            0.06,
            0.08,
            0.75,
            "center",
            0.5,
            "Auto",
            subject,
            protected,
            crop_tolerance_px=0,
        )
        self.assertLessEqual(result[6], 95)
        self.assertGreaterEqual(result[6] + result[8], 650)
        self.assertIn("protected", result[12])

    def test_protection_is_independent_of_smart_crop(self):
        for forced in ("Auto", "9:16"):
            for protected_top in (15, 815):
                with self.subTest(forced=forced, protected_top=protected_top):
                    result = self.prepare(
                        protected_top=protected_top,
                        forced=forced,
                        smart_crop=False,
                    )
                    self.assertIn("protected", result[12])
                    if forced == "Auto":
                        # FireRed can fit the full vertical extent by trimming
                        # only empty side space in a narrower native preset.
                        self.assertEqual(result[3:9], (720, 1456, 2, 0, 495, 1000))
                        self.assertNotIn("edge padding", result[12])
                    else:
                        self.assertEqual(result[3:9], (720, 1280, 0, 0, 500, 1000))
                        self.assertIn("edge padding", result[12])

    def test_connected_protection_is_lazy_in_both_crop_modes(self):
        for smart_crop in (False, True):
            with self.subTest(smart_crop=smart_crop):
                self.assertEqual(
                    SmartPhotoPrepare().check_lazy_status(
                        forced_aspect_ratio="Auto",
                        smart_crop=smart_crop,
                        protected_region_mask=None,
                    ),
                    ["protected_region_mask"],
                )
                config = CropConfigNodeV2().build(
                    "── AUTOCROP · SOLID BORDER REMOVAL ──",
                    False,
                    True,
                    0.1,
                    0.89,
                    0,
                    "── SMART CROP · SUBJECT / NATIVE SIZE ──",
                    smart_crop,
                    "Force 1 MP",
                    "Auto",
                    "a person",
                )[0]
                self.assertEqual(
                    PhotoPrepareStandardV2().check_lazy_status(
                        config, protected_region_mask=None
                    ),
                    ["protected_region_mask"],
                )
                self.assertEqual(PhotoPrepareStandardV2().check_lazy_status(config), [])
        self.assertEqual(
            SmartPhotoPrepare().check_lazy_status(forced_aspect_ratio="Auto"),
            [],
        )


class SubjectFramingTests(unittest.TestCase):
    def prepare(self, source_mask, protected=None, image=None, **overrides):
        mask = source_mask
        height, width = mask.shape[-2:]
        if image is None:
            image = torch.zeros((1, height, width, 3))
        options = dict(
            image=image, editor_target=TARGET_FIRERED, smart_crop=True,
            resolution_policy="Force 1 MP", resize_method="bilinear",
            invert_mask="false", q_left=0.005, q_right=0.995,
            q_top=0.005, q_bottom=0.995, min_span_px=8,
            headroom_ratio=0.12, footroom_ratio=0.06, side_margin_ratio=0.08,
            bottom_priority=0.75, horiz_gravity="center",
            framing_tolerance_percent=0.5, forced_aspect_ratio="Auto",
            mask=mask, protected_region_mask=protected, crop_tolerance_px=0,
        )
        options.update(overrides)
        return SmartPhotoPrepare().prepare(**options)

    def assert_contains(self, result, mask):
        x, y, width, height = result[5:9]
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, mask.shape[-1])
        self.assertLessEqual(y + height, mask.shape[-2])
        self.assertEqual(float(mask[:, y:y + height, x:x + width].sum()), float(mask.sum()))
        self.assertEqual(tuple(result[0].shape[1:3]), (result[4], result[3]))
        self.assertEqual(tuple(result[2].shape[-2:]), (result[4], result[3]))
        self.assertEqual(tuple(result[1].shape[1:3]), (height, width))

    def assert_margins(self, result, bbox, headroom=0.12, footroom=0.06, side=0.08):
        self.assertNotIn("edge padding", result[12])
        x, y, width, height = result[5:9]
        bx, by, bw, bh = bbox
        actual = (bx - x, x + width - bx - bw, by - y, y + height - by - bh)
        required = (math.ceil(side * width), math.ceil(side * width),
                    math.ceil(headroom * height), math.ceil(footroom * height))
        for edge, available, requested in zip(("left", "right", "top", "bottom"), actual, required):
            self.assertGreaterEqual(available, requested, f"{edge}: {actual} vs {required}; {result[12]}")

    def test_tight_framing_keeps_margins_with_every_horizontal_gravity(self):
        mask = torch.zeros((1, 1000, 1000))
        mask[:, 400:600, 400:600] = 1
        protected = torch.zeros_like(mask)
        protected[:, 450:500, 450:550] = 1
        for target in (TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21, TARGET_DIRECT):
            for aspect in ("Auto", "1:1"):
                for gravity in ("left", "center", "right"):
                    for protection in (None, protected):
                        with self.subTest(target=target, aspect=aspect, gravity=gravity, protected=protection is not None):
                            result = self.prepare(mask, protected=protection, editor_target=target,
                                                  forced_aspect_ratio=aspect, horiz_gravity=gravity)
                            self.assert_margins(result, (400, 400, 200, 200))

    def test_bottom_priority_cannot_discard_top_or_bottom_margin(self):
        mask = torch.zeros((1, 1000, 1000))
        mask[:, 400:600, 400:600] = 1
        for priority in (0.0, 0.75, 1.0):
            for aspect in ("Auto", "1:1"):
                with self.subTest(priority=priority, aspect=aspect):
                    result = self.prepare(mask, bottom_priority=priority, forced_aspect_ratio=aspect)
                    self.assert_margins(result, (400, 400, 200, 200))

    def test_loose_protected_placement_keeps_subject_footroom(self):
        mask = torch.zeros((1, 1000, 500))
        mask[:, 350:896, 100:400] = 1
        protected = torch.zeros_like(mask)
        protected[:, 350:450, 160:340] = 1
        for aspect in ("Auto", "9:16"):
            with self.subTest(aspect=aspect):
                result = self.prepare(mask, protected=protected, smart_crop=False,
                                      forced_aspect_ratio=aspect)
                self.assert_margins(result, (100, 350, 300, 546))

    @patch("PortraitUtils.core.preparation._build_resolutions", new=lambda *args, **kwargs: GEOMETRY_RESOLUTIONS)
    def test_safe_crop_wins_even_when_padding_changes_fewer_pixels(self):
        mask = torch.zeros((1, 1000, 640))
        mask[:, :, 200:400] = 1
        for smart in (False, True):
            with self.subTest(smart=smart):
                result = self.prepare(mask, smart_crop=smart, crop_tolerance_px=4)
                self.assert_contains(result, mask)
                # Safe narrow-portrait cropping takes priority over 2:3 padding (27,000 pixels).
                self.assertEqual(result[3:5], (768, 1376))
                self.assertEqual(result[7:9], (558, 1000))
                self.assertIn("removed_px=82000; added_px=0; changed_px=82000", result[12])
                self.assertNotIn("edge padding", result[12])

    def test_small_mask_intrusion_can_crop_instead_of_padding(self):
        mask = torch.ones((1, 104, 100))
        result = self.prepare(mask, editor_target=TARGET_DIRECT, smart_crop=False,
                              forced_aspect_ratio="1:1", crop_tolerance_px=4)
        self.assertEqual(result[5:9], (0, 2, 100, 100))
        self.assertIn("removed_px=400; added_px=0; changed_px=400", result[12])
        self.assertIn("margin_intrusion_px L/R/T/B=0/0/2/2", result[12])

    def test_small_margin_intrusion_keeps_the_subject_itself(self):
        mask = torch.zeros((1, 104, 100))
        mask[:, 7:97, 35:65] = 1
        result = self.prepare(mask, editor_target=TARGET_DIRECT, smart_crop=False,
                              forced_aspect_ratio="1:1", crop_tolerance_px=4)
        self.assert_contains(result, mask)
        self.assertIn("limited crop", result[12])
        self.assertNotIn("edge padding", result[12])

    def test_tolerance_limit_and_zero_setting_are_respected(self):
        mask = torch.ones((1, 108, 100))
        for tolerance in (0, 3, 4):
            with self.subTest(tolerance=tolerance):
                result = self.prepare(mask, editor_target=TARGET_DIRECT, smart_crop=False,
                                      forced_aspect_ratio="1:1", crop_tolerance_px=tolerance)
                if tolerance < 4:
                    self.assert_contains(result, mask)
                    self.assertIn("edge padding", result[12])
                else:
                    self.assertEqual(result[5:9], (0, 4, 100, 100))
                    self.assertIn("margin_intrusion_px L/R/T/B=0/0/4/4", result[12])

    def test_substantial_intrusion_pads_even_when_cropping_is_cheaper(self):
        mask = torch.ones((1, 120, 100))
        result = self.prepare(mask, editor_target=TARGET_DIRECT, smart_crop=False,
                              forced_aspect_ratio="1:1", crop_tolerance_px=4)
        self.assert_contains(result, mask)
        self.assertIn("removed_px=0; added_px=2400; changed_px=2400", result[12])
        self.assertIn("edge padding", result[12])

    def test_protected_region_cannot_be_trimmed_even_with_tolerance(self):
        mask = torch.ones((1, 104, 100))
        result = self.prepare(mask, protected=mask, editor_target=TARGET_DIRECT,
                              smart_crop=False, forced_aspect_ratio="1:1", crop_tolerance_px=4)
        self.assert_contains(result, mask)
        self.assertIn("edge padding", result[12])

    @patch("PortraitUtils.core.preparation._build_resolutions", new=lambda *args, **kwargs: GEOMETRY_RESOLUTIONS)
    def test_fallback_padding_wins_over_an_allowed_more_expensive_crop(self):
        mask = torch.ones((1, 100, 64))
        protected = torch.zeros_like(mask)
        protected[:, :, 30:34] = 1
        result = self.prepare(mask, protected=protected, smart_crop=False, crop_tolerance_px=4)
        # A limited narrow-portrait crop removes 800 pixels; 2:3 padding adds only 300.
        self.assert_contains(result, mask)
        self.assertEqual(result[3:5], (832, 1248))
        self.assertIn("removed_px=0; added_px=300; changed_px=300", result[12])
        self.assertIn("edge padding", result[12])

    @patch("PortraitUtils.core.preparation._build_resolutions", new=lambda *args, **kwargs: GEOMETRY_RESOLUTIONS)
    def test_cropping_wins_when_it_changes_fewer_pixels_than_padding(self):
        mask = torch.zeros((1, 1000, 600))
        mask[:, :, 200:400] = 1
        for smart in (False, True):
            with self.subTest(smart=smart):
                result = self.prepare(mask, smart_crop=smart)
                self.assert_contains(result, mask)
                # A narrow-portrait crop removes 42,000 pixels; padding it adds 45,000.
                self.assertEqual(result[3:5], (768, 1376))
                self.assertEqual(result[7:9], (558, 1000))
                self.assertIn("removed_px=42000; added_px=0; changed_px=42000", result[12])
                self.assertNotIn("edge padding", result[12])

    @patch("PortraitUtils.core.preparation._build_resolutions", new=lambda *args, **kwargs: GEOMETRY_RESOLUTIONS)
    def test_equal_pixel_counts_prefer_cropping(self):
        mask = torch.zeros((1, 600, 700))
        mask[:, :, 250:450] = 1
        result = self.prepare(mask, smart_crop=False)
        self.assert_contains(result, mask)
        # Square cropping and 4:3 padding both change 60,000 pixels.
        self.assertEqual(result[3:5], (1024, 1024))
        self.assertEqual(result[7:9], (600, 600))
        self.assertIn("removed_px=60000; added_px=0; changed_px=60000", result[12])

    def test_mixed_crop_and_padding_costs_are_added_without_cancelling(self):
        mask = torch.zeros((1, 1000, 600))
        mask[:, 200:800, 150:450] = 1
        result = self.prepare(mask, forced_aspect_ratio="1:1")
        self.assert_contains(result, mask)
        self.assertEqual(result[7:9], (600, 732))
        self.assertIn("removed_px=160800; added_px=96624; changed_px=257424", result[12])

    @patch("PortraitUtils.core.preparation._build_resolutions", new=lambda *args, **kwargs: GEOMETRY_RESOLUTIONS)
    def test_framing_tolerance_cannot_override_fallback_pixel_comparison(self):
        mask = torch.ones((1, 100, 64))
        protected = torch.zeros_like(mask)
        protected[:, :, 30:34] = 1
        result = self.prepare(mask, protected=protected, smart_crop=False,
                              crop_tolerance_px=4, framing_tolerance_percent=10.0)
        self.assertEqual(result[3:5], (832, 1248))
        self.assertIn("changed_px=300", result[12])

    @patch("PortraitUtils.core.preparation._build_resolutions", new=lambda *args, **kwargs: GEOMETRY_RESOLUTIONS)
    def test_loose_auto_uses_subject_mask_for_placement(self):
        mask = torch.zeros((1, 1000, 500))
        mask[:, 400:1000, 100:400] = 1
        result = self.prepare(mask, smart_crop=False)
        self.assert_contains(result, mask)
        self.assertEqual(result[6], 104)
        self.assertNotIn("edge padding", result[12])

    def test_smart_crop_changes_tightness_only(self):
        mask = torch.zeros((1, 1000, 1000))
        mask[:, 400:600, 400:600] = 1
        for aspect in ("Auto", "1:1"):
            loose = self.prepare(mask, smart_crop=False, forced_aspect_ratio=aspect)
            tight = self.prepare(mask, smart_crop=True, forced_aspect_ratio=aspect)
            self.assert_contains(loose, mask)
            self.assert_contains(tight, mask)
            self.assertGreater(loose[7] * loose[8], tight[7] * tight[8])

    def test_disconnected_subjects_and_tiny_remote_component_are_retained(self):
        mask = torch.zeros((1, 1000, 1000))
        mask[:, 300:700, 200:400] = 1
        mask[:, 400:800, 650:800] = 1
        mask[:, 50:52, 920:922] = 1
        for smart in (False, True):
            result = self.prepare(mask, smart_crop=smart)
            self.assert_contains(result, mask)

    def test_padding_replaces_subject_clipping_with_or_without_protection(self):
        mask = torch.zeros((1, 1000, 500))
        mask[:, :, 100:400] = 1
        protected = torch.zeros_like(mask)
        protected[:, :170, 160:340] = 1
        for protection in (None, protected):
            for smart in (False, True):
                result = self.prepare(mask, protected=protection, smart_crop=smart,
                                      forced_aspect_ratio="9:16")
                self.assert_contains(result, mask)
                self.assertEqual(result[3:5], (720, 1280))
                self.assertIn("edge padding", result[12])
                self.assertEqual(result[10], 0.0)
                self.assertTrue(torch.all(result[2][:, :, 0] == 0))
                self.assertTrue(torch.all(result[2][:, :, -1] == 0))

    def test_protected_only_mask_can_trigger_padding(self):
        mask = torch.ones((1, 1000, 500))
        result = self.prepare(mask, mask=None, protected=mask, smart_crop=False)
        self.assert_contains(result, mask)
        self.assertIn("edge padding", result[12])
        self.assertEqual(float(result[2].sum()), 0.0)

    def test_forced_aspect_pads_all_subjects_and_protected_regions(self):
        mask = torch.ones((1, 200, 500))
        for target in (TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21):
            for smart in (False, True):
                with self.subTest(target=target, smart=smart):
                    result = self.prepare(mask, protected=mask, editor_target=target,
                                          smart_crop=smart, forced_aspect_ratio="9:16")
                    self.assert_contains(result, mask)
                    self.assertIn("edge padding", result[12])
                    self.assertEqual(result[3] * 16, result[4] * 9)
                    alignment = 32 if target == TARGET_QWEN21 else 16
                    self.assertEqual(result[3] % alignment, 0)
                    self.assertEqual(result[4] % alignment, 0)

    def test_model_and_resolution_policies_preserve_subjects(self):
        mask = torch.ones((1, 640, 200))
        for target in (TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21):
            for smart in (False, True):
                for policy in ("Force 1 MP", "Force 1.5 MP", "Auto (preserve detail)", "Auto (closest scale)"):
                    with self.subTest(target=target, smart=smart, policy=policy):
                        result = self.prepare(mask, editor_target=target,
                                              smart_crop=smart, resolution_policy=policy)
                        self.assert_contains(result, mask)
                        alignment = 32 if target == TARGET_QWEN21 else 16
                        self.assertEqual(result[3] % alignment, 0)
                        self.assertEqual(result[4] % alignment, 0)
                        if policy.startswith("Force"):
                            self.assertIn(policy[6:], result[11])

    def test_direct_padding_extends_pixels_and_keeps_mask_padding_black(self):
        image = torch.arange(10 * 20 * 3, dtype=torch.float32).reshape(1, 10, 20, 3) / 600
        mask = torch.ones((1, 10, 20))
        result = self.prepare(mask, image=image, editor_target=TARGET_DIRECT,
                              smart_crop=False, forced_aspect_ratio="1:1")
        self.assert_contains(result, mask)
        self.assertEqual(result[3:5], (20, 20))
        self.assertEqual(result[5:9], (0, 0, 20, 10))
        torch.testing.assert_close(result[1], image)
        torch.testing.assert_close(result[0][:, 5:15], image)
        torch.testing.assert_close(result[0][:, :5], image[:, :1].expand(-1, 5, -1, -1))
        torch.testing.assert_close(result[0][:, 15:], image[:, -1:].expand(-1, 5, -1, -1))
        self.assertEqual(float(result[2][:, :5].sum()), 0.0)
        self.assertEqual(float(result[2][:, 15:].sum()), 0.0)
        torch.testing.assert_close(result[2][:, 5:15], mask)
        self.assertEqual(result[9], 1.0)
        self.assertEqual(result[10], 0.0)

    def test_direct_loose_auto_keeps_source_even_with_subject_mask(self):
        mask = torch.zeros((1, 200, 400))
        mask[:, 80:120, 180:220] = 1
        result = self.prepare(mask, editor_target=TARGET_DIRECT, smart_crop=False)
        self.assertEqual(result[3:9], (400, 200, 0, 0, 400, 200))
        self.assertEqual(result[9], 1.0)

    def test_uniform_masks_have_no_automatic_inversion(self):
        for value in (0.0, 1.0):
            mask = torch.full((1, 1000, 500), value)
            result = self.prepare(mask, invert_mask="auto")
            if value:
                self.assert_contains(result, mask)
                self.assertIn("edge padding", result[12])
            else:
                self.assertEqual(float(result[2].sum()), 0.0)
                self.assertIn("subject mask was empty", result[12])

    def test_tiny_direct_source_is_padded_to_forced_aspect(self):
        mask = torch.ones((1, 2, 3))
        result = self.prepare(mask, editor_target=TARGET_DIRECT, forced_aspect_ratio="9:16")
        self.assert_contains(result, mask)
        self.assertEqual(result[3:5], (9, 16))

    def test_tight_frame_expands_when_protected_margin_needs_more_room(self):
        mask = torch.zeros((1, 200, 200))
        mask[:, 5:139, 92:99] = 1
        protected = torch.zeros_like(mask)
        protected[:, 154:155, 22:100] = 1
        for target in (TARGET_DIRECT, TARGET_FIRERED):
            for aspect in ("Auto", "1:1"):
                with self.subTest(target=target, aspect=aspect):
                    result = self.prepare(mask, protected=protected, editor_target=target,
                                          forced_aspect_ratio=aspect)
                    self.assert_contains(result, mask)
                    self.assert_contains(result, protected)
                    self.assertNotIn("edge padding", result[12])

    def test_v2_forwards_the_new_tolerance_setting(self):
        config = CropConfigNodeV2().build(
            "── AUTOCROP · SOLID BORDER REMOVAL ──", False, True, 0.1, 0.89, 0,
            "── SMART CROP · SUBJECT / NATIVE SIZE ──", False, "Force 1 MP", "1:1", "objects",
        )[0]
        for tolerance in (0, 4):
            result = PhotoPrepareStandardV2().prepare(
                torch.zeros((1, 104, 100, 3)), TARGET_DIRECT, config,
                "bilinear", "false", 0.005, 0.995, 0.005, 0.995, 8,
                0.12, 0.06, 0.08, 0.75, "center", 0.5,
                torch.ones((1, 104, 100)), crop_tolerance_px=tolerance,
            )
            self.assertEqual(
                (result[3].plan.target_width, result[3].plan.target_height),
                (104, 104) if tolerance == 0 else (100, 100),
            )

    @patch("PortraitUtils.core.preparation._build_resolutions", new=lambda *args, **kwargs: GEOMETRY_RESOLUTIONS)
    def test_v2_wrapper_preserves_all_subjects_when_smart_crop_is_off(self):
        mask = torch.zeros((1, 1000, 500))
        mask[:, 400:1000, 100:400] = 1
        config = CropConfigNodeV2().build(
            "── AUTOCROP · SOLID BORDER REMOVAL ──", False, True, 0.1, 0.89, 0,
            "── SMART CROP · SUBJECT / NATIVE SIZE ──", False, "Force 1 MP", "Auto", "objects",
        )[0]
        result = PhotoPrepareStandardV2().prepare(
            torch.zeros((1, 1000, 500, 3)), TARGET_FIRERED, config,
            "bilinear", "false", 0.005, 0.995, 0.005, 0.995, 8,
            0.12, 0.06, 0.08, 0.75, "center", 0.5, mask,
        )
        crop = result[3].plan.crop
        self.assertEqual(float(mask[:, crop.y:crop.y+crop.height,
                                    crop.x:crop.x+crop.width].sum()), float(mask.sum()))
        self.assertEqual(crop.y, 104)


if __name__ == "__main__":
    unittest.main()
