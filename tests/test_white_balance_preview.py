"""CPU-only regression tests for the separately registered WB replacement."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes")]

import PortraitUtils
from PortraitUtils.auto_color_match import AutoWBColorMatch
from PortraitUtils import white_balance_preview as wb
from PortraitUtils.core import white_balance as wb_engine


class WhiteBalancePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small CPU fixtures otherwise pay excessive thread-pool overhead.
        # This is a test setting only; the production node never changes it.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.node = wb.PortraitWhiteBalancePreview()

    def run_node(self, image, reference=None, **kwargs):
        return self.node.run(image, reference, **kwargs)[0]

    def fixture(self):
        grid = torch.linspace(0, 1, 20 * 24, dtype=torch.float64).reshape(1, 20, 24, 1)
        lab = torch.cat((45 + grid * 20, 2 + grid * 6, 5 + grid.square() * 5), -1)
        return wb.lab_to_rgb(lab), lab

    def test_only_promoted_white_balance_is_registered(self):
        self.assertNotIn("AutoWBColorMatch", PortraitUtils.NODE_CLASS_MAPPINGS)
        self.assertIs(PortraitUtils.NODE_CLASS_MAPPINGS["PortraitWhiteBalancePreview"], type(self.node))
        self.assertNotEqual(type(self.node), AutoWBColorMatch)

    def test_existing_socket_and_widget_order_is_compatible(self):
        old = AutoWBColorMatch.INPUT_TYPES()["required"]
        new = self.node.INPUT_TYPES()["required"]
        self.assertEqual(list(old), list(new))
        for key in old:
            self.assertEqual(old[key][0], new[key][0])
            if len(old[key]) > 1:
                for option, value in old[key][1].items():
                    self.assertEqual(new[key][1][option], value)
        self.assertEqual(self.node.RETURN_TYPES, AutoWBColorMatch.RETURN_TYPES)
        self.assertEqual(list(self.node.INPUT_TYPES()["optional"]), ["preserve_luminance"])

    def test_lab_white_black_and_neutral_gray(self):
        rgb = torch.tensor([0., 0.18, 0.5, 1.], dtype=torch.float64).reshape(1, 1, 4, 1).expand(-1, -1, -1, 3)
        lab = wb.rgb_to_lab(rgb)
        torch.testing.assert_close(lab[..., 1:], torch.zeros_like(lab[..., 1:]), atol=1e-12, rtol=0)
        self.assertAlmostEqual(lab[0, 0, -1, 0].item(), 100, places=10)
        self.assertEqual(lab[0, 0, 0, 0].item(), 0)

    def test_lab_known_srgb_primaries(self):
        rgb = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
        expected = torch.tensor([[53.2371, 80.0901, 67.2033],
                                 [87.7355, -86.1816, 83.1866],
                                 [32.3009, 79.1953, -107.8555]], dtype=torch.float64).reshape(1, 1, 3, 3)
        torch.testing.assert_close(wb.rgb_to_lab(rgb), expected, atol=0.001, rtol=0)

    def test_rgb_lab_round_trip(self):
        generator = torch.Generator().manual_seed(19)
        rgb = torch.rand(2, 13, 17, 3, generator=generator, dtype=torch.float64)
        torch.testing.assert_close(wb.lab_to_rgb(wb.rgb_to_lab(rgb)), rgb, atol=2e-13, rtol=0)

    def test_extended_transfer_curves_round_trip(self):
        rgb = torch.tensor([-1.5, -0.04, 0, 0.003, 0.04, 0.5, 1, 1.5], dtype=torch.float64)
        torch.testing.assert_close(wb.linear_to_srgb(wb.srgb_to_linear(rgb)), rgb, atol=1e-14, rtol=0)

    def test_lab_conversion_does_not_silently_clip(self):
        rgb = wb.lab_to_rgb(torch.tensor([[[[65., 120., 100.]]]], dtype=torch.float64))
        self.assertTrue(torch.isfinite(rgb).all())
        self.assertTrue(bool((rgb < 0).any() | (rgb > 1).any()))

    def test_grayworld_corrects_channels_independently_and_preserves_luminance(self):
        image = torch.tensor([0.75, 0.6, 0.5], dtype=torch.float64).expand(1, 9, 11, 3)
        result = self.run_node(image, method="wb_grayworld")
        torch.testing.assert_close(result[..., 0], result[..., 1], atol=1e-12, rtol=0)
        torch.testing.assert_close(result[..., 1], result[..., 2], atol=1e-12, rtol=0)
        weights = image.new_tensor(wb._RGB_XYZ[1])
        torch.testing.assert_close(wb.srgb_to_linear(result) @ weights,
                                   wb.srgb_to_linear(image) @ weights, atol=1e-12, rtol=0)

    def test_highlight_white_patch_corrects_channels_independently(self):
        image = torch.full((1, 8, 8, 3), 0.1, dtype=torch.float64)
        image[:, :2] = torch.tensor([0.8, 0.65, 0.55])
        result = self.run_node(image, method="wb_highlight")
        torch.testing.assert_close(result[:, :2, :, 0], result[:, :2, :, 1], atol=1e-12, rtol=0)
        torch.testing.assert_close(result[:, :2, :, 1], result[:, :2, :, 2], atol=1e-12, rtol=0)

    def test_neutral_highlights_are_not_forced_brighter_by_default(self):
        image = torch.full((1, 8, 8, 3), 0.65, dtype=torch.float64)
        torch.testing.assert_close(self.run_node(image, method="wb_highlight"), image, atol=1e-12, rtol=0)

    def test_optional_highlight_brightness_target(self):
        image = torch.full((1, 8, 8, 3), 0.8, dtype=torch.float64)
        result = self.run_node(image, method="wb_highlight", preserve_luminance=False)
        torch.testing.assert_close(result, torch.full_like(image, 0.95), atol=1e-12, rtol=0)

    def test_highlight_excludes_clipped_white(self):
        image = torch.ones(1, 8, 8, 3, dtype=torch.float64)
        image[:, :2] = torch.tensor([0.8, 0.65, 0.55])
        gains = wb._wb_gains(image, True, 95, True)
        self.assertNotAlmostEqual(gains[0, 0].item(), gains[0, 2].item())
        selected = wb.srgb_to_linear(image[:, :2]) * gains[0]
        torch.testing.assert_close(selected[..., 0], selected[..., 2], atol=1e-12, rtol=0)

    def test_black_white_and_absent_channels_are_safe(self):
        for color in ([0, 0, 0], [1, 1, 1], [0.8, 0, 0.3]):
            image = torch.tensor(color, dtype=torch.float64).expand(1, 8, 8, 3)
            for method in ("wb_grayworld", "wb_highlight"):
                with self.subTest(color=color, method=method):
                    torch.testing.assert_close(self.run_node(image, method=method), image, atol=1e-12, rtol=0)

    def test_wb_only_ignores_reference(self):
        image, _ = self.fixture()
        for method in ("wb_grayworld", "wb_highlight"):
            a = self.run_node(image, method=method)
            b = self.run_node(image, torch.full_like(image, 0.9), method=method)
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_strength_zero_is_exact_identity(self):
        image, _ = self.fixture()
        self.assertIs(self.run_node(image, None, strength=0), image)

    def test_strength_blends_once_in_srgb(self):
        image, _ = self.fixture()
        full = self.run_node(image, method="wb_grayworld", clip_gamut=False)
        partial = self.run_node(image, method="wb_grayworld", clip_gamut=False, strength=0.3)
        torch.testing.assert_close(partial, image + 0.3 * (full - image), atol=1e-12, rtol=0)

    def test_force_size_is_analysis_only_and_retains_pixel_detail(self):
        y, x = torch.meshgrid(torch.arange(80), torch.arange(120), indexing="ij")
        pattern = (0.3 + ((x + y) % 2) * 0.4).double()[None, ..., None]
        image = pattern * torch.tensor([0.9, 0.7, 0.5], dtype=torch.float64)
        with patch.object(wb.F, "interpolate", wraps=wb.F.interpolate) as resize:
            result = self.run_node(image, method="wb_grayworld", force_size=True, target_width=32, target_height=16)
        self.assertEqual(resize.call_count, 1)
        self.assertEqual(result.shape, image.shape)
        sample = wb._analysis_image(image, True, 32, 16)
        self.assertEqual(sample.shape[1:3], (16, 24))
        linear = wb.srgb_to_linear(sample)
        mean = linear.mean((1, 2))
        target = mean @ image.new_tensor(wb._RGB_XYZ[1])
        gains = (target[:, None] / mean).clamp(0.25, 4)
        expected = wb.linear_to_srgb(wb.srgb_to_linear(image) * gains[:, None, None]).clamp(0, 1)
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)
        self.assertGreater((result[:, :, 1:] - result[:, :, :-1]).abs().mean().item(), 0.2)

    def test_analysis_never_enlarges_small_images(self):
        image, _ = self.fixture()
        self.assertIs(wb._analysis_image(image, True, 1440, 1080), image)

    def test_reinhard_matches_reliable_lab_moments(self):
        image, lab = self.fixture()
        target = lab * lab.new_tensor([0.9, 1.2, 0.8]) + lab.new_tensor([5, 2, -1])
        reference = wb.lab_to_rgb(target)
        result = self.run_node(image, reference, method="reinhard_lab", clip_gamut=False)
        torch.testing.assert_close(wb.rgb_to_lab(result), target, atol=1e-10, rtol=0)

    def test_l_only_preserves_chroma(self):
        image, lab = self.fixture()
        target = lab * lab.new_tensor([0.9, 1.2, 0.8]) + lab.new_tensor([5, 2, -1])
        result = self.run_node(image, wb.lab_to_rgb(target), method="lab_l_only", clip_gamut=False)
        expected = torch.cat((target[..., :1], lab[..., 1:]), -1)
        torch.testing.assert_close(wb.rgb_to_lab(result), expected, atol=1e-10, rtol=0)

    def test_matching_self_reference_is_exact_identity(self):
        image, _ = self.fixture()
        for method in ("reinhard_lab", "lab_l_only"):
            self.assertIs(self.run_node(image, image, method=method), image)

    def test_flat_source_uses_mean_shift_without_noise_amplification(self):
        image = torch.full((1, 20, 24, 3), 0.5, dtype=torch.float64)
        image[:, ::2] += 1e-7
        reference, _ = self.fixture()
        result = self.run_node(image, reference, method="reinhard_lab", clip_gamut=False)
        before = wb.rgb_to_lab(image).reshape(-1, 3).std(0, unbiased=False)
        after = wb.rgb_to_lab(result).reshape(-1, 3).std(0, unbiased=False)
        torch.testing.assert_close(after, before, atol=1e-10, rtol=0)

    def test_flat_reference_does_not_erase_source_texture(self):
        image, lab = self.fixture()
        reference = torch.full((1, 10, 12, 3), 0.6, dtype=torch.float64)
        result = self.run_node(image, reference, method="reinhard_lab", clip_gamut=False)
        after = wb.rgb_to_lab(result)
        torch.testing.assert_close(after.reshape(-1, 3).std(0, unbiased=False),
                                   lab.reshape(-1, 3).std(0, unbiased=False), atol=1e-10, rtol=0)

    def test_clip_gamut_only_clips_the_final_output(self):
        image = torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64).expand(1, 8, 8, 3)
        reference = torch.full_like(image, 0.95)
        raw = self.run_node(image, reference, method="lab_l_only", clip_gamut=False)
        self.assertGreater(raw.max().item(), 1)
        clipped = self.run_node(image, reference, method="lab_l_only", clip_gamut=True)
        torch.testing.assert_close(clipped, raw.clamp(0, 1), atol=0, rtol=0)
        partial = self.run_node(image, reference, method="lab_l_only", clip_gamut=True, strength=0.25)
        torch.testing.assert_close(partial, (image + 0.25 * (raw - image)).clamp(0, 1), atol=1e-12, rtol=0)

    def test_combined_method_applies_wb_then_reference_transform(self):
        image, lab = self.fixture()
        reference = wb.lab_to_rgb(lab + lab.new_tensor([3, -3, 2]))
        result = self.run_node(image, reference, method="wb_highlight+reinhard", clip_gamut=False)
        gains = wb._wb_gains(image, True, 95, True)
        base = wb.linear_to_srgb(wb.srgb_to_linear(image) * gains[:, None, None])
        expected = self.run_node(base, reference, method="reinhard_lab", clip_gamut=False)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=0)

    def test_reference_single_image_broadcasts_without_expanding_source(self):
        image, lab = self.fixture()
        batch = torch.cat((image, wb.lab_to_rgb(lab + lab.new_tensor([2, 1, -1]))))
        reference = wb.lab_to_rgb(lab + lab.new_tensor([3, -2, 2]))
        result = self.run_node(batch, reference, method="reinhard_lab")
        self.assertEqual(result.shape[0], 2)
        for i in range(2):
            torch.testing.assert_close(result[i:i + 1], self.run_node(batch[i:i + 1], reference, method="reinhard_lab"))

    def test_pairwise_reference_batch(self):
        image, lab = self.fixture()
        batch = image.expand(2, -1, -1, -1)
        references = torch.cat((image, wb.lab_to_rgb(lab + lab.new_tensor([3, -2, 2]))))
        result = self.run_node(batch, references, method="reinhard_lab")
        for i in range(2):
            torch.testing.assert_close(result[i:i + 1], self.run_node(batch[i:i + 1], references[i:i + 1], method="reinhard_lab"))

    def test_different_reference_dimensions_and_noncontiguous_inputs(self):
        image, _ = self.fixture()
        image = image.transpose(1, 2)
        reference = image[:, ::2, ::3]
        self.assertFalse(image.is_contiguous())
        result = self.run_node(image, reference, method="reinhard_lab")
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_alpha_is_preserved_and_hidden_rgb_ignored(self):
        rgb = torch.tensor([0.75, 0.6, 0.5], dtype=torch.float64).expand(1, 8, 8, 3).clone()
        alpha = torch.ones(1, 8, 8, 1, dtype=torch.float64)
        alpha[:, :4] = 0
        alpha[:, 4] = 0.25
        rgb[:, :4] = torch.tensor([0., 1., 0.])
        image = torch.cat((rgb, alpha), -1)
        result = self.run_node(image, method="wb_grayworld")
        torch.testing.assert_close(result[..., 3:], alpha, atol=0, rtol=0)
        torch.testing.assert_close(result[:, :4, :, :3], rgb[:, :4], atol=0, rtol=0)
        torch.testing.assert_close(result[:, 4:, :, 0], result[:, 4:, :, 2], atol=1e-12, rtol=0)

    def test_reference_hidden_rgb_does_not_poison_matching(self):
        image, _ = self.fixture()
        rgba = torch.cat((image.clone(), torch.ones_like(image[..., :1])), -1)
        rgba[:, :5, :, 3] = 0
        reference = rgba.clone()
        reference[:, :5, :, :3] = torch.tensor([0, 1, 0])
        result = self.run_node(rgba, reference, method="reinhard_lab")
        torch.testing.assert_close(result, rgba, atol=1e-12, rtol=0)

    def test_transparent_source_is_unchanged(self):
        image, _ = self.fixture()
        rgba = torch.cat((image, torch.zeros_like(image[..., :1])), -1)
        for method in wb.METHODS:
            result = self.run_node(rgba, image, method=method)
            torch.testing.assert_close(result, rgba, atol=0, rtol=0)

    def test_premultiplied_analysis_prevents_hidden_color_bleed(self):
        rgba = torch.zeros(1, 64, 64, 4, dtype=torch.float64)
        rgba[..., 1] = 1
        rgba[:, 16:48, 16:48] = torch.tensor([0.8, 0.6, 0.4, 1.])
        sample = wb._analysis_image(rgba, True, 16, 16)
        visible = sample[..., 3] > wb._EPS
        expected = sample.new_tensor([0.8, 0.6, 0.4]).expand_as(sample[..., :3])
        torch.testing.assert_close(sample[..., :3][visible], expected[visible], atol=3e-8, rtol=0)

    def test_chunking_preserves_results(self):
        image, lab = self.fixture()
        reference = wb.lab_to_rgb(lab + lab.new_tensor([3, -2, 2]))
        for method in wb.METHODS:
            with self.subTest(method=method):
                expected = self.run_node(image, reference, method=method)
                with patch.object(wb_engine, "_TILE_PIXELS", 13):
                    result = self.run_node(image, reference, method=method)
                torch.testing.assert_close(result, expected, atol=1e-11, rtol=0)

    def test_float_dtypes_and_device_are_preserved(self):
        image, _ = self.fixture()
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                src = image.to(dtype)
                for method in wb.METHODS:
                    result = self.run_node(src, torch.full_like(src, 0.6), method=method)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.device, src.device)
                    self.assertTrue(torch.isfinite(result).all())

    def test_grayscale_expands_to_rgb(self):
        image = torch.full((1, 8, 8, 1), 0.5)
        result = self.run_node(image, method="wb_grayworld")
        self.assertEqual(result.shape, (1, 8, 8, 3))
        torch.testing.assert_close(result, image.expand_as(result))

    def test_invalid_image_reports_a_helpful_error(self):
        bad = [(torch.zeros(8, 8, 3), "shape"),
               (torch.zeros(1, 8, 8, 2), "shape"),
               (torch.zeros(1, 8, 8, 3, dtype=torch.uint8), "floating-point"),
               (torch.full((1, 8, 8, 3), float("nan")), "NaN"),
               (torch.full((1, 8, 8, 3), 255.), "normalized"),
               (torch.full((1, 8, 8, 3), -0.1), "normalized")]
        for image, message in bad:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_node(image, method="wb_grayworld")

    def test_invalid_controls_report_helpful_errors(self):
        image, _ = self.fixture()
        bad = [({"method": "missing"}, "unknown method"),
               ({"strength": float("nan")}, "strength"),
               ({"strength": 1.1}, "strength"),
               ({"percentile": 100}, "percentile"),
               ({"force_size": True, "target_width": 0}, "dimensions"),
               ({"force_size": True, "target_height": 16.5}, "dimensions")]
        for controls, message in bad:
            with self.subTest(controls=controls):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_node(image, image, **controls)

    def test_missing_reference_and_incompatible_batch_report_errors(self):
        image, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "reference.*floating-point"):
            self.run_node(image, None, method="reinhard_lab")
        with self.assertRaisesRegex(ValueError, "reference batch"):
            self.run_node(image, image.expand(2, -1, -1, -1), method="reinhard_lab")

    def test_transparent_reference_reports_helpful_error(self):
        image, _ = self.fixture()
        reference = torch.cat((image, torch.zeros_like(image[..., :1])), -1)
        with self.assertRaisesRegex(ValueError, "reference has no visible pixels"):
            self.run_node(image, reference, method="reinhard_lab")


if __name__ == "__main__":
    unittest.main()
