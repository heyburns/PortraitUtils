"""Synthetic border/banner cases and pixel-preservation contract regressions."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes")]

import PortraitUtils  # noqa: E402
from PortraitUtils.auto_crop_preview import PortraitAutoCropPreview  # noqa: E402
from PortraitUtils.workflow_config_v2 import (  # noqa: E402
    CropConfigV2, IntelligentAutoCropV2, SourceInfoV2,
)


class AutoCropPreviewTests(unittest.TestCase):
    def setUp(self):
        self.node = PortraitAutoCropPreview()
        self.config = CropConfigV2(True, True, 0.07, 0.85, 0, False,
                                   "Auto (preserve detail)", "Auto", "person")
        self.rng = np.random.default_rng(123)

    def photo(self, height=128, width=192):
        return self.rng.uniform(0.25, 0.95, (height, width, 3)).astype(np.float32)

    def frame(self, pixels, margins=(11, 13, 17, 19), color=(0, 0, 0)):
        left, top, right, bottom = margins
        height, width = pixels.shape[:2]
        canvas = np.empty((height + top + bottom, width + left + right, 3), dtype=np.float32)
        canvas[:] = color
        canvas[top:top + height, left:left + width] = pixels
        return canvas

    def tensor(self, pixels, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(pixels), dtype=dtype).unsqueeze(0)

    def run_crop(self, pixels, policy="Conservative", **changes):
        image = pixels if isinstance(pixels, torch.Tensor) else self.tensor(pixels)
        return self.node.run(image, replace(self.config, **changes), policy)

    def footer(self, pixels, height=24, text="alamy  Image ID 123456", color=(0, 0, 0)):
        width = pixels.shape[1]
        with Image.new("RGB", (width, height), color) as banner:
            draw = ImageDraw.Draw(banner)
            draw.text((8, max(0, (height - 12) // 2)), text, fill=(255, 255, 255))
            band = np.array(banner, dtype=np.float32) / 255
        return np.concatenate((pixels, band))

    def assert_trim(self, output, margins, expected):
        self.assertEqual(output[1:5], margins)
        self.assertEqual(output[5], any(margins))
        torch.testing.assert_close(output[0], self.tensor(expected), rtol=0, atol=0)

    def test_registered_separately_with_compatible_inputs_and_outputs(self):
        self.assertNotIn("IntelligentAutoCropV2", PortraitUtils.NODE_CLASS_MAPPINGS)
        self.assertIs(PortraitUtils.NODE_CLASS_MAPPINGS["PortraitAutoCropPreview"], PortraitAutoCropPreview)
        self.assertEqual(self.node.RETURN_TYPES, IntelligentAutoCropV2.RETURN_TYPES)
        self.assertEqual(self.node.RETURN_NAMES, IntelligentAutoCropV2.RETURN_NAMES)
        self.assertEqual(self.node.INPUT_TYPES()["required"], IntelligentAutoCropV2.INPUT_TYPES()["required"])

    def test_black_white_gray_and_colored_frames(self):
        pixels = self.photo()
        for color in [(0, 0, 0), (1, 1, 1), (0.4, 0.4, 0.4), (0.12, 0.3, 0.65)]:
            with self.subTest(color=color):
                self.assert_trim(self.run_crop(self.frame(pixels, color=color)), (11, 13, 17, 19), pixels)

    def test_single_edge_and_asymmetric_borders(self):
        pixels = self.photo()
        for margins in [(6, 0, 0, 0), (0, 9, 0, 0), (0, 0, 12, 0), (0, 0, 0, 18), (1, 3, 2, 4)]:
            with self.subTest(margins=margins):
                self.assert_trim(self.run_crop(self.frame(pixels, margins)), margins, pixels)

    def test_no_border_is_exact_identity(self):
        image = self.tensor(self.photo())
        output = self.run_crop(image)
        self.assertIs(output[0], image)
        self.assertEqual(output[1:], (0, 0, 0, 0, False))

    def test_blank_images_are_not_cropped(self):
        for color in [(0, 0, 0), (1, 1, 1), (0.1, 0.1, 0.1), (0.2, 0.4, 0.7)]:
            pixels = np.empty((160, 240, 3), dtype=np.float32)
            pixels[:] = color
            with self.subTest(color=color):
                self.assert_trim(self.run_crop(pixels), (0, 0, 0, 0), pixels)

    def test_smooth_sky_and_dark_gradients_are_not_borders(self):
        for start, stop in [(0.0, 0.5), (0.35, 0.7)]:
            pixels = np.repeat(np.linspace(start, stop, 300, dtype=np.float32)[:, None, None], 180, axis=1)
            pixels = np.repeat(pixels, 3, axis=2)
            with self.subTest(start=start):
                self.assert_trim(self.run_crop(pixels), (0, 0, 0, 0), pixels)

    def test_low_contrast_ambiguous_transition_is_left_alone(self):
        pixels = np.full((128, 192, 3), 0.025, dtype=np.float32)
        source = self.frame(pixels)
        self.assert_trim(self.run_crop(source), (0, 0, 0, 0), source)

    def test_uniform_studio_background_is_not_subject_framing(self):
        pixels = np.ones((160, 192, 3), dtype=np.float32)
        y, x = np.mgrid[:160, :192]
        pixels[(y - 80) ** 2 + (x - 96) ** 2 <= 60 ** 2] = (0.2, 0.3, 0.4)
        self.assert_trim(self.run_crop(pixels), (0, 0, 0, 0), pixels)

    def test_heavy_letterboxing_is_corroborated_in_the_inner_rectangle(self):
        pixels = self.photo(50, 80)
        margins = (80, 80, 80, 80)
        self.assert_trim(self.run_crop(self.frame(pixels, margins)), margins, pixels)

    def test_configured_policy_uses_literal_tolerance_for_noisier_frames(self):
        pixels = self.photo()
        source = self.frame(pixels, color=(0.3, 0.3, 0.3))
        # A repeating mild blemish exceeds the conservative strong-pixel guard,
        # but still fits the explicitly configured border tolerance.
        source[:13, ::40] = 0.49
        config = dict(autocrop_fuzz_tolerance=0.2, autocrop_edge_uniformity=0.90)
        output = self.run_crop(source, "Configured tolerances", **config)
        self.assert_trim(output, (11, 13, 17, 19), pixels)

    def test_recompressed_footer_is_removed_without_resizing(self):
        pixels = self.photo(300, 600)
        source = self.footer(pixels, height=32)
        with BytesIO() as buffer:
            with Image.fromarray((source * 255).astype(np.uint8)) as image:
                image.save(buffer, format="JPEG", quality=90)
            buffer.seek(0)
            with Image.open(buffer) as image:
                decoded = np.array(image, dtype=np.float32) / 255
        output = self.run_crop(decoded, autocrop_detect_borders=False)
        self.assertTrue(output[5])
        self.assertGreaterEqual(output[4], 29)
        self.assertLessEqual(output[4], 32)
        torch.testing.assert_close(output[0], self.tensor(decoded[:-output[4]]), rtol=0, atol=0)

    def test_tiny_bright_subject_tip_stops_scan_before_photo(self):
        pixels = self.photo()
        pixels[0] = 0
        pixels[0, 96, :] = 0.9
        self.assert_trim(self.run_crop(self.frame(pixels)), (11, 13, 17, 19), pixels)

    def test_no_jumping_across_photo_content_to_matching_black_rows(self):
        pixels = self.photo()
        pixels[2:12] = 0
        self.assert_trim(self.run_crop(self.frame(pixels, (0, 9, 0, 0))), (0, 9, 0, 0), pixels)

    def test_noise_in_uniform_border_is_tolerated(self):
        pixels = self.photo()
        source = self.frame(pixels, color=(0.4, 0.4, 0.4))
        noise = self.rng.uniform(-0.007, 0.007, source.shape).astype(np.float32)
        source += noise
        source[13:13 + 128, 11:11 + 192] = pixels
        self.assert_trim(self.run_crop(source), (11, 13, 17, 19), pixels)

    def test_nested_uniform_frames(self):
        pixels = self.photo()
        inner = self.frame(pixels, (7, 9, 11, 13), (1, 1, 1))
        source = self.frame(inner, (3, 4, 5, 6))
        self.assert_trim(self.run_crop(source), (10, 13, 16, 19), pixels)

    def test_alamy_style_footer_is_removed_without_extra_photo_rows(self):
        pixels = self.photo(height=180)
        source = self.footer(pixels)
        self.assert_trim(self.run_crop(source, autocrop_detect_borders=False), (0, 0, 0, 24), pixels)

    def test_dark_gray_footer_with_text_is_supported(self):
        pixels = self.photo(height=180)
        source = self.footer(pixels, color=(20, 20, 20))
        self.assert_trim(self.run_crop(source, autocrop_detect_borders=False), (0, 0, 0, 24), pixels)

    def test_footer_and_four_solid_borders(self):
        pixels = self.photo(height=180)
        source = self.frame(self.footer(pixels), (11, 13, 17, 7))
        self.assert_trim(self.run_crop(source), (11, 13, 17, 31), pixels)

    def test_footer_behind_outer_white_border_is_found_on_next_pass(self):
        pixels = self.photo(height=180)
        source = self.frame(self.footer(pixels), (11, 13, 17, 7), (1, 1, 1))
        self.assert_trim(self.run_crop(source), (11, 13, 17, 31), pixels)

    def test_unbranded_black_strip_is_not_a_text_banner(self):
        pixels = self.photo(height=180)
        source = self.frame(pixels, (0, 0, 0, 24))
        self.assert_trim(self.run_crop(source, autocrop_detect_borders=False), (0, 0, 0, 0), source)
        self.assert_trim(self.run_crop(source), (0, 0, 0, 24), pixels)

    def test_disabling_banner_detection_keeps_footer_text(self):
        pixels = self.photo(height=180)
        source = self.footer(pixels)
        output = self.run_crop(source, autocrop_strip_bottom_banner=False)
        # Solid border detection may remove empty padding below the text, but
        # must not jump through the lettering or remove the entire footer.
        self.assertLess(output[4], 24)
        torch.testing.assert_close(output[0][0, :180], self.tensor(pixels)[0], rtol=0, atol=0)

    def test_dark_photo_details_are_not_enough_for_a_banner(self):
        pixels = np.zeros((180, 192, 3), dtype=np.float32)
        pixels[164:168, 30:34] = 0.95
        pixels[154:158, 120:124] = 0.95
        self.assert_trim(self.run_crop(pixels), (0, 0, 0, 0), pixels)

    def test_single_bright_object_and_solid_bright_band_are_not_text(self):
        for rectangle in [(60, 187, 130, 199), (0, 190, 192, 196)]:
            pixels = self.photo(height=180)
            source = self.frame(pixels, (0, 0, 0, 24))
            x0, y0, x1, y1 = rectangle
            source[y0:y1, x0:x1] = 1
            with self.subTest(rectangle=rectangle):
                self.assert_trim(self.run_crop(source, autocrop_detect_borders=False), (0, 0, 0, 0), source)

    def test_banner_padding_is_retained_and_not_applied_twice(self):
        pixels = self.photo(height=180)
        source = self.footer(pixels)
        expected = source[:-21]
        self.assert_trim(self.run_crop(source, autocrop_pad_px=3), (0, 0, 0, 21), expected)

    def test_padding_applies_to_each_final_border(self):
        pixels = self.photo()
        source = self.frame(pixels)
        self.assert_trim(self.run_crop(source, autocrop_pad_px=3), (8, 10, 14, 16), source[10:-16, 8:-14])

    def test_padding_larger_than_borders_reports_no_applied_crop(self):
        source = self.frame(self.photo())
        self.assert_trim(self.run_crop(source, autocrop_pad_px=30), (0, 0, 0, 0), source)

    def test_both_switches_off_returns_original_even_with_bad_pixel_values(self):
        image = torch.full((1, 16, 16, 3), float("nan"))
        output = self.run_crop(image, autocrop_detect_borders=False, autocrop_strip_bottom_banner=False)
        self.assertIs(output[0], image)
        self.assertEqual(output[1:], (0, 0, 0, 0, False))

    def test_batch_uses_safe_shared_trims_without_black_padding(self):
        first = self.tensor(self.frame(self.photo(), (11, 13, 17, 19)))
        second = self.tensor(self.frame(self.photo(144, 202), (7, 6, 11, 10)))
        batch = torch.cat((first, second))
        output = self.run_crop(batch)
        self.assertEqual(output[1:], (7, 6, 11, 10, True))
        torch.testing.assert_close(output[0], batch[:, 6:-10, 7:-11], rtol=0, atol=0)

    def test_border_free_batch_member_prevents_crop_for_whole_batch(self):
        frame = self.frame(self.photo())
        batch = torch.cat((self.tensor(frame), self.tensor(self.photo(*frame.shape[:2]))))
        output = self.run_crop(batch)
        self.assertIs(output[0], batch)
        self.assertEqual(output[1:], (0, 0, 0, 0, False))

    def test_input_precision_and_device_are_preserved(self):
        for dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            image = self.tensor(self.frame(self.photo()), dtype)
            with self.subTest(dtype=dtype):
                output = self.run_crop(image)
                self.assertEqual(output[0].dtype, dtype)
                self.assertEqual(output[0].device, image.device)
                torch.testing.assert_close(output[0], image[:, 13:-19, 11:-17], rtol=0, atol=0)

    def test_grayscale_and_opaque_rgba_keep_their_channel_count(self):
        rgb = self.frame(self.photo())
        for source in (rgb[..., :1], np.concatenate((rgb, np.ones((*rgb.shape[:2], 1), np.float32)), axis=-1)):
            with self.subTest(channels=source.shape[-1]):
                output = self.run_crop(source)
                self.assertEqual(output[0].shape[-1], source.shape[-1])
                torch.testing.assert_close(output[0], self.tensor(source[13:-19, 11:-17]), rtol=0, atol=0)

    def test_transparency_is_not_treated_as_a_border(self):
        source = self.frame(self.photo())
        source = np.concatenate((source, np.ones((*source.shape[:2], 1), np.float32)), axis=-1)
        source[0, :, 3] = 0
        self.assert_trim(self.run_crop(source), (0, 0, 0, 0), source)

    def test_mask_and_bad_image_shapes_give_actionable_error(self):
        for image in [torch.zeros((1, 30, 40)), torch.zeros((0, 30, 40, 3)),
                      torch.zeros((1, 30, 40, 2)), torch.zeros((1, 30, 40, 3), dtype=torch.uint8), "image"]:
            with self.subTest(image_type=type(image)):
                with self.assertRaisesRegex(ValueError, "loader's image output"):
                    self.node.run(image, self.config)

    def test_nonfinite_and_unnormalized_pixels_are_not_silently_clipped(self):
        for value in (float("nan"), float("inf"), -0.01, 255):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and normalized"):
                    self.run_crop(torch.full((1, 16, 16, 3), value, dtype=torch.float32))

    def test_wrong_config_bus_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "matching PortraitUtils V2 config"):
            self.node.run(self.tensor(self.photo()), SourceInfoV2("image", "", 192, 128))

    def test_invalid_configuration_values_give_named_errors(self):
        for changes, name in [({"autocrop_fuzz_tolerance": float("nan")}, "autocrop_fuzz_tolerance"),
                              ({"autocrop_fuzz_tolerance": 0.6}, "autocrop_fuzz_tolerance"),
                              ({"autocrop_edge_uniformity": -0.1}, "autocrop_edge_uniformity"),
                              ({"autocrop_pad_px": 1.5}, "autocrop_pad_px")]:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, name):
                    self.run_crop(self.photo(), **changes)

    def test_policy_must_be_explicitly_valid(self):
        with self.assertRaisesRegex(ValueError, "border_policy"):
            self.run_crop(self.photo(), "anything")

    def test_smart_crop_aspect_and_resolution_settings_do_not_affect_borders(self):
        source = self.frame(self.photo())
        a = self.run_crop(source)
        b = self.run_crop(source, smart_crop=True, forced_aspect_ratio="2:3", resolution_policy="Force 1.5 MP")
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        self.assertEqual(a[1:], b[1:])

    def test_jpeg_border_does_not_crop_inside_the_photo(self):
        pixels = self.photo()
        source = self.frame(pixels)
        with BytesIO() as buffer:
            with Image.fromarray((source * 255).astype(np.uint8)) as image:
                image.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            with Image.open(buffer) as image:
                decoded = np.array(image, dtype=np.float32) / 255
        output = self.run_crop(decoded)
        # JPEG ringing can leave a few border pixels: preservation is preferred
        # to consuming subject details. Never introduce spatial resampling.
        self.assertTrue(output[5])
        for proposed, actual in zip(output[1:5], (11, 13, 17, 19)):
            self.assertLessEqual(proposed, actual)
        l, t, r, b = output[1:5]
        torch.testing.assert_close(output[0], self.tensor(decoded[t:len(decoded) - b, l:decoded.shape[1] - r]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
