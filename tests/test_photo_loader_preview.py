"""Preview loader regressions using real image files, without models or a GPU."""

import gc
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageCms
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes")]

import PortraitUtils  # noqa: E402
from PortraitUtils.photo_loader_preview import (  # noqa: E402
    EMPTY_SELECTION, PortraitPhotoLoaderPreview, _LIVE_NODES,
)
from PortraitUtils.workflow_config_v2 import (  # noqa: E402
    LoadImageCombinedV2, SourceInfoV2, SourceSaveInfoV2,
)


class PhotoLoaderPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="portrait-loader-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = self.root / "photos"
        self.batch.mkdir()
        self.input_patch = patch("folder_paths.get_input_directory", return_value=str(self.root))
        self.input_patch.start()
        self.addCleanup(self.input_patch.stop)
        self.node = PortraitPhotoLoaderPreview()
        self.node_id = str(self.root)

    def photo(self, name, color=(25, 75, 125), size=(7, 5), folder=None):
        path = (folder or self.root) / name
        with Image.new("RGB", size, color) as image:
            image.save(path)
        return path

    def args(self, **changes):
        args = dict(mode="Single", input_dir="photos", output_dir="", pattern="*",
                    strip_trailing_numbers=False, repeat_last=False,
                    image=EMPTY_SELECTION, unique_id=self.node_id)
        args.update(changes)
        return args

    def single(self, name, **changes):
        return self.node.load(**self.args(image=name, **changes))

    def next_photo(self, node=None, **changes):
        return (node or self.node).load(**self.args(mode="Batch", **changes))[1].filename_no_ext

    def test_only_promoted_loader_is_registered(self):
        self.assertNotIn("LoadImageCombinedV2", PortraitUtils.NODE_CLASS_MAPPINGS)
        self.assertIs(PortraitUtils.NODE_CLASS_MAPPINGS["PortraitPhotoLoaderPreview"], PortraitPhotoLoaderPreview)
        self.assertEqual(self.node.RETURN_TYPES, LoadImageCombinedV2.RETURN_TYPES)
        self.assertEqual(self.node.RETURN_NAMES, LoadImageCombinedV2.RETURN_NAMES)

    def test_float_rgb_shape_and_existing_source_reader(self):
        self.photo("photo (2).png")
        tensor, info = self.single("photo (2).png", strip_trailing_numbers=True, output_dir=" test-out ")
        self.assertEqual(tensor.shape, (1, 5, 7, 3))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(info, SourceInfoV2("photo", "test-out", 7, 5))
        self.assertEqual(SourceSaveInfoV2().read(info), ("photo", "test-out"))
        torch.testing.assert_close(tensor[0, 0, 0], torch.tensor([25, 75, 125]) / 255)

    def test_filename_digits_are_not_copy_markers(self):
        self.photo("scan2024.png")
        self.assertEqual(self.single("scan2024.png", strip_trailing_numbers=True)[1].filename_no_ext, "scan2024")

    def test_empty_input_folder_has_a_usable_schema(self):
        self.assertEqual(self.node.INPUT_TYPES()["required"]["image"][0], [EMPTY_SELECTION])

    def test_upload_picker_filters_non_images_and_sorts_naturally(self):
        self.photo("image10.PNG")
        self.photo("image2.png")
        (self.root / "notes.txt").touch()
        self.assertEqual(self.node.INPUT_TYPES()["required"]["image"][0], ["image2.png", "image10.PNG"])

    def test_all_exif_orientations_and_dimensions(self):
        pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) * 10
        expected = {1: pixels, 2: pixels[:, ::-1], 3: pixels[::-1, ::-1],
                    4: pixels[::-1], 5: pixels.swapaxes(0, 1), 6: np.rot90(pixels, 3),
                    7: pixels.swapaxes(0, 1)[::-1, ::-1], 8: np.rot90(pixels, 1)}
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                exif = Image.Exif()
                exif[274] = orientation
                with Image.fromarray(pixels) as image:
                    image.save(self.root / "oriented.png", exif=exif)
                tensor, info = self.single("oriented.png")
                np.testing.assert_allclose(tensor[0].numpy(), expected[orientation] / 255, atol=1e-7)
                self.assertEqual((info.width, info.height), (tensor.shape[2], tensor.shape[1]))

    def test_transparency_composites_instead_of_discarding_alpha(self):
        with Image.new("RGBA", (3, 2), (255, 0, 0, 128)) as image:
            image.save(self.root / "alpha.png")
        white = self.single("alpha.png")[0][0, 0, 0]
        black = self.single("alpha.png", alpha_background="Black")[0][0, 0, 0]
        torch.testing.assert_close(white, torch.tensor([1, 127 / 255, 127 / 255]))
        torch.testing.assert_close(black, torch.tensor([128 / 255, 0, 0]))

    def test_palette_transparency_is_composited(self):
        with Image.new("P", (2, 2), 0) as image:
            image.putpalette([255, 0, 0] + [0] * 765)
            image.save(self.root / "palette.png", transparency=0)
        self.assertTrue(torch.all(self.single("palette.png")[0] == 1))

    def test_embedded_lab_profile_converts_to_srgb(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB"))
        with Image.new("LAB", (3, 2), (140, 140, 100)) as image:
            image.save(self.root / "profile.tif", icc_profile=profile.tobytes())
            with ImageCms.profileToProfile(image, profile, ImageCms.createProfile("sRGB"), outputMode="RGB") as expected:
                values = np.array(expected, dtype=np.float32) / 255
        np.testing.assert_allclose(self.single("profile.tif")[0][0].numpy(), values, atol=1e-7)

    def test_invalid_profile_produces_helpful_error_and_explicit_bypass(self):
        with Image.new("RGB", (3, 2), (20, 40, 60)) as image:
            image.save(self.root / "bad-profile.png", icc_profile=b"not a profile")
        with self.assertRaisesRegex(ValueError, "Keep encoded values"):
            self.single("bad-profile.png")
        self.assertEqual(self.single("bad-profile.png", color_management="Keep encoded values")[0].shape, (1, 2, 3, 3))

    def test_16bit_rgb_png_preserves_low_order_samples(self):
        data = np.full((2, 3, 3), [123, 32769, 64001], dtype=np.uint16)
        self.assertTrue(cv2.imwrite(str(self.root / "rgb16.png"), data))
        tensor, _ = self.single("rgb16.png")
        np.testing.assert_allclose(tensor[0].numpy(), data[..., ::-1].astype(np.float32) / 65535, atol=1e-7)

    def test_16bit_grayscale_uses_fixed_full_scale_not_image_max(self):
        data = np.array([[0, 1000], [32769, 65535]], dtype=np.uint16)
        with Image.fromarray(data) as image:
            image.save(self.root / "gray16.tif")
        tensor, _ = self.single("gray16.tif")
        np.testing.assert_allclose(tensor[0, ..., 0].numpy(), data.astype(np.float32) / 65535, atol=1e-7)
        torch.testing.assert_close(tensor[..., 0], tensor[..., 1])

    def test_16bit_rgb_tiff_preserves_color_order_and_precision(self):
        data = np.full((2, 3, 3), [123, 32769, 64001], dtype=np.uint16)
        self.assertTrue(cv2.imwrite(str(self.root / "rgb16.tif"), data))
        tensor, _ = self.single("rgb16.tif")
        np.testing.assert_allclose(tensor[0].numpy(), data[..., ::-1].astype(np.float32) / 65535, atol=1e-7)

    def test_16bit_tiff_planar_rgb(self):
        import tifffile
        data = np.full((3, 2, 3), 0, dtype=np.uint16)
        data[0], data[1], data[2] = 123, 32769, 64001
        tifffile.imwrite(self.root / "planar16.tif", data, photometric="rgb", planarconfig="separate")
        tensor, _ = self.single("planar16.tif")
        np.testing.assert_allclose(tensor[0].numpy(), np.moveaxis(data, 0, -1) / 65535, atol=1e-7)

    def test_16bit_tiff_white_is_zero(self):
        import tifffile
        data = np.array([[0, 65535]], dtype=np.uint16)
        tifffile.imwrite(self.root / "white16.tif", data, photometric="miniswhite")
        tensor, _ = self.single("white16.tif")
        np.testing.assert_allclose(tensor[0, ..., 0].numpy(), [[1, 0]], atol=1e-7)

    def test_16bit_tiff_premultiplied_alpha_is_not_applied_twice(self):
        import tifffile
        data = np.full((2, 3, 4), [32768, 0, 0, 32768], dtype=np.uint16)
        tifffile.imwrite(self.root / "premult16.tif", data, photometric="rgb", extrasamples="assocalpha")
        tensor, _ = self.single("premult16.tif")
        torch.testing.assert_close(tensor[0, 0, 0], torch.tensor([1, 32767 / 65535, 32767 / 65535]))

    def test_16bit_alpha_compositing(self):
        data = np.full((2, 3, 4), [0, 0, 65535, 32768], dtype=np.uint16)
        self.assertTrue(cv2.imwrite(str(self.root / "alpha16.png"), data))
        torch.testing.assert_close(self.single("alpha16.png")[0][0, 0, 0],
                                   torch.tensor([1, 32767 / 65535, 32767 / 65535]))

    def test_16bit_png_exif_orientation(self):
        data = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3) * 1000
        exif = Image.Exif()
        exif[274] = 6
        with Image.fromarray(data) as image:
            image.save(self.root / "oriented16.png", exif=exif)
        tensor, info = self.single("oriented16.png")
        np.testing.assert_allclose(tensor[0, ..., 0].numpy(), np.rot90(data, 3) / 65535, atol=1e-7)
        self.assertEqual((info.width, info.height), (2, 3))

    def test_16bit_tiff_exif_orientation_is_applied_exactly_once(self):
        data = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3) * 1000
        expected = {1: data, 2: data[:, ::-1], 3: data[::-1, ::-1],
                    4: data[::-1], 5: data.swapaxes(0, 1), 6: np.rot90(data, 3),
                    7: data.swapaxes(0, 1)[::-1, ::-1], 8: np.rot90(data, 1)}
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                with Image.fromarray(data) as image:
                    image.save(self.root / "oriented16.tif", tiffinfo={274: orientation})
                tensor, _ = self.single("oriented16.tif")
                np.testing.assert_allclose(tensor[0, ..., 0].numpy(), expected[orientation] / 65535, atol=1e-7)

    def test_8bit_srgb_profile_preserves_rgb_values(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        with Image.new("RGB", (3, 2), (42, 129, 202)) as image:
            image.save(self.root / "srgb.png", icc_profile=profile.tobytes())
        torch.testing.assert_close(self.single("srgb.png")[0][0, 0, 0], torch.tensor([42, 129, 202]) / 255)

    def test_16bit_tagged_images_do_not_silently_lose_precision(self):
        # Deliberately invalid tag also tests the explicit encoded-values bypass.
        profile = b"invalid embedded profile"
        with Image.fromarray(np.full((2, 3), 32769, dtype=np.uint16)) as image:
            image.save(self.root / "tagged16.tif", icc_profile=profile)
        with self.assertRaisesRegex(ValueError, "conversion would lose precision"):
            self.single("tagged16.tif")
        self.assertAlmostEqual(float(self.single("tagged16.tif", color_management="Keep encoded values")[0][0, 0, 0, 0]), 32769 / 65535)

    def test_hdr_fails_with_export_guidance(self):
        with Image.fromarray(np.ones((2, 3), dtype=np.float32)) as image:
            image.save(self.root / "hdr.tif")
        with self.assertRaisesRegex(ValueError, "32-bit/HDR"):
            self.single("hdr.tif")

    def test_batch_is_naturally_sorted_and_wraps(self):
        self.photo("image10.png", folder=self.batch)
        self.photo("image2.png", folder=self.batch)
        self.assertEqual([self.next_photo() for _ in range(3)], ["image2", "image10", "image2"])

    def test_two_loader_instances_have_independent_cursors(self):
        for name in ("a.png", "b.png", "c.png"):
            self.photo(name, folder=self.batch)
        other = PortraitPhotoLoaderPreview()
        self.assertEqual(self.next_photo(), "a")
        self.assertEqual(self.next_photo(), "b")
        self.assertEqual(self.next_photo(other, unique_id=self.node_id + "-other"), "a")
        self.assertEqual(self.next_photo(), "c")

    def test_repeat_holds_last_success_and_initial_repeat_loads_first(self):
        for name in ("a.png", "b.png"):
            self.photo(name, folder=self.batch)
        self.assertEqual(self.next_photo(repeat_last=True), "a")
        self.assertEqual(self.next_photo(), "b")
        self.assertEqual(self.next_photo(repeat_last=True), "b")

    def test_failed_load_does_not_skip_a_batch_photo(self):
        self.photo("a.png", folder=self.batch)
        (self.batch / "b.png").touch()
        self.photo("c.png", folder=self.batch)
        self.assertEqual(self.next_photo(), "a")
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "Batch position has not advanced"):
                self.next_photo()
        self.assertEqual(self.next_photo(repeat_last=True), "a")
        self.photo("b.png", folder=self.batch)
        self.assertEqual(self.next_photo(), "b")

    def test_listing_changes_keep_position_by_filename(self):
        self.photo("b.png", folder=self.batch)
        self.photo("c.png", folder=self.batch)
        self.assertEqual(self.next_photo(), "b")
        self.photo("a.png", folder=self.batch)
        self.assertEqual(self.next_photo(repeat_last=True), "b")
        self.assertEqual(self.next_photo(), "c")
        (self.batch / "c.png").unlink()
        self.assertEqual(self.next_photo(repeat_last=True), "a")

    def test_filename_options_do_not_reset_batch_cursor(self):
        self.photo("a (1).png", folder=self.batch)
        self.photo("b (2).png", folder=self.batch)
        self.assertEqual(self.next_photo(), "a (1)")
        self.assertEqual(self.next_photo(strip_trailing_numbers=True, output_dir="different"), "b")

    def test_single_mode_does_not_reset_batch_position(self):
        self.photo("single.png")
        self.photo("a.png", folder=self.batch)
        self.photo("b.png", folder=self.batch)
        self.assertEqual(self.next_photo(), "a")
        self.single("single.png")
        self.assertEqual(self.next_photo(), "b")

    def test_recursive_patterns_and_absolute_folders(self):
        nested = self.batch / "nested"
        nested.mkdir()
        self.photo("one.png", folder=nested)
        self.assertEqual(self.next_photo(input_dir=str(self.batch), pattern="**/*.png"), "one")

    def test_batch_validation_ignores_missing_single_selection(self):
        self.photo("a.png", folder=self.batch)
        self.assertIs(self.node.VALIDATE_INPUTS("Batch", "photos", "*", "deleted.png"), True)
        self.assertEqual(self.next_photo(image="deleted.png"), "a")

    def test_validation_errors_explain_missing_files_folders_and_patterns(self):
        for args, text in [(("Single", "", "*", EMPTY_SELECTION), "select or upload"),
                           (("Single", "", "*", "gone.png"), "no longer exists"),
                           (("Batch", "", "*", ""), "requires input_dir"),
                           (("Batch", "missing", "*", ""), "folder does not exist"),
                           (("Batch", "photos", "*.jpg", ""), "no supported images"),
                           (("Batch", "photos", "../*.png", ""), "relative filename glob"),
                           (("Wrong", "", "*", ""), "Single or Batch")]:
            with self.subTest(args=args):
                self.assertIn(text, self.node.VALIDATE_INPUTS(*args))

    def test_linked_values_can_be_validated_at_execution(self):
        self.assertIs(self.node.VALIDATE_INPUTS("Batch", None, None, ""), True)
        with self.assertRaisesRegex(ValueError, "must be text"):
            self.next_photo(input_dir=12)

    def test_single_hash_detects_same_size_same_timestamp_replacement(self):
        path = self.photo("same.bmp", color=(10, 20, 30))
        stat = path.stat()
        before = self.node.IS_CHANGED(**self.args(image="same.bmp"))
        self.photo("same.bmp", color=(30, 20, 10))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(path.stat().st_size, stat.st_size)
        self.assertNotEqual(self.node.IS_CHANGED(**self.args(image="same.bmp")), before)

    def test_batch_advance_is_never_cached_and_repeat_has_stable_content_hash(self):
        for name in ("a.png", "b.png"):
            self.photo(name, folder=self.batch)
        advancing = self.args(mode="Batch")
        holding = self.args(mode="Batch", repeat_last=True)
        self.assertTrue(math.isnan(self.node.IS_CHANGED(**advancing)))
        first = self.node.IS_CHANGED(**holding)
        self.assertEqual(self.next_photo(repeat_last=True), "a")
        self.assertEqual(self.node.IS_CHANGED(**holding), first)
        self.assertEqual(self.next_photo(), "b")
        second = self.node.IS_CHANGED(**holding)
        self.assertNotEqual(second, first)
        self.assertEqual(self.next_photo(repeat_last=True), "b")
        self.assertEqual(self.node.IS_CHANGED(**holding), second)
        self.photo("b.png", color=(240, 50, 60), folder=self.batch)
        self.assertNotEqual(self.node.IS_CHANGED(**holding), second)

    def test_missing_image_invalidates_cache_instead_of_returning_stale_output(self):
        self.assertTrue(math.isnan(self.node.IS_CHANGED(**self.args(image="gone.png"))))

    def test_node_state_is_weakly_held(self):
        self.photo("a.png", folder=self.batch)
        other = PortraitPhotoLoaderPreview()
        key = self.node_id + "-disposable"
        self.next_photo(other, unique_id=key)
        self.assertIn(key, _LIVE_NODES)
        del other
        gc.collect()
        self.assertNotIn(key, _LIVE_NODES)


if __name__ == "__main__":
    unittest.main()
