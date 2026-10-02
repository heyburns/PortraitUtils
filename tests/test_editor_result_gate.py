"""Routing regressions for the optional editor-to-upscaler safeguard."""

import sys
import unittest
from pathlib import Path

import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes")]

from PortraitUtils.core.editor_profiles import (
    TARGET_DIRECT, TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21,
)
from PortraitUtils.smart_photo_prepare import EditorResultGate


class EditorResultGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = EditorResultGate()
        self.firered = torch.zeros((1, 8, 8, 3))
        self.klein = torch.ones((1, 8, 8, 3))
        self.qwen21 = torch.full((1, 8, 8, 3), 0.5)
        self.direct = torch.full((1, 8, 8, 3), 0.25)
        self.images = {
            "firered_image": self.firered,
            "klein_image": self.klein,
            "qwen21_image": self.qwen21,
            "direct_image": self.direct,
        }

    def test_requires_matching_editor_output_and_preserves_tensor(self):
        for target, expected in (
            (TARGET_FIRERED, self.firered),
            (TARGET_KLEIN, self.klein),
            (TARGET_QWEN21, self.qwen21),
        ):
            with self.subTest(target=target):
                self.assertIs(self.gate.route(target, **self.images)[0], expected)

    def test_missing_editor_never_falls_back_to_other_available_images(self):
        for target, selected_socket in (
            (TARGET_FIRERED, "firered_image"),
            (TARGET_KLEIN, "klein_image"),
            (TARGET_QWEN21, "qwen21_image"),
        ):
            with self.subTest(target=target):
                other_images = dict(self.images)
                del other_images[selected_socket]
                with self.assertRaisesRegex(ValueError, selected_socket + " has no image"):
                    self.gate.route(target, allow_direct=True, **other_images)

    def test_direct_path_requires_explicit_permission(self):
        with self.assertRaisesRegex(ValueError, "no editor is active"):
            self.gate.route(TARGET_DIRECT, **self.images)
        self.assertIs(self.gate.route(TARGET_DIRECT, allow_direct=True,
                                      **self.images)[0], self.direct)

    def test_lazy_evaluates_only_the_selected_branch(self):
        schema = self.gate.INPUT_TYPES()
        self.assertTrue(all(schema["optional"][name][1]["lazy"] for name in (
            "firered_image", "klein_image", "qwen21_image", "direct_image")))
        for target, selected_socket in (
            (TARGET_FIRERED, "firered_image"),
            (TARGET_KLEIN, "klein_image"),
            (TARGET_QWEN21, "qwen21_image"),
            (TARGET_DIRECT, "direct_image"),
        ):
            with self.subTest(target=target):
                other_images = dict(self.images)
                other_images[selected_socket] = None
                self.assertEqual(self.gate.check_lazy_status(
                    target, allow_direct=True, **other_images), [selected_socket])
                self.assertEqual(self.gate.check_lazy_status(
                    target, allow_direct=True,
                    **{selected_socket: self.images[selected_socket]}), [])
        with self.assertRaisesRegex(ValueError, "no editor is active"):
            self.gate.check_lazy_status(TARGET_DIRECT)

    def test_rejects_bad_selected_image_without_touching_other_branches(self):
        with self.assertRaisesRegex(ValueError, "IMAGE layout"):
            self.gate.route(TARGET_FIRERED, firered_image=torch.zeros((8, 8)),
                            klein_image=self.klein)


if __name__ == "__main__":
    unittest.main()
