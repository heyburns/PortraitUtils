"""Regression checks for legacy-node removal and retained configured interfaces."""

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(COMFY_ROOT), str(CUSTOM_NODES)]

import PortraitUtils  # noqa: E402
from PortraitUtils.mask_utils import (  # noqa: E402
    _auto_pick_foreground, _quantile_bounds, _to_numpy_mask,
)
from PortraitUtils.seed_utils import (  # noqa: E402
    SEED_MAX, _clamp_seed, _resolve_seed,
)
from PortraitUtils.workflow_config_v2 import (  # noqa: E402
    AutoAdjustV2, CropConfigV2, CropPromptV2, EditConfigV2, EditPromptV2,
    FinishConfigV2, FinishScaleV2, FinishSeedV2, IntelligentAutoCropV2,
    OutpaintConfigV2, OutpaintPaddingComputeV2, OutpaintPromptV2,
    SourceInfoV2, SourceSaveInfoV2, StitchByMaskV2,
)


RETIRED_IDS = {
    "FluxResolutionPrepare", "MQBBoxMin", "FitAspectHeadSafe",
    "UniversalProjectConfig", "ExtractConfigValue", "AutoColorConfigNode",
    "OutpaintConfigNode", "QwenImageEditNativeReference", "LoadImageCombined",
    "IntelligentAutoCrop", "SmartPhotoPrepare", "OutpaintPaddingComputeNode",
    "AutoAdjustNode", "StitchByMask",
    "SmartPhotoPrepareV2", "PortraitPhotoPrepareCompactV2",
    "IntelligentAutoCropV2", "LoadImageCombinedV2",
    "AutoWBColorMatch", "StitchByMaskV2", "SimpleImageSaver",
    "ProcessScannedPhoto", "PairedImageLoader",
    "CropImageByMargins", "CropMaskByMargins",
}

CURRENT_IDS = {
    "FilenameAppendSuffix",
    "ComparisonGate",
    "PortraitEditorTargetBeacon",
    "PortraitActiveEditorTarget", "PortraitEditorResultGate", "PortraitCropConfigV2", "PortraitEditConfigV2",
    "PortraitOutpaintConfigV2", "PortraitFinishConfigV2",
    "OutpaintPaddingComputeV2",
    "AutoAdjustV2", "PortraitCropPromptV2",
    "PortraitEditPromptV2", "PortraitOutpaintPromptV2", "PortraitFinishSeedV2",
    "PortraitFinishScaleV2", "PortraitSourceSaveInfoV2",
    "PortraitPhotoLoaderPreview",
    "PortraitAutoCropPreview",
    "PortraitWhiteBalancePreview",
    "PortraitStitchPreview", "PortraitImageSaverPreview", "PortraitScannedPhotoPreview",
    "PortraitPairedLoaderPreview", "PortraitCropImageMarginsPreview", "PortraitCropMaskMarginsPreview",
    "PortraitPhotoPrepareV2", "PortraitPrepareDimensionsV2", "PortraitPrepareCropBoxV2",
    "PortraitPrepareScaleV2", "PortraitPrepareSummaryV2",
    "PortraitPhotoPanelSplitV2",
}


class CurrentRegistryTests(unittest.TestCase):
    def test_registry_contains_retained_ids_and_preview_nodes(self):
        self.assertEqual(set(PortraitUtils.NODE_CLASS_MAPPINGS), CURRENT_IDS)

    def test_retired_ids_are_not_exported(self):
        self.assertFalse(RETIRED_IDS & PortraitUtils.NODE_CLASS_MAPPINGS.keys())
        self.assertFalse(RETIRED_IDS & PortraitUtils.NODE_DISPLAY_NAME_MAPPINGS.keys())

    def test_every_registered_node_has_a_display_name(self):
        self.assertEqual(set(PortraitUtils.NODE_DISPLAY_NAME_MAPPINGS), CURRENT_IDS)

    def test_only_production_names_are_registered(self):
        promoted = {
            "PortraitPhotoPrepareV2", "PortraitAutoCropPreview",
            "PortraitPhotoLoaderPreview", "PortraitWhiteBalancePreview",
            "PortraitStitchPreview", "PortraitImageSaverPreview",
            "PortraitScannedPhotoPreview", "PortraitPairedLoaderPreview",
            "PortraitCropImageMarginsPreview", "PortraitCropMaskMarginsPreview",
            "PortraitPhotoPanelSplitV2",
        }
        for node_id in promoted:
            with self.subTest(promoted=node_id):
                self.assertNotIn("Preview", PortraitUtils.NODE_DISPLAY_NAME_MAPPINGS[node_id])
                self.assertFalse(getattr(PortraitUtils.NODE_CLASS_MAPPINGS[node_id],
                                         "DEPRECATED", False))

    def test_registered_schemas_match_execution_signatures(self):
        for node_id, cls in PortraitUtils.NODE_CLASS_MAPPINGS.items():
            with self.subTest(node_id=node_id):
                schema = cls.INPUT_TYPES()
                signature = inspect.signature(getattr(cls, cls.FUNCTION))
                names = set().union(*(schema.get(group, {}) for group in
                                      ("required", "optional", "hidden")))
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in signature.parameters.values()):
                    self.assertFalse(names - signature.parameters.keys())
                if hasattr(cls, "RETURN_NAMES"):
                    self.assertEqual(len(cls.RETURN_NAMES), len(cls.RETURN_TYPES))

    def test_scan_modes_are_retained(self):
        cls = PortraitUtils.NODE_CLASS_MAPPINGS["PortraitScannedPhotoPreview"]
        self.assertEqual(cls.INPUT_TYPES()["required"]["crop_mode"][0],
                         ["Inner Photo Frame", "Scanner Bed Only", "None"])



class RetainedHelpersTests(unittest.TestCase):
    def test_mask_conversion_and_quantiles(self):
        mask = torch.zeros((1, 30, 40))
        mask[:, 5:15, 10:20] = 1
        array = _to_numpy_mask({"mask": mask})
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(array.shape, (30, 40))
        self.assertEqual(_quantile_bounds(array, 0.005, 0.995, 0.005, 0.995, 8),
                         (10, 5, 10, 10))

    def test_foreground_interpretation(self):
        mask = np.zeros((40, 40), dtype=np.float32)
        mask[10:30, 10:30] = 1
        foreground, selection = _auto_pick_foreground(mask, "auto")
        np.testing.assert_array_equal(foreground, mask)
        self.assertEqual(selection, "auto->fg")
        inverted, selection = _auto_pick_foreground(mask, "true")
        np.testing.assert_array_equal(inverted, 1 - mask)
        self.assertEqual(selection, "bg")

    def test_fixed_seed_and_clamping(self):
        self.assertEqual(_resolve_seed(42, None, None, None), 42)
        self.assertEqual(_clamp_seed(-100), 0)
        self.assertEqual(_clamp_seed(SEED_MAX + 1), SEED_MAX)

    def test_random_seed_updates_existing_metadata(self):
        for sentinel in (-1, -2, -3):
            with self.subTest(sentinel=sentinel):
                prompt = {"7": {"inputs": {"seed": sentinel}}}
                extra = {"workflow": {"nodes": [
                    {"id": 7, "widgets_values": [2.0, sentinel]},
                ]}}
                with patch("PortraitUtils.seed_utils._new_random_seed", return_value=123):
                    self.assertEqual(_resolve_seed(sentinel, prompt, extra, "7"), 123)
                self.assertEqual(prompt["7"]["inputs"]["seed"], 123)
                self.assertEqual(extra["workflow"]["nodes"][0]["widgets_values"], [2.0, 123])


class RetainedConfiguredNodesTests(unittest.TestCase):
    def setUp(self):
        self.crop = CropConfigV2(False, False, 0.07, 0.85, 0, False,
                                 "Force 1 MP", "Auto", "subject")
        self.edit = EditConfigV2("edit", 0.4, 1.0, "hair", "background", "rocks")
        self.outpaint = OutpaintConfigV2("Pixels", "center", 20, 10, 1, 3, 2, 4, "expand")
        self.finish = FinishConfigV2(2, 4, False, False, False, False, 42)
        self.image = torch.linspace(0, 1, 16 * 16 * 3).reshape(1, 16, 16, 3)

    def test_typed_readers(self):
        self.assertEqual(CropPromptV2().read(self.crop), ("subject",))
        self.assertEqual(EditPromptV2().read(self.edit, "inpaint_prompt"), ("edit",))
        self.assertEqual(EditPromptV2().read(self.edit, "stitch_prompt_1"), ("hair",))
        self.assertEqual(EditPromptV2().read(self.edit, "stitch_prompt_2"), ("background",))
        self.assertEqual(EditPromptV2().read(self.edit, "stitch_prompt_3"), ("rocks",))
        self.assertEqual(OutpaintPromptV2().read(self.outpaint), ("expand",))
        self.assertEqual(FinishSeedV2().read(self.finish), (42,))
        self.assertEqual(FinishScaleV2().read(self.finish, "final_upscale"), (4.0,))
        self.assertEqual(SourceSaveInfoV2().read(SourceInfoV2("photo", "out", 16, 16)),
                         ("photo", "out"))

    def test_reader_rejects_the_wrong_config_type(self):
        with self.assertRaisesRegex(TypeError, "matching PortraitUtils V2 config"):
            CropPromptV2().read(self.edit)

    def test_disabled_border_crop_preserves_pixels(self):
        result = IntelligentAutoCropV2().run(self.image, self.crop)
        torch.testing.assert_close(result[0], self.image)
        self.assertEqual(result[1:], (0, 0, 0, 0, False))

    def test_padding_compute_still_uses_the_shared_implementation(self):
        self.assertEqual(OutpaintPaddingComputeV2().compute(self.image, self.outpaint),
                         (1, 2, 3, 4))

    def test_disabled_adjustments_preserve_pixels(self):
        node = AutoAdjustV2()
        defaults = {name: spec[1]["default"] for name, spec in
                    node.INPUT_TYPES()["required"].items()
                    if name not in ("image", "finish_config")}
        result = node.apply(self.image, self.finish, **defaults)
        torch.testing.assert_close(result[0], self.image)

    def test_global_blend_still_uses_configured_opacity(self):
        a, b = torch.zeros_like(self.image), torch.ones_like(self.image)
        result = StitchByMaskV2().blend(a, b, self.edit, "blend_opacity",
                                        bypass_mask=True, force_size=False)
        torch.testing.assert_close(result[0], torch.full_like(a, 0.4))


if __name__ == "__main__":
    unittest.main()
