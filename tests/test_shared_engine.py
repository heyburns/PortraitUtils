"""Shared-engine contracts, planner replay, compact readers, and interface freeze."""
import ast
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path[:0] = [str(COMFY_ROOT), str(ROOT.parent)]

import PortraitUtils
from PortraitUtils.core import preparation as engine
from PortraitUtils.core import tensors
from PortraitUtils.core.adjustments import AutoAdjustmentEngine, _percentiles_exact
from PortraitUtils.core.contracts import CropConfigV2, EditConfigV2, FinishConfigV2, OutpaintConfigV2, validate_config
from PortraitUtils.core.editor_profiles import TARGET_DIRECT, TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21, _build_resolutions
from PortraitUtils.core.export import ExportEnvironment, PhotoExportEngine
from PortraitUtils.core.outpaint import compute_padding
from PortraitUtils.core.preparation import PreparationOptions, MaskAnalysis, apply_plan, plan_photo, prepare_photo
from PortraitUtils.core.state import TransactionalCursor
from PortraitUtils.core import resampling
from PortraitUtils.preparation_nodes_v2 import (
    PhotoPrepareStandardV2, PrepareDimensionsV2, PrepareCropBoxV2, PrepareScaleV2, PrepareSummaryV2,
)
from PortraitUtils.workflow_config_v2 import (
    CropConfigNodeV2, EditConfigNodeV2, FinishConfigNodeV2, EditPromptV2, FinishScaleV2,
)


# Interface snapshots updated for intentional changes. Order is serialized.
INTERFACES = """
PortraitAutoCropPreview 8ce7c2ae18f375dc4d4b1796c00be00e49ee5bf391685a700f9ebbc3fbaa54cc
ComparisonGate 2ec2ab825f426b58a0d7777d3c4019c06c261794c2943d16c35a41f06a2dfc15
PortraitCropImageMarginsPreview ae5cfea78ea14459292990c06c0f3a6aad2ffbfbd1a9c751d2369ab95431e316
PortraitCropMaskMarginsPreview 202cdfbf43167bc763fc19df79df44c1007feff1303e985d25f7a5a3dea5c2b1
FilenameAppendSuffix 773585fb9d581946620f1d39cc980d01de87d369951b3bf25443692c94daad66
PortraitPairedLoaderPreview 3813c17cb23c45895705202defa9efd9dbe0bc2ab13954e6ade368182b28726e
PortraitPhotoLoaderPreview faa9375e1b6466b0cab5c3149c2e8764d711120756ca146db2936a0e9c4d5e44
PortraitImageSaverPreview d0d2d360d8f9a8e60fe2486aea58000132ffddb26b5e66d06967709e00cc512b
PortraitScannedPhotoPreview 57a3015b530e238ca6ae04ee387b006fc6a013a2186035c252af38fe7bd4ad8a
PortraitStitchPreview c6cd7f804e4152ed74428a15ee477d12662539c2203175a6d447e536dadfc13f
PortraitEditorTargetBeacon 95117443593b832d7eae8e3ae2bdb84a34249aaf01298a43f44582927b5d9feb
PortraitActiveEditorTarget 223ffb9f3f40efa0da56454d11e2977d209f6eddf0b612cbff3c9dd73d562305
PortraitWhiteBalancePreview 4970b23acb91dca8905c9c96dc932bc70c99cc470304e28774c66f443a8318bf
PortraitCropConfigV2 2466f3e21f070a741227d5a748e2423241d8df15cb046fb9a787a68dda591964
PortraitEditConfigV2 e07218698190064fe43b81288616df5866b3c691e14b2f07d14a21000de6eb0e
PortraitOutpaintConfigV2 475aa00be47b8ed5ebc5aa9a8b0e7779870d51d8d1c2e3f73e94446fc2d201a2
PortraitFinishConfigV2 6c862da09465eeb2b0d77ee198db1e821ca1b2b3adda01073efffee7ad3f3f6f
OutpaintPaddingComputeV2 e58e0d2ac0d6ff32786807cdc4279e0b2fcbee99a04e1b7d7b13ef92fd5c030f
AutoAdjustV2 8cc298fb1143c54d507285d42c1b2683edc69dd6fbbb8d489a482274dc767ab8
PortraitCropPromptV2 e8caebd7efeb08bd912f386d1a10ba64be5630430538d56e39475c015e456e73
PortraitEditPromptV2 7a7dc4c31b4338d74ad058981138f9c19a6851b39e86e904ec60fc5546dbc9b6
PortraitOutpaintPromptV2 34faf854d226ab63b756317410fbfde9ddb6215066bfc8099db759d79fbfe1ac
PortraitFinishSeedV2 a28fc6954afacf17f256bd1d3b52ce00ed25141a8c357803c888f9ac45eaa498
PortraitFinishScaleV2 be03b385d6a4d9e59d899ac611db09529d56f6eb29006f25cb960ce92ca8699d
PortraitSourceSaveInfoV2 0680839a5b42eef3c0da5e1a213ee7fa5bf7d075c766e6f117bd42d857137637
"""


class SharedEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.image = torch.linspace(0.1, 0.9, 80 * 120 * 3, dtype=torch.float64).reshape(1, 120, 80, 3)
        self.options = PreparationOptions(resize_method="bilinear", resolution_policy="Force 1 MP")
        self.crop_config = CropConfigV2(False, False, .07, .85, 0, False, "Force 1 MP", "Auto", "person")

    def test_registered_node_interfaces_match_snapshots(self):
        for line in INTERFACES.splitlines():
            if not line:
                continue
            name, expected = line.split()
            cls = PortraitUtils.NODE_CLASS_MAPPINGS[name]
            with self.subTest(node=name):
                schema = cls.INPUT_TYPES()
                # New optional controls may be appended without moving legacy widgets.
                # Their schema/defaults are covered by the padding-fill tests.
                if name == "PortraitCropConfigV2":
                    schema["optional"].pop("padding_fill")
                    if not schema["optional"]:
                        del schema["optional"]
                # Upload dropdown contents vary with user files, not suite code.
                if name == "PortraitPhotoLoaderPreview":
                    schema["required"]["image"] = (["(no images uploaded)"], {"image_upload": True})
                    expected = "d8b32cc7758aa5c85278fa65ffa5c426c0eb759503150cde29d6ccc177208f8b"
                value = [schema, cls.RETURN_TYPES, getattr(cls, "RETURN_NAMES", None), cls.FUNCTION]
                actual = hashlib.sha256(json.dumps(value, sort_keys=False, default=str).encode()).hexdigest()
                self.assertEqual(actual, expected)

    def test_core_has_no_node_or_comfy_imports(self):
        for file in ROOT.joinpath("core").glob("*.py"):
            with self.subTest(file=file.name):
                tree = ast.parse(file.read_text())
                self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.level > 1 for n in ast.walk(tree)))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn((node.module or "").split(".")[0], ("folder_paths", "comfy"))
                    elif isinstance(node, ast.Import):
                        self.assertFalse({n.name.split(".")[0] for n in node.names} & {"comfy", "folder_paths"})

    def test_core_runs_without_comfyui_package_initialization(self):
        script = f"""
import sys, types, importlib
package = types.ModuleType('PortraitUtils')
package.__path__ = [{str(ROOT)!r}]
sys.modules['PortraitUtils'] = package
for module in ('contracts', 'geometry', 'preparation', 'adjustments', 'borders', 'white_balance',
               'composite', 'scans', 'margins', 'photo_io', 'pairing', 'state', 'export', 'outpaint', 'panels'):
    importlib.import_module('PortraitUtils.core.' + module)
assert 'comfy' not in sys.modules and 'folder_paths' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_noop_preserves_exact_pixels_dtype_device_and_channels(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for channels in (1, 3, 4):
                with self.subTest(dtype=dtype, channels=channels):
                    image = torch.rand(1, 18, 12, channels).to(dtype)
                    result = prepare_photo(image, TARGET_DIRECT, self.options)
                    self.assertEqual(result.image.data_ptr(), image.data_ptr())
                    self.assertEqual(result.image.dtype, dtype)
                    self.assertEqual(result.image.device, image.device)
                    torch.testing.assert_close(result.image, image, atol=0, rtol=0)

    def test_resize_retains_precision_and_alpha_layout(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            image = torch.rand(1, 8, 6, 4).to(dtype)
            result = prepare_photo(image, TARGET_FIRERED, self.options)
            self.assertEqual(result.image.shape[-1], 4)
            self.assertEqual(result.image.dtype, dtype)
            self.assertEqual(result.native_crop.dtype, dtype)
            self.assertTrue(torch.isfinite(result.image).all())

    def test_premultiplied_resize_avoids_hidden_color_halos(self):
        image = torch.tensor([[[[1., 0, 0, 1], [0, 1, 0, 0]]]], dtype=torch.float64)
        output = tensors.resize_image(image, 7, 3, "bilinear")
        visible = output[..., 3] > 1e-8
        torch.testing.assert_close(output[..., :3][visible], image.new_tensor([1, 0, 0]).expand(visible.sum(), 3))

    def test_native_crop_is_an_exact_source_slice(self):
        result = prepare_photo(self.image, TARGET_FIRERED, replace(self.options, forced_aspect_ratio="1:1"))
        c = result.info.plan.crop
        torch.testing.assert_close(result.native_crop, self.image[:, c.y:c.y+c.height, c.x:c.x+c.width], atol=0, rtol=0)
        self.assertEqual(result.native_crop.untyped_storage().data_ptr(), self.image.untyped_storage().data_ptr())

    def test_masks_of_width_one_are_bhw_not_hwc(self):
        image, mask = torch.zeros(1, 23, 1, 3), torch.ones(1, 23, 1)
        output = prepare_photo(image, TARGET_DIRECT, self.options, mask)
        self.assertEqual(output.mask.shape, (1, 23, 1))
        torch.testing.assert_close(output.mask, mask, atol=0, rtol=0)

    def test_prepared_mask_dtype_and_noop_storage_are_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            mask = torch.rand(1, 120, 80).to(dtype)
            result = prepare_photo(self.image, TARGET_DIRECT, self.options, mask)
            self.assertEqual(result.mask.dtype, dtype)
            self.assertEqual(result.mask.data_ptr(), mask.data_ptr())
            torch.testing.assert_close(result.mask, mask, atol=0, rtol=0)

    def test_missing_subject_mask_does_not_allocate_or_resize_source_mask(self):
        with patch.object(engine, "aligned_mask", side_effect=AssertionError("unneeded alignment")), \
                patch.object(engine, "resize_mask", side_effect=AssertionError("resized empty mask")):
            output = prepare_photo(self.image, TARGET_FIRERED, self.options)
        self.assertEqual(output.mask.shape, output.image.shape[:3])
        self.assertFalse(output.mask.any())

    def test_mask_size_mismatch_is_explicit_not_silently_resized(self):
        for label, kwargs in (("subject mask", {"mask": torch.zeros(1, 119, 80)}),
                              ("protected-region mask", {"protected_region_mask": torch.zeros(1, 119, 80)})):
            with self.assertRaisesRegex(ValueError, label + ".*Align"):
                prepare_photo(self.image, TARGET_FIRERED, self.options, **kwargs)

    def test_no_image_color_conversion_in_prepare(self):
        with patch("PortraitUtils.image_utils.enforce_image_format", side_effect=AssertionError("legacy conversion")):
            result = prepare_photo(self.image, TARGET_DIRECT, self.options)
        torch.testing.assert_close(result.image, self.image, atol=0, rtol=0)

    def test_plan_replay_matches_preparation(self):
        result = prepare_photo(self.image, TARGET_FIRERED, replace(self.options, forced_aspect_ratio="2:3"))
        image, native, mask = apply_plan(self.image, None, result.info.plan)
        for actual, expected in ((image, result.image), (native, result.native_crop), (mask, result.mask)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_plan_and_config_are_immutable(self):
        result = prepare_photo(self.image, TARGET_DIRECT, self.options)
        for value, field in ((result.info, "plan"), (result.info.plan, "source_width"),
                             (result.info.plan.crop, "x"), (self.options, "smart_crop"),
                             (self.crop_config, "subject_mask")):
            with self.assertRaises(FrozenInstanceError):
                setattr(value, field, None)

    def test_info_contains_no_pixel_tensors_and_serializes_to_json(self):
        result = prepare_photo(self.image, TARGET_FIRERED, self.options)
        serialized = json.loads(PrepareSummaryV2().read(result.info, "JSON")[0])
        self.assertEqual(serialized["schema_version"], 1)
        self.assertEqual(serialized["plan"]["source_width"], 80)
        self.assertEqual(serialized["plan"]["subject_status"], "not connected")
        self.assertIn("pixel_changes", serialized)
        def check(value):
            self.assertNotIsInstance(value, (torch.Tensor, np.ndarray))
            if isinstance(value, dict):
                for v in value.values(): check(v)
            elif isinstance(value, (list, tuple)):
                for v in value: check(v)
        check(asdict(result.info))

    def test_serialization_is_not_a_mutable_backdoor(self):
        info = prepare_photo(self.image, TARGET_FIRERED, self.options).info
        serialized = info.to_dict()
        serialized["plan"]["crop"]["x"] = 999
        self.assertNotEqual(info.plan.crop.x, 999)

    def test_source_output_coordinate_maps_are_inverse(self):
        p = prepare_photo(self.image, TARGET_FIRERED, replace(self.options, forced_aspect_ratio="1:1")).info.plan
        for point in ((0, 0), (79, 119), (22.2, 52.5)):
            output = p.source_to_output(*point)
            recovered = p.output_to_source(*output)
            self.assertAlmostEqual(recovered[0], point[0], places=12)
            self.assertAlmostEqual(recovered[1], point[1], places=12)

    def test_invalid_plan_rejected_before_resampling(self):
        p = prepare_photo(self.image, TARGET_FIRERED, self.options).info.plan
        cases = ((replace(p, crop=replace(p.crop, x=80)), "plan.crop.x"),
                 (replace(p, crop=replace(p.crop, padding=[0, 0, 0, 0])), "immutable"),
                 (replace(p, crop=replace(p.crop, resolution=replace(p.crop.resolution, width=1000))), "native"),
                 (replace(p, source_width=81), "source size"))
        for plan, message in cases:
            with self.subTest(message=message), patch.object(engine, "resize_image", side_effect=AssertionError("resized")):
                with self.assertRaisesRegex(ValueError, message):
                    apply_plan(self.image, None, plan)

    def test_plan_cannot_crop_protected_pixels_even_with_smart_crop_off(self):
        p = prepare_photo(self.image, TARGET_DIRECT, self.options).info.plan
        p = replace(p, protected_bbox=(20, 5, 20, 20))
        c = replace(p.crop, y=6, height=114, resolution=replace(p.crop.resolution, height=114))
        with self.assertRaisesRegex(ValueError, "protected region"):
            apply_plan(self.image, None, replace(p, crop=c))

    def test_plan_cannot_move_to_center_inside_protected_safety_margin(self):
        p = prepare_photo(self.image, TARGET_DIRECT, self.options).info.plan
        p = replace(p, protected_bbox=(20, 20, 20, 20))
        c = replace(p.crop, y=20, height=100, resolution=replace(p.crop.resolution, height=100))
        with self.assertRaisesRegex(ValueError, "safety margin"):
            apply_plan(self.image, None, replace(p, crop=c))

    def test_public_planner_needs_no_image(self):
        analysis = MaskAnalysis(None, None, None, "no masks", "not connected", "not connected")
        plan = plan_photo(3000, 4500, TARGET_FIRERED, analysis, self.options)
        self.assertEqual(plan.crop.resolution.tier, "1 MP")

    def test_all_editors_policies_and_forced_aspects_use_native_profiles(self):
        analysis = MaskAnalysis(None, None, None, "no masks", "not connected", "not connected")
        for target in (TARGET_FIRERED, TARGET_KLEIN, TARGET_QWEN21):
            policies = ("Force 1 MP", "Force 1.5 MP")
            if target == TARGET_QWEN21:
                policies += ("Force 2K (~4 MP)",)
            for policy in policies:
                for aspect in ("Auto", "1:1", "2:3", "3:4", "9:16"):
                    with self.subTest(target=target, policy=policy, aspect=aspect):
                        opts = replace(self.options, resolution_policy=policy, forced_aspect_ratio=aspect)
                        p = plan_photo(3000, 4500, target, analysis, opts)
                        self.assertIn(p.crop.resolution, _build_resolutions(target, 2/3, None, aspect))
                        self.assertEqual(p.crop.resolution.tier, policy.removeprefix("Force "))
                        if target == TARGET_QWEN21:
                            self.assertEqual(p.target_width % 32, 0)
                            self.assertEqual(p.target_height % 32, 0)
                        if target == TARGET_FIRERED:
                            self.assertEqual(p.target_width % 16, 0)
                            self.assertEqual(p.target_height % 16, 0)
                        if aspect != "Auto":
                            a, b = map(int, aspect.split(":"))
                            self.assertEqual(p.target_width * b, p.target_height * a)

    def test_firered_auto_uses_workflow_sizes_and_larger_output_is_opt_in(self):
        analysis = MaskAnalysis(None, None, None, "no masks", "not connected", "not connected")
        for policy in ("Auto (preserve detail)", "Auto (closest scale)"):
            plan = plan_photo(3000, 4500, TARGET_FIRERED, analysis,
                              replace(self.options, resolution_policy=policy))
            self.assertEqual((plan.target_width, plan.target_height), (832, 1248))
            self.assertEqual(plan.crop.resolution.tier, "1 MP")
        larger = plan_photo(3000, 4500, TARGET_FIRERED, analysis,
                           replace(self.options, resolution_policy="Force 1.5 MP"))
        self.assertEqual(larger.crop.resolution.tier, "1.5 MP")
        self.assertGreater(larger.target_width * larger.target_height, 1_400_000)
        self.assertIn("experimental", larger.resolution)
        # A serialized larger plan must not bypass the automatic size policy.
        with self.assertRaisesRegex(ValueError, "eligible"):
            replace(larger, options=replace(larger.options,
                                           resolution_policy="Auto (preserve detail)")).validate()
        with self.assertRaisesRegex(ValueError, "only for Qwen Image Edit 2.1"):
            plan_photo(3000, 4500, TARGET_FIRERED, analysis,
                       replace(self.options, resolution_policy="Force 2K (~4 MP)"))

    def test_retired_2511_target_cannot_silently_select_another_editor(self):
        analysis = MaskAnalysis(None, None, None, "no masks", "not connected", "not connected")
        with self.assertRaisesRegex(ValueError, "2511 has been retired.*Beacon"):
            plan_photo(1000, 1500, "Qwen Image Edit 2511", analysis, self.options)

    def test_qwen21_auto_uses_2k_to_avoid_downscaling(self):
        analysis = MaskAnalysis(None, None, None, "no masks", "not connected", "not connected")
        plan = plan_photo(1728, 2304, TARGET_QWEN21, analysis,
                          replace(self.options, resolution_policy="Auto (preserve detail)"))
        self.assertEqual(plan.crop.resolution.tier, "2K (~4 MP)")
        self.assertGreaterEqual(plan.crop.scale, 1.0)
        self.assertEqual(plan.target_width % 32, 0)
        self.assertEqual(plan.target_height % 32, 0)
        self.assertEqual(plan.crop.resolution, _build_resolutions(TARGET_QWEN21, 0.75, None)[2])
        with self.assertRaisesRegex(ValueError, "only for Qwen Image Edit 2.1"):
            plan_photo(1728, 2304, TARGET_KLEIN, analysis,
                       replace(self.options, resolution_policy="Force 2K (~4 MP)"))

    def test_protected_mask_remains_active_in_standard_lazy_mode(self):
        for smart in (True, False):
            config = replace(self.crop_config, smart_crop=smart)
            self.assertEqual(PhotoPrepareStandardV2().check_lazy_status(config, protected_region_mask=None), ["protected_region_mask"])
            self.assertEqual(PhotoPrepareStandardV2().check_lazy_status(config), [])

    def test_standard_prepare_matches_shared_engine_and_readers(self):
        local = {key: getattr(self.options, key) for key in (
            "resize_method", "invert_mask", "q_left", "q_right", "q_top", "q_bottom", "min_span_px",
            "headroom_ratio", "footroom_ratio", "side_margin_ratio", "bottom_priority",
            "horiz_gravity", "framing_tolerance_percent")}
        kwargs = dict(image=self.image, editor_target=TARGET_FIRERED,
                      crop_config=self.crop_config, **local)
        standard = PhotoPrepareStandardV2().prepare(**kwargs)
        expected = prepare_photo(
            self.image, TARGET_FIRERED,
            replace(self.options, smart_crop=self.crop_config.smart_crop,
                    forced_aspect_ratio=self.crop_config.forced_aspect_ratio,
                    padding_fill=self.crop_config.padding_fill),
        )
        self.assertEqual(len(standard), 4)
        torch.testing.assert_close(standard[0], expected.image, atol=0, rtol=0)
        torch.testing.assert_close(standard[1], expected.native_crop, atol=0, rtol=0)
        torch.testing.assert_close(standard[2], expected.mask, atol=0, rtol=0)
        self.assertEqual(PrepareDimensionsV2().read(standard[3]),
                         (expected.info.plan.target_width, expected.info.plan.target_height))
        self.assertEqual(PrepareCropBoxV2().read(standard[3]),
                         tuple(getattr(expected.info.plan.crop, field)
                               for field in ("x", "y", "width", "height")))
        self.assertEqual(PrepareScaleV2().read(standard[3], "scale_factor"),
                         (expected.info.plan.crop.scale,))
        self.assertEqual(PrepareSummaryV2().read(standard[3], "debug"),
                         (expected.info.plan.debug,))

    def test_metadata_reader_types_and_fields_are_checked(self):
        with self.assertRaisesRegex(TypeError, "prepare_info.*connect"):
            PrepareDimensionsV2().read(self.crop_config)
        info = prepare_photo(self.image, TARGET_DIRECT, self.options).info
        for node in (PrepareScaleV2(), PrepareSummaryV2()):
            with self.assertRaisesRegex(ValueError, "not supported"):
                node.read(info, "__dict__")

    def test_optional_timing_is_inert_by_default(self):
        with patch.object(engine, "perf_counter", side_effect=AssertionError("timer called")):
            result = prepare_photo(self.image, TARGET_DIRECT, self.options)
        self.assertEqual(result.info.timings_ms, ())

    def test_profile_provides_host_stages_without_cuda_sync(self):
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA sync")):
            result = prepare_photo(self.image, TARGET_DIRECT, self.options, profile=True)
        times = dict(result.info.timings_ms)
        self.assertEqual(set(times), {"validate_align", "mask_analysis", "planning", "apply", "total_host"})
        self.assertTrue(all(value >= 0 for value in times.values()))

    def test_empty_protection_is_visible_in_warnings(self):
        with self.assertLogs("PortraitUtils.preparation", level="WARNING"):
            result = prepare_photo(self.image, TARGET_DIRECT, self.options, protected_region_mask=torch.zeros(1, 120, 80))
        self.assertTrue(any("Protected mask is empty" in s for s in result.info.to_dict()["warnings"]))

    def test_configuration_fields_reject_text_nan_and_out_of_range_values(self):
        cases = ((replace(self.crop_config, autocrop_fuzz_tolerance="0.1"), "autocrop_fuzz_tolerance"),
                 (replace(self.crop_config, autocrop_pad_px=True), "autocrop_pad_px"),
                 (replace(self.crop_config, forced_aspect_ratio="oops"), "forced_aspect_ratio"),
                 (EditConfigV2(3.0, .5, 1, "subject"), "inpaint_prompt"),
                 (EditConfigV2("edit", float("nan"), 1, "subject"), "blend_opacity"),
                 (FinishConfigV2(0, 4, False, False, False, False, 42), "initial_upscale"))
        for config, name in cases:
            with self.assertRaisesRegex(ValueError, name):
                validate_config(config)

    def test_producers_do_not_silently_cast_or_clamp(self):
        for changes in ({"subject_mask": 1.0}, {"autocrop_pad_px": -1}, {"smart_crop": "false"}):
            args = asdict(self.crop_config) | changes
            with self.assertRaises(ValueError):
                CropConfigNodeV2().build(autocrop_section="", smart_crop_section="", **args)
        with self.assertRaisesRegex(ValueError, "inpaint_prompt"):
            EditConfigNodeV2().build(.5, 1, 3, "subject", "", "")
        with self.assertRaisesRegex(ValueError, "seed"):
            FinishConfigNodeV2().build(2, 4, False, False, False, False, "42")

    def test_edit_config_three_compact_stitch_prompts(self):
        fields = ("stitch_prompt_1", "stitch_prompt_2", "stitch_prompt_3")
        schema = EditConfigNodeV2.INPUT_TYPES()["required"]
        self.assertEqual(tuple(schema)[-3:], fields)
        for field in fields:
            with self.subTest(field=field):
                kind, settings = schema[field]
                self.assertEqual(kind, "STRING")
                self.assertFalse(settings["multiline"])
                self.assertEqual(settings["default"], "")

        self.assertEqual(EditConfigNodeV2.RETURN_TYPES, ("PORTRAIT_EDIT_CONFIG_V2",))
        self.assertEqual(EditPromptV2.INPUT_TYPES()["required"]["prompt"][0],
                         ["inpaint_prompt", *fields])
        config, = EditConfigNodeV2().build(.4, 1, "edit", "hair", "background", "rocks")
        for field, value in zip(fields, ("hair", "background", "rocks")):
            with self.subTest(field=field):
                self.assertEqual(EditPromptV2().read(config, field), (value,))
                with self.assertRaisesRegex(ValueError, field):
                    validate_config(replace(config, **{field: 3}))
        self.assertEqual(EditPromptV2().read(config, "inpaint_prompt"), ("edit",))
        self.assertEqual(EditConfigNodeV2().build(.4, 1, "edit", "", "", "")[0].stitch_prompt_3, "")

    def test_accidental_forced_aspect_error_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "Forced Aspect Ratio to Auto"):
            prepare_photo(self.image, TARGET_FIRERED, replace(self.options, forced_aspect_ratio="wrong"))

    def test_unsupported_reader_field_is_not_getattr(self):
        with self.assertRaisesRegex(ValueError, "prompt"):
            EditPromptV2().read(EditConfigV2("edit", .5, 1, "subject"), "blend_opacity")
        with self.assertRaisesRegex(ValueError, "scale"):
            FinishScaleV2().read(FinishConfigV2(2, 4, False, False, False, False, 42), "seed")

    def adjustment(self, image, **changes):
        args = dict(image=image, precision="Exact", auto_levels=False, auto_tone=False, auto_color=False,
                    levels_shadow_clip_pct=0, levels_highlight_clip_pct=0, levels_gamma_normalize=False,
                    tone_mode="Per-channel", tone_shadow_clip_pct=0, tone_highlight_clip_pct=0,
                    snap_neutral_midtones=False, flip_horizontal=False)
        return AutoAdjustmentEngine().apply(**(args | changes))[0]

    def test_auto_adjust_noop_uses_original_tensor(self):
        self.assertIs(self.adjustment(self.image), self.image)
        self.assertIs(self.adjustment(self.image, auto_tone=True, strength=0), self.image)

    def test_auto_adjust_dtype_and_alpha_are_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            image = torch.rand(1, 9, 11, 4).to(dtype)
            result = self.adjustment(image, auto_tone=True)
            self.assertEqual(result.dtype, dtype)
            torch.testing.assert_close(result[..., 3], image[..., 3], atol=0, rtol=0)

    def test_exact_statistics_do_not_cast_float64_to_float32(self):
        values = torch.linspace(.4, .4+1e-9, 101, dtype=torch.float64).reshape(1, 101, 1, 1)
        self.assertAlmostEqual(_percentiles_exact(values, .5).item(), .4 + 5e-10, places=15)

    def test_cursor_selection_does_not_commit_until_success(self):
        cursor = TransactionalCursor()
        self.assertEqual(cursor.select("folder", ["a", "b"]), "a")
        self.assertEqual(cursor.positions, {})
        cursor.commit("folder", "a")
        self.assertEqual(cursor.select("folder", ["a", "b"]), "b")
        self.assertEqual(cursor.select("folder", ["a", "b"], repeat=True), "a")
        self.assertEqual(cursor.select("folder", ["a", "b"], reverse=True), "b")
        self.assertEqual(cursor.positions["folder"], "a")

    def test_cursor_state_is_independent_and_bounded(self):
        left, right = TransactionalCursor(2), TransactionalCursor(2)
        for n in range(4): left.commit(n, str(n))
        self.assertEqual(list(left.positions), [2, 3])
        self.assertEqual(right.positions, {})
        left.commit(2, "updated")
        self.assertEqual(list(left.positions), [3, 2])

    def test_resolution_cache_is_bounded_and_contains_no_images(self):
        analysis = MaskAnalysis(None, None, None, "no masks", "not connected", "not connected")
        for width in range(150, 300):
            plan_photo(width, 500, TARGET_KLEIN, analysis, self.options)
        self.assertLessEqual(_build_resolutions.cache_info().currsize, 128)

    def test_export_engine_uses_explicit_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env = ExportEnvironment(directory, directory, True)
            result = PhotoExportEngine(env).save(self.image, filename="engine", prompt={"not": "saved"})
            self.assertEqual(result["ui"]["images"][0]["type"], "output")
            self.assertTrue(Path(directory, "engine.png").is_file())

    def test_pure_outpaint_preserves_gravity_and_even_size_rules(self):
        config = OutpaintConfigV2("Percent", "top left", 20, 10, 0, 0, 0, 0, "expand")
        self.assertEqual(compute_padding(100, 200, config), (0, 0, 20, 20))
        config = replace(config, mode="Pixels", left_px=1, top_px=2, right_px=3, bottom_px=4)
        self.assertEqual(compute_padding(101, 201, config), (1, 2, 4, 5))

    def test_lanczos_never_quantizes_a_high_precision_constant(self):
        for dtype, level in ((torch.float32, .500123), (torch.float64, .500000000123)):
            for channels in (1, 3, 4):
                image = torch.full((1, 8, 6, channels), level, dtype=dtype)
                for width, height in ((17, 23), (3, 4), (5, 21)):
                    result = tensors.resize_image(image, width, height, "lanczos")
                    self.assertEqual(result.dtype, dtype)
                    torch.testing.assert_close(result, torch.full_like(result, level),
                                               atol=3e-7 if dtype == torch.float32 else 1e-15, rtol=0)

    def test_lanczos_preserves_noop_storage_and_one_pixel_dimensions(self):
        source = torch.rand(1, 7, 1, 1, dtype=torch.float64)
        self.assertIs(tensors.resize_image(source, 1, 7, "lanczos"), source)
        result = tensors.resize_image(source, 3, 11, "lanczos")
        self.assertEqual(result.shape, (1, 11, 3, 1))
        torch.testing.assert_close(result[:, :, 0], result[:, :, 1], atol=1e-15, rtol=0)

    def test_lanczos_alpha_does_not_import_hidden_colors(self):
        source = torch.tensor([[[[1., 0, 0, 1], [0, 1, 0, 0]]]], dtype=torch.float64)
        result = tensors.resize_image(source, 9, 5, "lanczos")
        visible = result[..., 3] > 1e-8
        torch.testing.assert_close(result[..., 1][visible], torch.zeros_like(result[..., 1][visible]), atol=0, rtol=0)

    def test_alpha_filter_overshoot_does_not_change_unassociated_color(self):
        source = torch.zeros(1, 9, 15, 4, dtype=torch.float64)
        source[..., :3] = source.new_tensor([.2, .4, .6])
        source[:, :, 4:11, 3] = 1
        for method in ("bicubic", "lanczos"):
            result = tensors.resize_image(source, 45, 21, method)
            visible = result[..., 3] > 1e-6
            expected = source.new_tensor([.2, .4, .6]).expand(visible.sum(), 3)
            torch.testing.assert_close(result[..., :3][visible], expected, atol=1e-13, rtol=0)

    def test_protected_padding_retains_dtype_and_zero_mask_on_canvas(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            image = torch.rand(1, 18, 12, 4).to(dtype)
            mask = torch.ones(1, 18, 12).to(dtype)
            output = prepare_photo(image, TARGET_DIRECT, replace(self.options, forced_aspect_ratio="1:1"), mask, mask)
            self.assertEqual(output.info.plan.crop.padding, (3, 3, 0, 0))
            self.assertEqual(output.image.dtype, dtype)
            self.assertEqual(output.mask.dtype, dtype)
            torch.testing.assert_close(output.image[:, :, 3:15], image, atol=0, rtol=0)
            self.assertFalse(output.mask[:, :, :3].any())
            self.assertFalse(output.mask[:, :, 15:].any())

    def test_lanczos_chunking_does_not_change_pixels(self):
        source = torch.rand(1, 23, 17, 3, dtype=torch.float64)
        expected = tensors.resize_image(source, 29, 15, "lanczos")
        with patch.object(resampling, "_GATHER_ELEMENTS", 17):
            actual = tensors.resize_image(source, 29, 15, "lanczos")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_lanczos_rejects_bad_working_tensor_and_dimensions(self):
        for source, width, height in ((torch.zeros(1, 3, 4, 5, dtype=torch.uint8), 5, 6),
                                      (torch.zeros(1, 3, 4, 5), 0, 6),
                                      (torch.zeros(1, 3, 4, 5), 5.2, 6)):
            with self.assertRaisesRegex(ValueError, "Lanczos"):
                resampling.lanczos_resize(source, width, height)


if __name__ == "__main__":
    unittest.main()
