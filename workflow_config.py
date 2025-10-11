import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import folder_paths
import random


@dataclass(frozen=True)
class _Preset:
    name: str
    values: Dict[str, Any]


_BUILT_IN_PRESETS: Iterable[_Preset] = (
    _Preset(
        "Default",
        {
            "initial_upscale": 1.0,
            "final_upscale": 4.0,
            "tight_crop": False,
            "stitch_bypass_mask": False,
            "stitch_opacity": 1.0,
            "crop_fuzz_percent": 5.0,
            "crop_pad_px": 0,
            "seed": 0,
            "job_notes": "",
        },
    ),
)


def _preset_directory() -> Path:
    input_root = Path(folder_paths.get_input_directory()).resolve()
    target = input_root.parent / "user" / "default" / "workflows" / "workflow_config_presets"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _built_in_options() -> Dict[str, Dict[str, Any]]:
    return {preset.name: dict(preset.values) for preset in _BUILT_IN_PRESETS}


def _list_saved_presets() -> Iterable[str]:
    for path in _preset_directory().glob("*.json"):
        if path.is_file():
            yield path.stem


def _load_saved_payload(file_path: Optional[Path]) -> Dict[str, Any]:
    if not file_path:
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}

    if isinstance(data, dict) and "blend_factor" in data and "stitch_opacity" not in data:
        data["stitch_opacity"] = data["blend_factor"]
    return data if isinstance(data, dict) else {}


def _persist_snapshot(target_name: Optional[str], payload: Dict[str, Any]) -> None:
    filename = (target_name or "").strip()
    if not filename:
        return
    if not filename.lower().endswith(".json"):
        filename = f"{filename}.json"
    destination = _preset_directory() / filename
    try:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}")


SEED_MIN = 0
SEED_MAX = 0xFFFFFFFF


def _clamp_seed(value: Any) -> int:
    try:
        candidate = int(value)
    except Exception:
        return SEED_MIN
    return max(SEED_MIN, min(SEED_MAX, candidate))

_initial_state = random.getstate()
random.seed(datetime.now().timestamp())
_seed_rng_state = random.getstate()
random.setstate(_initial_state)

LOG_PREFIX = "WorkflowConfig"


def _new_random_seed() -> int:
    global _seed_rng_state  # pylint: disable=global-statement
    prev_state = random.getstate()
    random.setstate(_seed_rng_state)
    seed_value = random.randint(SEED_MIN, SEED_MAX)
    _seed_rng_state = random.getstate()
    random.setstate(prev_state)
    return seed_value


def _update_workflow_widgets(
    extra_pnginfo: Dict[str, Any],
    node_id: Any,
    original_seed: Any,
    new_seed: int,
) -> None:
    workflow = extra_pnginfo.get("workflow", {}) if isinstance(extra_pnginfo, dict) else {}
    nodes = workflow.get("nodes", []) if isinstance(workflow.get("nodes", []), list) else []
    workflow_node = next((node for node in nodes if str(node.get("id")) == str(node_id)), None)
    if not workflow_node or "widgets_values" not in workflow_node:
        _log(LOG_PREFIX, "Unable to store seed in workflow metadata (node not found).")
    else:
        for index, value in enumerate(workflow_node["widgets_values"]):
            if value == original_seed:
                workflow_node["widgets_values"][index] = new_seed


def _update_prompt_inputs(prompt_nodes: Dict[str, Any], node_id: Any, new_seed: int) -> None:
    prompt_node = prompt_nodes.get(str(node_id)) if isinstance(prompt_nodes, dict) else None
    if not prompt_node or "inputs" not in prompt_node or "seed" not in prompt_node["inputs"]:
        _log(LOG_PREFIX, "Unable to store seed in prompt metadata (node not found).")
    else:
        prompt_node["inputs"]["seed"] = new_seed


def _resolve_seed(
    seed: int,
    prompt: Optional[Dict[str, Any]],
    extra_pnginfo: Optional[Dict[str, Any]],
    unique_id: Optional[Any],
) -> int:
    if seed in (-1, -2, -3):
        _log(LOG_PREFIX, f'Received special seed "{seed}". Generating a new random seed server-side.')
        if seed in (-2, -3):
            action = "increment" if seed == -2 else "decrement"
            _log(LOG_PREFIX, f"Cannot {action} without prior seed; using random seed.")

        original_seed = seed
        seed = _new_random_seed()
        _log(LOG_PREFIX, f"Generated random seed {seed} and will attempt to persist it.")

        if unique_id is None:
            _log(LOG_PREFIX, "Node unique_id was not provided; cannot persist seed.")
        else:
            if extra_pnginfo is None:
                _log(LOG_PREFIX, "Workflow metadata not provided; cannot persist seed.")
            else:
                _update_workflow_widgets(extra_pnginfo, unique_id, original_seed, seed)

            if prompt is None:
                _log(LOG_PREFIX, "Prompt metadata not provided; cannot persist seed.")
            else:
                _update_prompt_inputs(prompt, unique_id, seed)

    return seed


class WorkflowConfig:
    """
    Presents workflow-level controls (upscale amounts, crop policy, stitch options, seed control)
    and emits both socket values and a serialized JSON bundle for downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        built_in = sorted(_built_in_options().keys())
        saved_files = ["<none>"] + sorted(set(_list_saved_presets()))
        return {
            "required": {
                "template_name": (built_in, {"default": built_in[0]}),
                "saved_preset": (saved_files, {"default": saved_files[0]}),
                "load_saved_preset": ("BOOLEAN", {"default": False}),
                "initial_upscale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "final_upscale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "tight_crop": ("BOOLEAN", {"default": False}),
                "stitch_bypass_mask": ("BOOLEAN", {"default": False}),
                "stitch_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_fuzz_percent": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "crop_pad_px": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "seed": ("INT", {"default": 0, "min": SEED_MIN, "max": SEED_MAX}),
                "job_notes": ("STRING", {"default": "", "multiline": True, "placeholder": "Notes"}),
            },
            "optional": {
                "save_preset_as": ("STRING", {"default": "", "multiline": False, "placeholder": "preset name (e.g., portrait)"}),
                "write_preset": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "BOOLEAN",
        "BOOLEAN",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "initial_upscale",
        "final_upscale",
        "tight_crop",
        "stitch_bypass_mask",
        "stitch_opacity",
        "crop_fuzz_percent",
        "crop_pad_px",
        "seed",
        "job_notes",
        "config_bundle",
    )
    FUNCTION = "configure"
    CATEGORY = "config"

    @classmethod
    def IS_CHANGED(
        cls,
        template_name,
        saved_preset,
        load_saved_preset,
        initial_upscale,
        final_upscale,
        tight_crop,
        stitch_bypass_mask,
        stitch_opacity,
        crop_fuzz_percent,
        crop_pad_px,
        seed,
        job_notes,
        save_preset_as=None,
        write_preset=False,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        if seed in (-1, -2, -3):
            return _new_random_seed()
        return seed

    def configure(
        self,
        template_name,
        saved_preset,
        load_saved_preset,
        initial_upscale,
        final_upscale,
        tight_crop,
        stitch_bypass_mask,
        stitch_opacity,
        crop_fuzz_percent,
        crop_pad_px,
        seed=0,
        job_notes="",
        save_preset_as=None,
        write_preset=False,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        templates = _built_in_options()
        template_payload = dict(templates.get(template_name, templates["Default"]))

        preset_payload: Dict[str, Any] = {}
        if load_saved_preset and saved_preset and saved_preset != "<none>":
            preset_path = _preset_directory() / f"{saved_preset}.json"
            preset_payload = _load_saved_payload(preset_path)

        merged = {**template_payload, **preset_payload}

        values: Dict[str, Any] = {
            "initial_upscale": float(initial_upscale if initial_upscale is not None else merged["initial_upscale"]),
            "final_upscale": float(final_upscale if final_upscale is not None else merged["final_upscale"]),
            "tight_crop": bool(tight_crop if tight_crop is not None else merged["tight_crop"]),
            "stitch_bypass_mask": bool(stitch_bypass_mask if stitch_bypass_mask is not None else merged["stitch_bypass_mask"]),
            "stitch_opacity": float(stitch_opacity if stitch_opacity is not None else merged["stitch_opacity"]),
            "crop_fuzz_percent": float(crop_fuzz_percent if crop_fuzz_percent is not None else merged["crop_fuzz_percent"]),
            "crop_pad_px": int(crop_pad_px if crop_pad_px is not None else merged["crop_pad_px"]),
            "seed": _clamp_seed(seed if seed is not None else merged.get("seed", 0)),
            "job_notes": str(job_notes if job_notes is not None else merged.get("job_notes", "")),
        }

        seed_value = _clamp_seed(_resolve_seed(values["seed"], prompt, extra_pnginfo, unique_id))
        values["seed"] = seed_value

        if write_preset and save_preset_as:
            to_store = dict(values)
            _persist_snapshot(save_preset_as, to_store)

        bundle = json.dumps(values, ensure_ascii=False)

        return (
            values["initial_upscale"],
            values["final_upscale"],
            values["tight_crop"],
            values["stitch_bypass_mask"],
            values["stitch_opacity"],
            values["crop_fuzz_percent"],
            values["crop_pad_px"],
            seed_value,
            values["job_notes"],
            bundle,
        )


NODE_CLASS_MAPPINGS = {
    "WorkflowConfig": WorkflowConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowConfig": "Workflow Config (central)",
}
