import json
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ComfyUI hack to allow wildcard connections
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

ANY_TYPE = AnyType("*")

SEED_MIN = 0
SEED_MAX = 0xffffffff  # 2**32 - 1

def _log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}")

def _clamp_seed(value: Any) -> int:
    try:
        candidate = int(value)
    except Exception:
        return SEED_MIN
    return max(SEED_MIN, min(SEED_MAX, candidate))

# HI-6: Use a dedicated Random instance + Lock instead of save/restore of the
# global Python RNG state.  ComfyUI runs node execution in threads, so the
# original get/setstate pattern had a race condition.
_seed_rng = random.Random(datetime.now().timestamp())
_seed_rng_lock = threading.Lock()

LOG_PREFIX = "UniversalConfig"

def _new_random_seed() -> int:
    with _seed_rng_lock:
        return _seed_rng.randint(SEED_MIN, SEED_MAX)

def _update_workflow_widgets(extra_pnginfo: Dict[str, Any], node_id: Any, original_seed: Any, new_seed: int) -> None:
    workflow = extra_pnginfo.get("workflow", {}) if isinstance(extra_pnginfo, dict) else {}
    nodes = workflow.get("nodes", []) if isinstance(workflow.get("nodes", []), list) else []
    workflow_node = next((node for node in nodes if str(node.get("id")) == str(node_id)), None)
    if not workflow_node or "widgets_values" not in workflow_node:
        _log(LOG_PREFIX, "Unable to store seed in workflow metadata (node not found).")
    else:
        # HI-7: Guard against matching non-seed widgets that happen to share a
        # sentinel value (-1/-2/-3) by also requiring the value to be an int and
        # updating only the first match, not all occurrences.
        for index, value in enumerate(workflow_node["widgets_values"]):
            if isinstance(value, int) and value == original_seed:
                workflow_node["widgets_values"][index] = new_seed
                break

def _update_prompt_inputs(prompt_nodes: Dict[str, Any], node_id: Any, new_seed: int) -> None:
    prompt_node = prompt_nodes.get(str(node_id)) if isinstance(prompt_nodes, dict) else None
    if not prompt_node or "inputs" not in prompt_node or "seed" not in prompt_node["inputs"]:
        _log(LOG_PREFIX, "Unable to store seed in prompt metadata (node not found).")
    else:
        prompt_node["inputs"]["seed"] = new_seed

def _resolve_seed(seed: int, prompt: Optional[Dict[str, Any]], extra_pnginfo: Optional[Dict[str, Any]], unique_id: Optional[Any]) -> int:
    if seed in (-1, -2, -3):
        original_seed = seed
        seed = _new_random_seed()
        if unique_id is not None:
            if extra_pnginfo is not None:
                _update_workflow_widgets(extra_pnginfo, unique_id, original_seed, seed)
            if prompt is not None:
                _update_prompt_inputs(prompt, unique_id, seed)
    return seed


SCHEMA_FILE = Path(__file__).parent / "config_schema.json"

# ME-8: Cache the parsed schema so we don't re-read and re-parse the JSON file
# on every INPUT_TYPES call (which ComfyUI invokes on every UI refresh).
_schema_cache: Optional[Dict[str, Any]] = None


def _load_schema() -> Dict[str, Any]:
    """Return the parsed config_schema.json, reading from disk only once."""
    global _schema_cache
    if _schema_cache is None:
        if SCHEMA_FILE.exists():
            try:
                with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
                    _schema_cache = json.load(f)
            except Exception as e:
                _log(LOG_PREFIX, f"Error reading config_schema.json: {e}")
                _schema_cache = {}
        else:
            _schema_cache = {}
    return _schema_cache


_schema = _load_schema()
_dynamic_keys = [k for k, v in _schema.items() if isinstance(v, list) and len(v) == 2]
_dynamic_types = [v[0] for v in _schema.values() if isinstance(v, list) and len(v) == 2]

class UniversalProjectConfig:
    """
    Schema-driven configuration node. It builds native UI widgets from a JSON file, 
    so users get a perfect visual experience while keeping the Python code universally applicable.
    Crucially, it also dynamically generates exact output pins for everything in the schema!
    """
    @classmethod
    def INPUT_TYPES(cls):
        required_inputs = {}

        schema = _load_schema()
        for k, v in schema.items():
            if isinstance(v, list) and len(v) == 2:
                required_inputs[k] = (v[0], v[1])

        required_inputs["seed"] = ("INT", {"default": 0, "min": SEED_MIN, "max": SEED_MAX})

        return {
            "required": required_inputs,
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = tuple(_dynamic_types) + ("STRING", "INT")
    RETURN_NAMES = tuple(_dynamic_keys) + ("config_bundle", "seed")
    FUNCTION = "configure"
    CATEGORY = "PortraitUtils/Config"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed = kwargs.get("seed", 0)
        if seed in (-1, -2, -3):
            return _new_random_seed()

        safe_kwargs = {k: v for k, v in kwargs.items() if k not in ("prompt", "extra_pnginfo", "unique_id", "seed")}
        return f"{seed}_{safe_kwargs}"

    def configure(self, **kwargs):
        seed = kwargs.pop("seed", 0)
        prompt = kwargs.pop("prompt", None)
        extra_pnginfo = kwargs.pop("extra_pnginfo", None)
        unique_id = kwargs.pop("unique_id", None)

        values = dict(kwargs)
        seed_value = _clamp_seed(_resolve_seed(seed, prompt, extra_pnginfo, unique_id))
        values["seed"] = seed_value

        bundle = json.dumps(values, ensure_ascii=False)

        # Output everything in exact schema order to match dynamically generated return types
        outputs = []
        for key in _dynamic_keys:
            outputs.append(values.get(key))

        return tuple(outputs) + (bundle, seed_value)


class ExtractConfigValue:
    """
    Extracts a value dynamically from a serialized JSON config bundle.
    Provides ANY_TYPE so you don't need to hardcode pin outputs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        schema = _load_schema()
        keys = ["<missing_or_custom>"] + list(schema.keys())

        return {
            "required": {
                "config_bundle": ("STRING", {"forceInput": True}),
                "key": (keys, {"default": keys[1] if len(keys) > 1 else keys[0]}),
                "default_fallback": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "extract"
    CATEGORY = "PortraitUtils/Config"

    def extract(self, **kwargs):
        config_bundle = kwargs.get("config_bundle", "")
        key = kwargs.get("key", "")
        default_fallback = kwargs.get("default_fallback", "")

        if config_bundle and isinstance(config_bundle, str):
            try:
                data = json.loads(config_bundle)
                if key in data:
                    return (data[key],)
            except Exception as e:
                _log("ExtractConfigValue", f"JSON parse error: {e}")

        schema = _load_schema()
        if key in schema and isinstance(schema[key], list) and len(schema[key]) > 1:
            if "default" in schema[key][1]:
                return (schema[key][1]["default"],)

        val = default_fallback
        if not val or val.strip() == "":
            return (0.0,)

        if str(val).lower() == "true": return (True,)
        if str(val).lower() == "false": return (False,)
        try:
            return (float(val) if '.' in str(val) else int(val),)
        except Exception:
            return (val,)


NODE_CLASS_MAPPINGS = {
    "UniversalProjectConfig": UniversalProjectConfig,
    "ExtractConfigValue": ExtractConfigValue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalProjectConfig": "Universal Project Config",
    "ExtractConfigValue": "Extract Config Value",
}
