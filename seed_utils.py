"""Internal seed utilities extracted from the retired Universal Project Config.

Seed resolution and saved-metadata behavior are unchanged.
"""

import random
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional


SEED_MIN = 0
SEED_MAX = 0xffffffff  # 2**32 - 1
LOG_PREFIX = "PortraitConfig"


def _log(prefix: str, message: str) -> None:
    logging.getLogger("PortraitUtils.seed").info("[%s] %s", prefix, message)


def _clamp_seed(value: Any) -> int:
    try:
        candidate = int(value)
    except Exception:
        return SEED_MIN
    return max(SEED_MIN, min(SEED_MAX, candidate))


_seed_rng = random.Random(datetime.now().timestamp())
_seed_rng_lock = threading.Lock()


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
        # Preserve the existing first-integer-match metadata behavior.
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
