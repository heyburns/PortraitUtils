"""
MemoryReporter — diagnostic passthrough node for tracking GPU/CPU memory
between workflow runs. Wire one before and one after a suspected leaky node.

On each execution it logs:
  - VRAM allocated / reserved / free-within-reserved
  - VRAM peak (since last reset or server start)
  - CPU RSS
  - Delta vs. the same label's previous run

The image tensor is passed through unchanged so the node is zero-overhead
in the execution graph.
"""

import gc
import os

import torch

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Persistent across workflow runs within a single ComfyUI session.
# Keyed by label string → {run, allocated, reserved, cpu_rss}
_history: dict = {}

_SEP  = "=" * 58
_HALF = "-" * 58


def _mb(b: int) -> float:
    return b / (1024 * 1024)


def _delta(current: int, previous: int | None) -> str:
    if previous is None:
        return "  (baseline)"
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f"  Δ {sign}{_mb(diff):.1f} MB"


class MemoryReporter:
    """
    Diagnostic passthrough node. Place it immediately before and after a
    suspected leaky node to capture VRAM / RAM deltas per run.

    Fields
    ------
    label       : Unique name for this checkpoint (shows up in the log).
    reset_peak  : If True, resets torch peak-memory counter before measuring
                  (useful at the very start of a workflow to get clean peaks).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label": ("STRING", {"default": "before_seedvr2", "multiline": False}),
            },
            "optional": {
                "reset_peak": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("image", "memory_report")
    FUNCTION      = "report"
    CATEGORY      = "PortraitUtils/Debug"
    OUTPUT_NODE   = True  # always execute even if downstream is muted

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Never cache — run every time.
        return float("nan")

    # ------------------------------------------------------------------
    def report(self, image, label: str = "checkpoint", reset_peak: bool = False):

        # --- 1. force GC so we measure what's actually held, not pending ---
        gc.collect()
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.synchronize()

        # --- 2. optional peak reset ---
        if reset_peak and has_cuda:
            torch.cuda.reset_peak_memory_stats()

        # --- 3. sample GPU ---
        if has_cuda:
            vram_alloc    = torch.cuda.memory_allocated()
            vram_rsvd     = torch.cuda.memory_reserved()
            vram_peak     = torch.cuda.max_memory_allocated()
            vram_free_rsv = vram_rsvd - vram_alloc
        else:
            vram_alloc = vram_rsvd = vram_peak = vram_free_rsv = 0

        # --- 4. sample CPU RAM ---
        if _HAS_PSUTIL:
            proc    = _psutil.Process(os.getpid())
            cpu_rss = proc.memory_info().rss
            sys_avail = _psutil.virtual_memory().available
        else:
            cpu_rss = sys_avail = 0

        # --- 5. run counter + deltas ---
        prev = _history.get(label, {"run": 0, "alloc": None, "rsvd": None, "cpu": None})
        run_n = prev["run"] + 1
        _history[label] = {"run": run_n, "alloc": vram_alloc, "rsvd": vram_rsvd, "cpu": cpu_rss}

        d_alloc = _delta(vram_alloc, prev["alloc"])
        d_rsvd  = _delta(vram_rsvd,  prev["rsvd"])
        d_cpu   = _delta(cpu_rss,    prev["cpu"])

        # --- 6. build report ---
        lines = [
            "",
            _SEP,
            f"  MemoryReporter [{label}]  —  Run #{run_n}",
            _HALF,
            f"  VRAM allocated : {_mb(vram_alloc):9.1f} MB{d_alloc}",
            f"  VRAM reserved  : {_mb(vram_rsvd):9.1f} MB{d_rsvd}",
            f"  VRAM free/rsvd : {_mb(vram_free_rsv):9.1f} MB",
            f"  VRAM peak alloc: {_mb(vram_peak):9.1f} MB",
        ]

        if _HAS_PSUTIL:
            lines += [
                _HALF,
                f"  CPU RSS        : {_mb(cpu_rss):9.1f} MB{d_cpu}",
                f"  Sys free RAM   : {_mb(sys_avail):9.1f} MB",
            ]
        else:
            lines.append("  (install psutil for CPU RAM stats)")

        if not has_cuda:
            lines.append("  (no CUDA device detected)")

        lines += [_SEP, ""]

        report_str = "\n".join(lines)
        print(report_str)

        return (image, report_str)


# -----------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "MemoryReporter": MemoryReporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryReporter": "Memory Reporter [Debug]",
}
