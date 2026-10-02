"""Internal mask-analysis helpers extracted from the retired smart-crop nodes.

The helper behavior is unchanged; this module does not register ComfyUI nodes.
"""

import numpy as np
import torch


def _to_numpy_mask(m):
    if torch is not None and isinstance(m, torch.Tensor):
        arr = m.detach().cpu().numpy()
    elif isinstance(m, dict) and "mask" in m:
        v = m["mask"]
        arr = v.detach().cpu().numpy() if torch is not None and isinstance(v, torch.Tensor) else np.asarray(v)
    else:
        arr = np.asarray(m)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr.max(axis=0) if arr.shape[0] in (1, 3, 4) else arr.max(axis=-1)
    if arr.ndim != 2:
        h, w = arr.shape[-2], arr.shape[-1]
        arr = arr.reshape(h, w)
    return arr.astype(np.float32, copy=False)


def _normalize01(m):
    mmax = float(m.max()) if m.size else 0.0
    if mmax <= 0:
        return m
    if mmax > 1.5:
        return (m / 255.0).astype(np.float32, copy=False)
    return m


def _border_mass_ratio(m, border_px=8):
    H, W = m.shape
    border_px = int(max(1, min(border_px, min(H, W) // 4)))
    total = float(m.sum()) + 1e-12
    top = m[:border_px, :].sum()
    bottom = m[-border_px:, :].sum()
    left = m[:, :border_px].sum()
    right = m[:, -border_px:].sum()
    corners = (
        m[:border_px, :border_px].sum()
        + m[:border_px, -border_px:].sum()
        + m[-border_px:, :border_px].sum()
        + m[-border_px:, -border_px:].sum()
    )
    border = top + bottom + left + right - corners
    return float(border) / total


def _auto_pick_foreground(m, mode="auto"):
    m = _normalize01(m)
    if mode == "false":
        fg, pick = m, "fg"
    elif mode == "true":
        fg, pick = 1.0 - m, "bg"
    else:
        r_fg = _border_mass_ratio(m)
        r_bg = _border_mass_ratio(1.0 - m)
        if r_bg < r_fg:
            fg, pick = 1.0 - m, "auto->bg"
        else:
            fg, pick = m, "auto->fg"
    return fg, pick


def _quantile_bounds(mass, qL, qR, qT, qB, min_span):
    H, W = mass.shape
    eps = 1e-8
    c = mass.sum(axis=0) + eps
    r = mass.sum(axis=1) + eps
    cc = np.cumsum(c)
    rc = np.cumsum(r)
    ct = float(cc[-1])
    rt = float(rc[-1])

    def qidx(cum, total, q):
        return int(np.clip(np.searchsorted(cum, total * float(q), "left"), 0, len(cum) - 1))

    x0 = qidx(cc, ct, qL)
    x1 = qidx(cc, ct, qR)
    y0 = qidx(rc, rt, qT)
    y1 = qidx(rc, rt, qB)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if (x1 - x0 + 1) < min_span:
        cx = (x0 + x1) // 2
        half = max(1, min_span // 2)
        x0 = max(0, cx - half)
        x1 = min(W - 1, cx + half)
    if (y1 - y0 + 1) < min_span:
        cy = (y0 + y1) // 2
        half = max(1, min_span // 2)
        y0 = max(0, cy - half)
        y1 = min(H - 1, cy + half)
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)
