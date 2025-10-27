import math
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


def _pad_rect(x, y, w, h, pad, W, H):
    x2 = max(0, x - pad)
    y2 = max(0, y - pad)
    w2 = min(W - x2, w + 2 * pad)
    h2 = min(H - y2, h + 2 * pad)
    return int(x2), int(y2), int(w2), int(h2)


def _container_for_ar(x, y, w, h, R, W, H):
    """
    Smallest rectangle of aspect R (=w/h) that contains (x,y,w,h), centered,
    then clamped to image. If it can't fit (too large), return None.
    """
    wA = max(w, int(np.ceil(h * R)))
    hA = int(np.ceil(wA / R))
    hB = max(h, int(np.ceil(w / R)))
    wB = int(np.ceil(hB * R))
    cand = []
    if wA <= W and hA <= H:
        cand.append((wA, hA))
    if wB <= W and hB <= H:
        cand.append((wB, hB))
    if not cand:
        return None
    cw, ch = min(cand, key=lambda t: t[0] * t[1])

    cx = x + w / 2.0
    cy = y + h / 2.0
    X = int(np.round(cx - cw / 2.0))
    Y = int(np.round(cy - ch / 2.0))
    X = max(0, min(W - cw, X))
    Y = max(0, min(H - ch, Y))
    return X, Y, int(cw), int(ch)


class MQBBoxMin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": (["auto", "false", "true"], {"default": "auto"}),
                "q_left": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.2, "step": 0.001}),
                "q_right": ("FLOAT", {"default": 0.995, "min": 0.8, "max": 1.0, "step": 0.001}),
                "q_top": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.2, "step": 0.001}),
                "q_bottom": ("FLOAT", {"default": 0.995, "min": 0.8, "max": 1.0, "step": 0.001}),
                "min_span_px": ("INT", {"default": 8, "min": 1, "max": 2048}),
                "tight_crop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "ON: tight subject bbox with smart AR; OFF: full frame (minimal AR crop downstream).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("x", "y", "w", "h", "debug")
    FUNCTION = "run"
    CATEGORY = "utils/debug"

    def run(self, mask, invert_mask, q_left, q_right, q_top, q_bottom, min_span_px, tight_crop=True):
        m = _to_numpy_mask(mask)
        H, W = m.shape

        if not tight_crop:
            dbg = f"tight=False full=({0},{0},{W},{H})"
            return (0, 0, int(W), int(H), dbg)

        # --- original tight bbox ---
        fg, picked = _auto_pick_foreground(m, invert_mask)
        x, y, w, h = _quantile_bounds(fg, q_left, q_right, q_top, q_bottom, int(min_span_px))

        # --- smart AR (minimal-area container), internal only ---
        pad_px = max(8, int(round(0.02 * min(H, W))))
        sx, sy, sw, sh = _pad_rect(x, y, w, h, pad_px, W, H)

        # Baseline: container using current image AR
        R0 = float(W) / max(1.0, float(H))
        base = _container_for_ar(sx, sy, sw, sh, R0, W, H)
        if base is None:
            dbg = f"tight=True pick={picked} mq=({x},{y},{w},{h})"
            return (x, y, w, h, dbg)
        bx, by, bw, bh = base
        base_area = bw * bh

        # Try common ARs in BOTH orientations (includes 9:16)
        ratios = [
            (1, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (9, 16),
            (16, 9),
        ]

        best_rect = base
        best_area = base_area
        best_info = f"AR={W}:{H}"

        for aw, ah in ratios:
            R = float(aw) / float(ah)
            cand = _container_for_ar(sx, sy, sw, sh, R, W, H)
            if cand is None:
                continue
            cx, cy, cw, ch = cand
            area = cw * ch
            if area < best_area:
                best_area = area
                best_rect = cand
                best_info = f"AR={aw}:{ah}"

        rx, ry, rw, rh = best_rect
        dbg = f"tight=True pick={picked} mq=({x},{y},{w},{h}) -> smart({rx},{ry},{rw},{rh}) {best_info}, pad={pad_px}"
        return (int(rx), int(ry), int(rw), int(rh), dbg)


def _get_hw(image):
    t = image["image"] if isinstance(image, dict) and "image" in image else image
    s = tuple(getattr(t, "shape", np.asarray(t).shape))
    if len(s) == 4:
        H, W = (s[1], s[2]) if s[-1] in (1, 3, 4) else (s[2], s[3])
    elif len(s) == 3:
        H, W = (s[0], s[1]) if s[0] not in (1, 3, 4) else (s[1], s[2])
    else:
        H, W = s[-2], s[-1]
    return int(H), int(W)


def _parse_aspects(s):
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        if ":" in p:
            a, b = p.split(":", 1)
        else:
            a, b = p.replace("X", "x").split("x", 1)
        try:
            a = float(a)
            b = float(b)
            if a > 0 and b > 0:
                out.append((a / b, f"{int(a)}:{int(b)}"))
        except Exception:
            pass
    return out or [(2 / 3.0, "2:3")]


def _closest_ratio(target_ratio, candidates):
    tr = float(target_ratio)
    best = None
    for r, tag in candidates:
        d = abs(math.log(r) - math.log(tr))
        if best is None or d < best[0]:
            best = (d, r, tag)
    return best[1], best[2]


def _cover_min_rect(x0, y0, w0, h0, ratio):
    """
    Smallest rectangle of aspect 'ratio' that fully covers the rect (x0,y0,w0,h0).
    Returns floats (xa,ya,wa,ha) without clamping to image.
    """
    if w0 <= 0 or h0 <= 0:
        return x0, y0, w0, h0
    r0 = w0 / max(1.0, h0)
    if r0 >= ratio:
        wa = w0
        ha = w0 / ratio
    else:
        ha = h0
        wa = h0 * ratio
    cx = x0 + w0 / 2.0
    cy = y0 + h0 / 2.0
    return cx - wa / 2.0, cy - ha / 2.0, wa, ha


class FitAspectHeadSafe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT",),
                "y": ("INT",),
                "w": ("INT",),
                "h": ("INT",),
                "aspects_csv": (
                    "STRING",
                    {"default": "2:3,3:4,1:1,9:16,16:9,5:8,8:5"},
                ),
                "match_to": (["image", "mq_box"], {"default": "mq_box"}),
                "headroom_ratio": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
                "footroom_ratio": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01}),
                "side_margin_ratio": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "bottom_priority": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "horiz_gravity": (["center", "left", "right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("w", "h", "x", "y", "aspect_used", "debug")
    FUNCTION = "run"
    CATEGORY = "utils/geometry"

    def run(
        self,
        image,
        x,
        y,
        w,
        h,
        aspects_csv,
        match_to,
        headroom_ratio,
        footroom_ratio,
        side_margin_ratio,
        bottom_priority,
        horiz_gravity,
    ):

        H, W = _get_hw(image)
        img_ratio = W / max(1.0, H)
        mq_ratio = w / max(1.0, h)
        target_ratio = img_ratio if match_to == "image" else mq_ratio

        aspects = _parse_aspects(aspects_csv)
        ratio, tag = _closest_ratio(target_ratio, aspects)

        req_x0 = x
        req_y0 = y
        req_x1 = x + w
        req_y1 = y + h

        guess_h = H
        head_px = headroom_ratio * guess_h
        foot_px = footroom_ratio * guess_h
        side_px = side_margin_ratio * (guess_h * ratio)

        req_x0p = req_x0 - side_px
        req_x1p = req_x1 + side_px
        req_y0p = req_y0 - head_px
        req_y1p = req_y1 + foot_px

        xa, ya, wa, ha = _cover_min_rect(
            req_x0p,
            req_y0p,
            max(1.0, req_x1p - req_x0p),
            max(1.0, req_y1p - req_y0p),
            ratio,
        )

        head_px = headroom_ratio * ha
        foot_px = footroom_ratio * ha
        side_px = side_margin_ratio * wa
        req_x0p = req_x0 - side_px
        req_x1p = req_x1 + side_px
        req_y0p = req_y0 - head_px
        req_y1p = req_y1 + foot_px
        xa, ya, wa, ha = _cover_min_rect(
            req_x0p,
            req_y0p,
            max(1.0, req_x1p - req_x0p),
            max(1.0, req_y1p - req_y0p),
            ratio,
        )

        if horiz_gravity == "left":
            xa = 0.0
        elif horiz_gravity == "right":
            xa = W - wa
        else:
            cx = x + w / 2.0
            xa = cx - wa / 2.0
        xa = max(0.0, min(xa, W - wa))

        y_head = y
        y_feet = y + h
        y_top_des = y_head - head_px
        y_bot_des = y_feet + foot_px
        ya = y_top_des
        ya = max(0.0, min(ya, H - ha))

        def violations(Y):
            top_gap = (Y) - (y_head - head_px)
            bot_gap = (Y + ha) - (y_feet + foot_px)
            loss_top = max(0.0, -top_gap)
            loss_bot = max(0.0, -bot_gap)
            return loss_top, loss_bot, (1.0 - bottom_priority) * loss_top + bottom_priority * loss_bot

        candidates = [
            max(0.0, min(y_top_des, H - ha)),
            max(0.0, min(y_bot_des - ha, H - ha)),
            0.0,
            float(H - ha),
        ]
        best = None
        for Yc in candidates:
            lt, lb, cost = violations(Yc)
            if best is None or cost < best[0]:
                best = (cost, Yc, lt, lb)
        _, ya, lt, lb = best
        ya = max(0.0, min(ya, H - ha))

        X = int(round(xa))
        Y = int(round(ya))
        Wc = int(round(wa))
        Hc = int(round(ha))
        X = max(0, min(X, W - 1))
        Y = max(0, min(Y, H - 1))
        Wc = max(1, min(Wc, W - X))
        Hc = max(1, min(Hc, H - Y))

        dbg = (
            f"match_to={match_to}, chosen={tag}({ratio:.4f}), "
            f"img={W}x{H}, mq=({x},{y},{w},{h}), "
            f"margins(px): head={head_px:.1f}, foot={foot_px:.1f}, side={side_px:.1f}, "
            f"viol: top={lt:.1f}, bot={lb:.1f}; final=({X},{Y},{Wc},{Hc})"
        )
        return (Wc, Hc, X, Y, tag, dbg)


NODE_CLASS_MAPPINGS = {
    "MQBBoxMin": MQBBoxMin,
    "FitAspectHeadSafe": FitAspectHeadSafe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MQBBoxMin": "MQ BBox (MIN)",
    "FitAspectHeadSafe": "Fit Aspect (Head-Safe) - Closest AR + Tight Cover",
}
