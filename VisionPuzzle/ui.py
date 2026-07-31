"""Shared premium UI primitives — palette, glass panels, smooth motion."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ── Design tokens (BGR) ─────────────────────────────────────────────────────
# These names stay as plain module globals (not a class/dict lookup) so
# every other file can keep writing `ui.ACCENT`, `ui.BG`, etc. exactly as
# before — set_theme() below reassigns them in place, and since Python
# resolves `ui.X` by attribute lookup at call time (not at import time),
# every existing call site picks up the new theme automatically.

THEMES: dict[str, dict[str, tuple[int, int, int]]] = {
    "dark": {
        "BG": (16, 17, 20), "BG_ELEVATED": (22, 24, 28),
        "STROKE": (70, 76, 82), "STROKE_SOFT": (48, 52, 58),
        "TEXT": (236, 238, 240), "TEXT_MUTED": (160, 166, 172),
        "ACCENT": (196, 178, 92), "ACCENT_HOT": (140, 200, 255),
        "SUCCESS": (120, 200, 150), "DANGER": (90, 90, 220),
    },
    "light": {
        "BG": (238, 240, 242), "BG_ELEVATED": (222, 225, 228),
        "STROKE": (185, 190, 196), "STROKE_SOFT": (204, 208, 212),
        "TEXT": (30, 32, 36), "TEXT_MUTED": (108, 114, 120),
        "ACCENT": (50, 130, 210), "ACCENT_HOT": (190, 120, 40),
        "SUCCESS": (80, 160, 80), "DANGER": (55, 55, 195),
    },
    "neon": {
        "BG": (14, 6, 20), "BG_ELEVATED": (32, 12, 44),
        "STROKE": (150, 50, 190), "STROKE_SOFT": (90, 32, 120),
        "TEXT": (245, 245, 255), "TEXT_MUTED": (175, 145, 205),
        "ACCENT": (255, 0, 200), "ACCENT_HOT": (255, 220, 0),
        "SUCCESS": (140, 255, 60), "DANGER": (60, 40, 255),
    },
    "mono": {
        "BG": (15, 15, 15), "BG_ELEVATED": (36, 36, 36),
        "STROKE": (120, 120, 120), "STROKE_SOFT": (80, 80, 80),
        "TEXT": (245, 245, 245), "TEXT_MUTED": (170, 170, 170),
        "ACCENT": (255, 255, 255), "ACCENT_HOT": (205, 205, 205),
        "SUCCESS": (235, 235, 235), "DANGER": (150, 150, 150),
    },
}

_CURRENT_THEME = "dark"

BG = THEMES["dark"]["BG"]
BG_ELEVATED = THEMES["dark"]["BG_ELEVATED"]
STROKE = THEMES["dark"]["STROKE"]
STROKE_SOFT = THEMES["dark"]["STROKE_SOFT"]
TEXT = THEMES["dark"]["TEXT"]
TEXT_MUTED = THEMES["dark"]["TEXT_MUTED"]
ACCENT = THEMES["dark"]["ACCENT"]
ACCENT_HOT = THEMES["dark"]["ACCENT_HOT"]
SUCCESS = THEMES["dark"]["SUCCESS"]
DANGER = THEMES["dark"]["DANGER"]
SHADOW = (0, 0, 0)


def set_theme(name: str) -> bool:
    """Swap the active palette. Returns False (no-op) for an unknown name."""
    global _CURRENT_THEME, BG, BG_ELEVATED, STROKE, STROKE_SOFT
    global TEXT, TEXT_MUTED, ACCENT, ACCENT_HOT, SUCCESS, DANGER
    theme = THEMES.get(name)
    if theme is None:
        return False
    BG = theme["BG"]
    BG_ELEVATED = theme["BG_ELEVATED"]
    STROKE = theme["STROKE"]
    STROKE_SOFT = theme["STROKE_SOFT"]
    TEXT = theme["TEXT"]
    TEXT_MUTED = theme["TEXT_MUTED"]
    ACCENT = theme["ACCENT"]
    ACCENT_HOT = theme["ACCENT_HOT"]
    SUCCESS = theme["SUCCESS"]
    DANGER = theme["DANGER"]
    _CURRENT_THEME = name
    return True


def get_theme_name() -> str:
    return _CURRENT_THEME


def theme_names() -> list[str]:
    return list(THEMES.keys())


def next_theme_name() -> str:
    names = theme_names()
    idx = names.index(_CURRENT_THEME) if _CURRENT_THEME in names else -1
    return names[(idx + 1) % len(names)]


_VIGNETTE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_point(
    cur: Optional[tuple[float, float]],
    target: tuple[float, float],
    t: float = 0.38,
) -> tuple[float, float]:
    if cur is None:
        return target
    return (lerp(cur[0], target[0], t), lerp(cur[1], target[1], t))


def smooth_toward(current: float, target: float, alpha: float = 0.35) -> float:
    return current * (1.0 - alpha) + target * alpha


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.55,
    color: Optional[tuple[int, int, int]] = None,
    weight: int = 1,
    shadow: bool = True,
) -> None:
    if color is None:
        color = TEXT  # resolved now, not baked in at function-definition time
    if shadow:
        cv2.putText(
            img, text, (org[0] + 1, org[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 1, cv2.LINE_AA,
        )
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA,
    )


def rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color,
    *,
    radius: int = 12,
    thickness: int = -1,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def glass_panel(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    alpha: float = 0.58,
    radius: int = 14,
    border: bool = True,
    accent_top: bool = False,
) -> np.ndarray:
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    # Fast darken without full-frame copy
    dark = np.empty_like(roi)
    dark[:] = BG_ELEVATED
    cv2.addWeighted(dark, alpha, roi, 1.0 - alpha, 0, dst=roi)

    if accent_top and y2 - y1 > 8:
        cv2.line(frame, (x1 + max(1, radius), y1 + 1), (x2 - max(1, radius), y1 + 1), ACCENT, 1, cv2.LINE_AA)
    if border:
        if radius <= 1:
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), STROKE, 1, cv2.LINE_AA)
        else:
            rounded_rect(frame, (x1, y1), (x2, y2), STROKE, radius=radius, thickness=1)
    return frame


def fit_image_to_canvas(img: np.ndarray, target_w: int, target_h: int, *, bg: Optional[tuple[int, int, int]] = None) -> np.ndarray:
    """Resize `img` to fit within (target_w, target_h) preserving its
    aspect ratio, centered on a canvas of exactly that size. Used so an
    uploaded image sits in the same coordinate space as a live camera
    frame — the framing gesture math doesn't need to know the source.
    """
    if bg is None:
        bg = BG
    canvas = np.empty((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = bg
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return canvas
    scale = min(target_w / iw, target_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def vignette(frame: np.ndarray, strength: float = 0.22) -> np.ndarray:
    """Cached radial vignette — cheap after first frame."""
    h, w = frame.shape[:2]
    key = (h, w, int(strength * 100))
    mask = _VIGNETTE_CACHE.get(key)
    if mask is None:
        ys = np.linspace(-1, 1, h, dtype=np.float32)
        xs = np.linspace(-1, 1, w, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv * xv + yv * yv)
        mask = (1.0 - np.clip((r - 0.55) / 0.75, 0.0, 1.0) * strength).astype(np.float32)
        _VIGNETTE_CACHE.clear()
        _VIGNETTE_CACHE[key] = mask
    out = frame.astype(np.float32)
    out *= mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def progress_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    *,
    display: Optional[float] = None,
) -> float:
    target = float(np.clip(value, 0.0, 1.0))
    if display is None:
        display = target
    else:
        display = smooth_toward(display, target, 0.18)

    cv2.rectangle(frame, (x, y), (x + width, y + height), STROKE_SOFT, -1)
    fill = int(width * display)
    if fill > 2:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), ACCENT, -1)
    return display


def chip(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: Optional[tuple[int, int, int]] = None,
    filled: bool = False,
) -> None:
    if color is None:
        color = ACCENT
    pad_x, pad_y = 14, 8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    w, h = tw + pad_x * 2, th + pad_y * 2
    if filled:
        rounded_rect(frame, (x, y), (x + w, y + h), color, radius=h // 2)
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.48, color=BG, weight=1, shadow=False)
    else:
        rounded_rect(frame, (x, y), (x + w, y + h), BG_ELEVATED, radius=h // 2)
        rounded_rect(frame, (x, y), (x + w, y + h), color, radius=h // 2, thickness=1)
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.48, color=color, weight=1)
