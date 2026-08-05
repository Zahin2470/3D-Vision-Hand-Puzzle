"""Shared premium UI primitives — palette, glass panels, smooth motion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # Pillow missing — put_text() falls back to cv2's Hershey font
    _PIL_OK = False


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


def ease_out_cubic(t: float) -> float:
    """Gentle, no-overshoot easing — good for panels/HUD entrances where
    a bouncy spring would feel too playful for a persistent widget."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


# ── Typography (Poppins via PIL, anti-aliased) ─────────────────────────────
# cv2.putText only offers the old Hershey stroke fonts — blocky, no anti-
# aliasing, reads as "programmer UI" rather than a polished product. We
# render text with a real TTF (Poppins, OFL-licensed, bundled under
# assets/fonts/) via Pillow instead, composited onto the same numpy frame.
# Every existing `ui.put_text(...)` call site keeps working unchanged —
# only what's drawn under the hood is nicer.

_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_FONT_FILES = {
    1: "Poppins-Medium.ttf",     # body / default weight
    2: "Poppins-SemiBold.ttf",   # titles, emphasis
    3: "Poppins-ExtraBold.ttf",  # big hero text (win banner, share card)
}
_FONT_CACHE: dict[tuple[int, int], "ImageFont.FreeTypeFont"] = {}
_GLYPH_CACHE: dict[tuple, np.ndarray] = {}
_SCALE_TO_PX = 31  # tuned so existing `scale=` call sites keep their old apparent size


def _get_font(weight: int, px_size: int):
    key = (weight, px_size)
    font = _FONT_CACHE.get(key)
    if font is not None:
        return font
    fname = _FONT_FILES.get(weight, _FONT_FILES[1])
    try:
        font = ImageFont.truetype(str(_FONT_DIR / fname), px_size)
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _render_glyph(text: str, font, color: tuple[int, int, int], shadow: bool) -> np.ndarray:
    """Rasterize text to a small transparent RGBA patch, cached by
    (text, font, color, shadow) — most on-screen text repeats identically
    frame after frame (HUD title, static labels, help lines), so this
    turns a slow PIL draw into a one-time cost. `id(font)` is a stable
    cache key component because _get_font() never evicts loaded fonts."""
    key = (text, id(font), color, shadow)
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    canvas_w = max(1, bbox[2]) + 4
    canvas_h = ascent + descent + 4
    patch = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    rgb = (color[2], color[1], color[0])
    if shadow:
        draw.text((3, 3), text, font=font, fill=(0, 0, 0, 255))
    draw.text((2, 2), text, font=font, fill=(rgb[0], rgb[1], rgb[2], 255))
    arr = np.array(patch)
    if len(_GLYPH_CACHE) > 400:  # simple guard against unbounded growth
        _GLYPH_CACHE.clear()
    _GLYPH_CACHE[key] = arr
    return arr


def _blend_rgba(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Fast vectorized alpha-composite of a cached RGBA text patch onto
    the frame — no PIL/color-space round trip on the hot path."""
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + pw), min(h, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    px0, py0 = x0 - x, y0 - y
    px1, py1 = px0 + (x1 - x0), py0 + (y1 - y0)

    region = img[y0:y1, x0:x1].astype(np.float32)
    src = patch[py0:py1, px0:px1]
    alpha = src[..., 3:4].astype(np.float32) / 255.0
    rgb_bgr = src[..., (2, 1, 0)].astype(np.float32)
    img[y0:y1, x0:x1] = (region * (1.0 - alpha) + rgb_bgr * alpha).astype(np.uint8)


def _put_text_cv2_fallback(img, text, org, scale, color, weight, shadow) -> None:
    """Original Hershey-font path — used only if Pillow or the bundled
    fonts aren't available, so the app never crashes over typography."""
    if shadow:
        cv2.putText(
            img, text, (org[0] + 1, org[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 1, cv2.LINE_AA,
        )
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA,
    )


def text_size(text: str, *, scale: float = 0.55, weight: int = 1) -> tuple[int, int]:
    """Measure rendered (width, height) for a string at the given scale —
    used to size panels/pills to their actual text content. Uses the
    same font resolution as put_text() so measurements stay accurate."""
    if not _PIL_OK or not text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
        return tw, th
    px_size = max(10, int(round(scale * _SCALE_TO_PX)))
    font = _get_font(weight, px_size)
    bbox = font.getbbox(text)
    ascent, descent = font.getmetrics()
    return bbox[2], ascent + descent


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
        color = TEXT
    if not _PIL_OK or not text:
        _put_text_cv2_fallback(img, text, org, scale, color, weight, shadow)
        return

    px_size = max(10, int(round(scale * _SCALE_TO_PX)))
    font = _get_font(weight, px_size)
    ascent, _descent = font.getmetrics()
    patch = _render_glyph(text, font, color, shadow)
    x, y = org
    _blend_rgba(img, patch, x - 2, y - ascent - 2)


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


def _lighten(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in color)  # type: ignore[return-value]


def _vertical_gradient(h: int, w: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(1, h), dtype=np.float32).reshape(-1, 1, 1)
    top_a = np.array(top, dtype=np.float32).reshape(1, 1, 3)
    bot_a = np.array(bottom, dtype=np.float32).reshape(1, 1, 3)
    grad = top_a * (1.0 - t) + bot_a * t
    return np.broadcast_to(grad, (max(1, h), max(1, w), 3)).astype(np.uint8)


def drop_shadow(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    radius: int = 14,
    offset: int = 5,
    alpha: float = 0.30,
) -> None:
    """Cheap soft-ish shadow: a few progressively larger, fainter rounded
    rects offset down-right. No gaussian blur needed to read as "lifted"
    off the background — a handful of layers is enough at UI sizes."""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    for i, (spread, a) in enumerate(((0, alpha), (2, alpha * 0.55), (4, alpha * 0.3))):
        sx1, sy1 = max(0, x1 - spread + offset), max(0, y1 - spread + offset)
        sx2, sy2 = min(w, x2 + spread + offset), min(h, y2 + spread + offset)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        roi = frame[sy1:sy2, sx1:sx2]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, a, roi, 1.0 - a, 0, dst=roi)


def glow_dot(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    *,
    intensity: float = 0.35,
    layers: int = 4,
) -> None:
    """Soft radial glow via a few concentric semi-transparent circles —
    cheaper than a real gaussian bloom pass and reads well at UI scale."""
    cx, cy = center
    h, w = frame.shape[:2]
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(w, cx + radius), min(h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    glow = roi.copy()
    for i in range(layers, 0, -1):
        r = int(radius * i / layers)
        a = intensity * (1.0 - i / (layers + 1))
        cv2.circle(glow, (cx - x0, cy - y0), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, a, roi, 1.0 - a, 0, dst=roi)
        glow[:] = roi


def glass_panel(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    alpha: float = 0.58,
    radius: int = 14,
    border: bool = True,
    accent_top: bool = False,
    gradient: bool = False,
    shadow: bool = False,
) -> np.ndarray:
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    if shadow:
        drop_shadow(frame, (x1, y1), (x2, y2), radius=radius)

    roi = frame[y1:y2, x1:x2]
    if gradient:
        dark = _vertical_gradient(y2 - y1, x2 - x1, _lighten(BG_ELEVATED, 0.10), BG_ELEVATED)
    else:
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
    tw, th = text_size(text, scale=0.48, weight=1)
    w, h = tw + pad_x * 2, th + pad_y * 2
    if filled:
        rounded_rect(frame, (x, y), (x + w, y + h), color, radius=h // 2)
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.48, color=BG, weight=1, shadow=False)
    else:
        rounded_rect(frame, (x, y), (x + w, y + h), BG_ELEVATED, radius=h // 2)
        rounded_rect(frame, (x, y), (x + w, y + h), color, radius=h // 2, thickness=1)
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.48, color=color, weight=1)
