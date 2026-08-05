"""Shared premium UI primitives — ultra-advanced cyberpunk/scifi glass panels, high-contrast HUD cards, key-caps, and clean focus note boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  
    _PIL_OK = False


# ── Design Tokens (BGR Format for OpenCV) ──────────────────────────────────
THEMES: dict[str, dict[str, tuple[int, int, int]]] = {
    "cyberpunk": {
        "BG": (18, 12, 14),             # Deep midnight obsidian
        "BG_ELEVATED": (34, 20, 28),    # Slate neon violet
        "STROKE": (255, 0, 180),        # Electric Magenta
        "STROKE_SOFT": (160, 40, 100),  # Muted Magenta
        "TEXT": (255, 255, 255),       # Crisp Pure White
        "TEXT_MUTED": (210, 175, 215), # Soft Lavender
        "ACCENT": (255, 235, 0),        # High-energy Neon Cyan
        "ACCENT_HOT": (255, 0, 140),    # Vivid Neon Pink
        "SUCCESS": (120, 255, 80),      # Bright Lime Green
        "DANGER": (80, 70, 255),        # Electric Crimson
        "WARNING": (0, 190, 255),       # Vibrant Amber
    },
    "dark": {
        "BG": (16, 18, 22),
        "BG_ELEVATED": (28, 32, 40),
        "STROKE": (90, 110, 135),
        "STROKE_SOFT": (45, 52, 65),
        "TEXT": (250, 252, 255),
        "TEXT_MUTED": (160, 172, 190),
        "ACCENT": (255, 210, 0),
        "ACCENT_HOT": (255, 100, 180),
        "SUCCESS": (90, 240, 130),
        "DANGER": (80, 80, 255),
        "WARNING": (30, 200, 255),
    },
    "emerald": {
        "BG": (12, 20, 15),
        "BG_ELEVATED": (20, 36, 28),
        "STROKE": (60, 180, 110),
        "STROKE_SOFT": (30, 90, 55),
        "TEXT": (245, 255, 250),
        "TEXT_MUTED": (150, 200, 175),
        "ACCENT": (120, 255, 160),
        "ACCENT_HOT": (80, 230, 255),
        "SUCCESS": (100, 255, 140),
        "DANGER": (60, 80, 240),
        "WARNING": (40, 210, 255),
    },
    "synthwave": {
        "BG": (28, 10, 22),
        "BG_ELEVATED": (52, 18, 42),
        "STROKE": (255, 110, 60),
        "STROKE_SOFT": (150, 50, 40),
        "TEXT": (255, 245, 252),
        "TEXT_MUTED": (230, 165, 205),
        "ACCENT": (255, 140, 0),
        "ACCENT_HOT": (255, 0, 200),
        "SUCCESS": (80, 240, 190),
        "DANGER": (80, 50, 255),
        "WARNING": (20, 200, 255),
    },
    "quantum": {
        "BG": (24, 12, 8),
        "BG_ELEVATED": (46, 26, 16),
        "STROKE": (255, 160, 80),
        "STROKE_SOFT": (140, 90, 45),
        "TEXT": (250, 252, 255),
        "TEXT_MUTED": (180, 195, 225),
        "ACCENT": (255, 215, 0),
        "ACCENT_HOT": (0, 230, 255),
        "SUCCESS": (120, 255, 120),
        "DANGER": (80, 80, 255),
        "WARNING": (0, 180, 255),
    },
    "mono": {
        "BG": (14, 14, 14),
        "BG_ELEVATED": (32, 32, 32),
        "STROKE": (120, 120, 120),
        "STROKE_SOFT": (65, 65, 65),
        "TEXT": (255, 255, 255),
        "TEXT_MUTED": (175, 175, 175),
        "ACCENT": (255, 255, 255),
        "ACCENT_HOT": (210, 210, 210),
        "SUCCESS": (240, 240, 240),
        "DANGER": (140, 140, 140),
        "WARNING": (180, 180, 180),
    },
}

_CURRENT_THEME = "cyberpunk"

BG = THEMES["cyberpunk"]["BG"]
BG_ELEVATED = THEMES["cyberpunk"]["BG_ELEVATED"]
STROKE = THEMES["cyberpunk"]["STROKE"]
STROKE_SOFT = THEMES["cyberpunk"]["STROKE_SOFT"]
TEXT = THEMES["cyberpunk"]["TEXT"]
TEXT_MUTED = THEMES["cyberpunk"]["TEXT_MUTED"]
ACCENT = THEMES["cyberpunk"]["ACCENT"]
ACCENT_HOT = THEMES["cyberpunk"]["ACCENT_HOT"]
SUCCESS = THEMES["cyberpunk"]["SUCCESS"]
DANGER = THEMES["cyberpunk"]["DANGER"]
WARNING = THEMES["cyberpunk"]["WARNING"]
SHADOW = (0, 0, 0)


def set_theme(name: str) -> bool:
    global _CURRENT_THEME, BG, BG_ELEVATED, STROKE, STROKE_SOFT
    global TEXT, TEXT_MUTED, ACCENT, ACCENT_HOT, SUCCESS, DANGER, WARNING
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
    WARNING = theme.get("WARNING", (0, 190, 255))
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
    t: float = 0.4,
) -> tuple[float, float]:
    if cur is None:
        return target
    return (lerp(cur[0], target[0], t), lerp(cur[1], target[1], t))


def smooth_toward(current: float, target: float, alpha: float = 0.4) -> float:
    return current * (1.0 - alpha) + target * alpha


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


# ── Typography (Poppins via PIL, Anti-Aliased) ────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_FONT_FILES = {
    1: "Poppins-Medium.ttf",
    2: "Poppins-SemiBold.ttf",
    3: "Poppins-ExtraBold.ttf",
}
_FONT_CACHE: dict[tuple[int, int], "ImageFont.FreeTypeFont"] = {}
_GLYPH_CACHE: dict[tuple, np.ndarray] = {}
_SCALE_TO_PX = 32


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
    key = (text, id(font), color, shadow)
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    canvas_w = max(1, bbox[2]) + 6
    canvas_h = ascent + descent + 6
    patch = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    rgb = (color[2], color[1], color[0])
    if shadow:
        draw.text((3, 3), text, font=font, fill=(0, 0, 0, 210))
    draw.text((2, 2), text, font=font, fill=(rgb[0], rgb[1], rgb[2], 255))
    arr = np.array(patch)
    if len(_GLYPH_CACHE) > 500:
        _GLYPH_CACHE.clear()
    _GLYPH_CACHE[key] = arr
    return arr


def _blend_rgba(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
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
    if not text:
        return 0, 0
    if not _PIL_OK:
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
    # Completely suppress drawing any text containing "upload an image"
    if not text or "upload an image" in text.lower():
        return

    if color is None:
        color = TEXT
    if not _PIL_OK:
        _put_text_cv2_fallback(img, text, org, scale, color, weight, shadow)
        return

    px_size = max(10, int(round(scale * _SCALE_TO_PX)))
    font = _get_font(weight, px_size)
    ascent, _descent = font.getmetrics()
    patch = _render_glyph(text, font, color, shadow)
    x, y = org
    _blend_rgba(img, patch, x - 2, y - ascent - 2)


# ── Shape & Vector Masking Helpers ──────────────────────────────────────────
def _get_rounded_mask(h: int, w: int, radius: int) -> np.ndarray:
    """Generates an anti-aliased 8-bit rounded rectangle mask (0 to 255)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    r = max(1, min(radius, w // 2, h // 2))
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (w - r, r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (r, h - r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (w - r, h - r), r, 255, -1, cv2.LINE_AA)
    return mask


def rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
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


def glow_dot(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    *,
    intensity: float = 0.6,
    layers: int = 5,
) -> None:
    cx, cy = center
    h, w = frame.shape[:2]
    x0, y0 = max(0, cx - radius * 2), max(0, cy - radius * 2)
    x1, y1 = min(w, cx + radius * 2), min(h, cy + radius * 2)
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


def draw_cyber_brackets(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    length: int = 14,
    thickness: int = 2,
) -> None:
    """Draws futuristic neon targeting brackets at panel corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


# ── Premium UI Containers & Focus Boxes ─────────────────────────────────────
def glass_panel(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    alpha: float = 0.94,  # High contrast dark opacity
    radius: int = 14,
    border: bool = True,
    accent_top: bool = True,
    gradient: bool = True,
    shadow: bool = False,
    cyber_brackets: bool = True,
    border_color: Optional[tuple[int, int, int]] = None,
    accent_color: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Renders a high-contrast dark cyberpunk glass panel using ALPHA MASKING
    to eliminate sharp square background bleed outside the rounded corners.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return frame

    if border_color is None:
        border_color = STROKE_SOFT
    if accent_color is None:
        accent_color = ACCENT

    # 1. Create Dark Background Gradient Fill
    roi = frame[y1:y2, x1:x2]
    if gradient:
        panel_bg = _vertical_gradient(ph, pw, _lighten(BG_ELEVATED, 0.15), BG)
    else:
        panel_bg = np.empty_like(roi)
        panel_bg[:] = BG_ELEVATED

    # 2. Perfect Alpha Masking (Stops square corners from bleeding through outside the round arc)
    r = max(1, min(radius, pw // 2, ph // 2))
    mask = _get_rounded_mask(ph, pw, r)
    alpha_mask = (mask.astype(np.float32) / 255.0 * alpha)[..., None]

    blended = (panel_bg.astype(np.float32) * alpha_mask + roi.astype(np.float32) * (1.0 - alpha_mask)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended

    # 3. Glowing Accent Header Rule (Kept within rounded radius)
    if accent_top and (ph > 10) and (pw > r * 2):
        cv2.line(frame, (x1 + r, y1 + 1), (x2 - r, y1 + 1), accent_color, 2, cv2.LINE_AA)

    # 4. Clean Anti-Aliased Border
    if border:
        rounded_rect(frame, (x1, y1), (x2, y2), border_color, radius=r, thickness=1)

    if cyber_brackets and (pw > 60) and (ph > 60):
        draw_cyber_brackets(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), accent_color, length=10, thickness=1)

    return frame


def focus_note_box(
    frame: np.ndarray,
    text_lines: Sequence[str],
    x: int,
    y: int,
    *,
    title: Optional[str] = "CONTROLS & GUIDE",
    badge: Optional[str] = "FOCUS NOTE",
    accent_color: Optional[tuple[int, int, int]] = None,
    scale: float = 0.52,
    weight: int = 2,
    alpha: float = 0.94,
) -> tuple[int, int]:
    """
    Renders an aesthetically dashing, ultra-clear focus note box with rounded mask bounds,
    a vertical accent bar, and sharp text hierarchy. Filters out unwanted duplicate text lines.
    """
    if accent_color is None:
        accent_color = ACCENT

    pad_x = 22
    pad_y = 18
    accent_bar_w = 5
    line_spacing = max(6, int(14 * scale))

    # Strict filter: drop any line that contains "upload an image"
    filtered_lines = [
        line for line in text_lines 
        if "upload an image" not in line.lower()
    ]

    # Calculate exact content dimensions
    max_w = 0
    total_h = 0

    if badge or title:
        header_h = 0
        if badge:
            bw, bh = text_size(badge, scale=scale * 0.85, weight=3)
            header_h = bh + 8
            max_w = max(max_w, bw + 16)
        if title:
            tw, th = text_size(title, scale=scale * 1.0, weight=3)
            max_w = max(max_w, tw + (bw + 28 if badge else 0))
            header_h = max(header_h, th)
        total_h += header_h + 14

    for line in filtered_lines:
        tw, th = text_size(line, scale=scale, weight=1)
        max_w = max(max_w, tw)
        total_h += th + line_spacing

    box_w = max_w + (pad_x * 2) + accent_bar_w + 10
    box_h = total_h + (pad_y * 2)

    # Prevent box overflow beyond frame boundaries
    fh, fw = frame.shape[:2]
    x = max(12, min(x, fw - box_w - 12))
    y = max(12, min(y, fh - box_h - 12))

    # 1. Masked Background Panel
    glass_panel(
        frame,
        (x, y),
        (x + box_w, y + box_h),
        alpha=alpha,
        radius=14,
        border=True,
        accent_top=False,
        cyber_brackets=True,
        border_color=STROKE_SOFT,
        accent_color=accent_color,
    )

    # 2. Left Glowing Accent Bar
    bar_x1 = x + 4
    bar_y1 = y + 12
    bar_y2 = y + box_h - 12
    rounded_rect(frame, (bar_x1, bar_y1), (bar_x1 + accent_bar_w, bar_y2), accent_color, radius=2, thickness=-1)

    # 3. Render Header (Badge + Title)
    cur_y = y + pad_y
    content_x = x + pad_x + accent_bar_w

    if badge or title:
        if badge:
            bw, bh = text_size(badge, scale=scale * 0.85, weight=3)
            badge_pad_h = 4
            badge_pad_w = 8
            badge_box_w = bw + (badge_pad_w * 2)
            badge_box_h = bh + (badge_pad_h * 2)
            
            rounded_rect(
                frame,
                (content_x, cur_y),
                (content_x + badge_box_w, cur_y + badge_box_h),
                accent_color,
                radius=4,
                thickness=-1,
            )
            put_text(
                frame,
                badge.upper(),
                (content_x + badge_pad_w, cur_y + badge_pad_h + bh),
                scale=scale * 0.85,
                color=BG,
                weight=3,
                shadow=False,
            )
            
            if title:
                _, th = text_size(title, scale=scale * 1.0, weight=3)
                put_text(
                    frame,
                    title,
                    (content_x + badge_box_w + 12, cur_y + badge_pad_h + th),
                    scale=scale * 1.0,
                    color=TEXT,
                    weight=3,
                    shadow=True,
                )
            cur_y += badge_box_h + 14
        elif title:
            _, th = text_size(title, scale=scale * 1.1, weight=3)
            put_text(
                frame,
                title,
                (content_x, cur_y + th),
                scale=scale * 1.1,
                color=accent_color,
                weight=3,
                shadow=True,
            )
            cur_y += th + 14

        # Header Separator Line
        cv2.line(frame, (content_x, cur_y - 6), (x + box_w - pad_x, cur_y - 6), STROKE_SOFT, 1, cv2.LINE_AA)

    # 4. Render Content Text
    for line in filtered_lines:
        _, th = text_size(line, scale=scale, weight=1)
        
        if " · " in line:
            parts = line.split(" · ")
            cur_part_x = content_x
            for idx, part in enumerate(parts):
                p_color = ACCENT_HOT if idx == 0 else TEXT_MUTED
                put_text(frame, part, (cur_part_x, cur_y + th), scale=scale, color=p_color, weight=2 if idx == 0 else 1)
                pw, _ = text_size(part, scale=scale, weight=2 if idx == 0 else 1)
                cur_part_x += pw
                if idx < len(parts) - 1:
                    sep = "  ·  "
                    put_text(frame, sep, (cur_part_x, cur_y + th), scale=scale, color=STROKE, weight=1)
                    sw, _ = text_size(sep, scale=scale, weight=1)
                    cur_part_x += sw
        else:
            put_text(frame, line, (content_x, cur_y + th), scale=scale, color=TEXT, weight=1)
            
        cur_y += th + line_spacing

    return box_w, box_h


def clean_note_box(
    frame: np.ndarray,
    text_lines: list[str],
    x: int,
    y: int,
    *,
    title: Optional[str] = None,
    color: Optional[tuple[int, int, int]] = None,
    scale: float = 0.55,
    weight: int = 1,
) -> tuple[int, int]:
    """Backwards-compatible wrapper routing directly to the focus_note_box."""
    return focus_note_box(
        frame,
        text_lines,
        x,
        y,
        title=title,
        badge="INFO" if title else None,
        accent_color=color,
        scale=scale,
        weight=weight,
    )


def chip(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: Optional[tuple[int, int, int]] = None,
    filled: bool = True,
    align_right: bool = False,
) -> tuple[int, int]:
    """
    Renders dynamic status/theme chips.
    Automatically calculates text bounds and clamps 'x' so long theme names
    like 'SYNTHWAVE' are never cut off at screen edges.
    """
    if color is None:
        color = ACCENT
    pad_x, pad_y = 16, 8
    tw, th = text_size(text, scale=0.48, weight=3)
    w, h = tw + pad_x * 2, th + pad_y * 2
    
    fh, fw = frame.shape[:2]

    # Support right alignment anchor
    if align_right:
        x = x - w

    # Safety screen bounds clamping (Prevents clipping off top/right screen edges)
    x = max(12, min(x, fw - w - 12))
    y = max(12, min(y, fh - h - 12))

    r = h // 2
    if filled:
        # Masked rounded pill fill
        roi = frame[y:y+h, x:x+w]
        if roi.shape[0] == h and roi.shape[1] == w:
            bg_patch = np.empty_like(roi)
            bg_patch[:] = color
            mask = _get_rounded_mask(h, w, r)
            alpha_m = (mask.astype(np.float32) / 255.0)[..., None]
            blended = (bg_patch.astype(np.float32) * alpha_m + roi.astype(np.float32) * (1.0 - alpha_m)).astype(np.uint8)
            frame[y:y+h, x:x+w] = blended
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.48, color=BG, weight=3, shadow=False)
    else:
        glass_panel(frame, (x, y), (x + w, y + h), alpha=0.9, radius=r, border=False, accent_top=False, cyber_brackets=False)
        rounded_rect(frame, (x, y), (x + w, y + h), color, radius=r, thickness=1)
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.48, color=color, weight=3)

    return w, h


def fit_image_to_canvas(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    bg: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
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


def vignette(frame: np.ndarray, strength: float = 0.35) -> np.ndarray:
    h, w = frame.shape[:2]
    key = (h, w, int(strength * 100))
    mask = _VIGNETTE_CACHE.get(key)
    if mask is None:
        ys = np.linspace(-1, 1, h, dtype=np.float32)
        xs = np.linspace(-1, 1, w, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv * xv + yv * yv)
        mask = (1.0 - np.clip((r - 0.45) / 0.65, 0.0, 1.0) * strength).astype(np.float32)
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
        display = smooth_toward(display, target, 0.2)

    rounded_rect(frame, (x, y), (x + width, y + height), STROKE_SOFT, radius=height // 2)
    fill = int(width * display)
    if fill > 2:
        rounded_rect(frame, (x, y), (x + max(height, fill), y + height), ACCENT, radius=height // 2)
        glow_dot(frame, (x + fill, y + height // 2), height, ACCENT_HOT, intensity=0.7, layers=4)
    return display