"""Shared premium UI primitives — palette, glass panels, smooth motion."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ── Design tokens (BGR) ─────────────────────────────────────────────────────
BG = (16, 17, 20)
BG_ELEVATED = (22, 24, 28)
STROKE = (70, 76, 82)
STROKE_SOFT = (48, 52, 58)
TEXT = (236, 238, 240)
TEXT_MUTED = (160, 166, 172)
ACCENT = (196, 178, 92)
ACCENT_HOT = (140, 200, 255)
SUCCESS = (120, 200, 150)
DANGER = (90, 90, 220)
SHADOW = (0, 0, 0)

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
    color=TEXT,
    weight: int = 1,
    shadow: bool = True,
) -> None:
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
    color=ACCENT,
    filled: bool = False,
) -> None:
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
