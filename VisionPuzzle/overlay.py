"""Lean visual overlays — fast drawing, minimal on-screen text."""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from VisionPuzzle.landmarks import CONNECTIONS, LANDMARK_COLOR
from VisionPuzzle.tracker import HandResult
from VisionPuzzle import ui


def _dotted_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    gap: int = 5,
    dash: int = 8,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist < 1.0:
        return
    steps = max(1, int(dist // (dash + gap)))
    for i in range(steps + 1):
        t0 = i * (dash + gap) / dist
        t1 = min(1.0, (i * (dash + gap) + dash) / dist)
        if t0 >= 1.0:
            break
        a = (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0))
        b = (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def draw_hands(frame: np.ndarray, hands: list[HandResult], *, light: bool = False) -> np.ndarray:
    """Skeleton overlay. `light` uses solid lines (faster) for play mode."""
    h, w = frame.shape[:2]
    tip_r = 4 if light else 6

    for hand in hands:
        pts: list[tuple[int, int]] = []
        for lm in hand.landmarks:
            x = int(np.clip(lm[0], 0.0, 1.0) * (w - 1))
            y = int(np.clip(lm[1], 0.0, 1.0) * (h - 1))
            pts.append((x, y))

        for a, b, color in CONNECTIONS:
            if light:
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
            else:
                _dotted_line(frame, pts[a], pts[b], color, thickness=2, gap=5, dash=9)

        tips = {4, 8, 12, 16, 20}
        for i, (x, y) in enumerate(pts):
            color = LANDMARK_COLOR.get(i, (200, 200, 200))
            radius = tip_r if i in tips or i == 0 else 3
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
    return frame


def draw_hand_cursors(frame: np.ndarray, pointers, *, t: Optional[float] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    now = t if t is not None else time.perf_counter()
    pulse = 0.5 + 0.5 * math.sin(now * 6.0)

    for p in pointers:
        cx, cy = int(p.x * (w - 1)), int(p.y * (h - 1))
        color = ui.ACCENT_HOT if p.pinching else ui.ACCENT
        rad = 26 + int(5 * pulse) if p.pinching else 18
        # ROI-only glow (no full-frame copy)
        x0, y0 = max(0, cx - rad - 2), max(0, cy - rad - 2)
        x1, y1 = min(w, cx + rad + 2), min(h, cy + rad + 2)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            glow = roi.copy()
            cv2.circle(glow, (cx - x0, cy - y0), rad, color, -1, cv2.LINE_AA)
            alpha = 0.12 if p.pinching else 0.07
            cv2.addWeighted(glow, alpha, roi, 1.0 - alpha, 0, dst=roi)

        cv2.circle(frame, (cx, cy), 15, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
        if p.pinching:
            cv2.circle(frame, (cx, cy), 22 + int(3 * pulse), color, 1, cv2.LINE_AA)
    return frame


def draw_dual_selection(
    frame: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    locked: bool = False,
    active: bool = False,
    grid: int = 4,
    label: str = "",
) -> np.ndarray:
    h, w = frame.shape[:2]
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w - 1, xb), min(h - 1, yb)
    if xb - xa < 2 or yb - ya < 2:
        return frame

    color = ui.SUCCESS if locked else (ui.ACCENT_HOT if active else ui.ACCENT)

    # Fast exterior dim — four rectangles, no blur / float32 full-frame math
    dim = 0.42
    if ya > 0:
        frame[0:ya, :] = (frame[0:ya, :].astype(np.float32) * dim).astype(np.uint8)
    if yb < h - 1:
        frame[yb:, :] = (frame[yb:, :].astype(np.float32) * dim).astype(np.uint8)
    if xa > 0:
        frame[ya:yb, 0:xa] = (frame[ya:yb, 0:xa].astype(np.float32) * dim).astype(np.uint8)
    if xb < w - 1:
        frame[ya:yb, xb:] = (frame[ya:yb, xb:].astype(np.float32) * dim).astype(np.uint8)

    cv2.rectangle(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)
    arm = max(14, min(36, (xb - xa) // 8, (yb - ya) // 8))
    for (cx, cy, dx, dy) in (
        (xa, ya, 1, 1), (xb, ya, -1, 1), (xa, yb, 1, -1), (xb, yb, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)

    if grid >= 2 and (xb - xa) > 80 and (yb - ya) > 80:
        tw, th = (xb - xa) / grid, (yb - ya) / grid
        for i in range(1, grid):
            x = int(xa + i * tw)
            y = int(ya + i * th)
            cv2.line(frame, (x, ya), (x, yb), color, 1, cv2.LINE_AA)
            cv2.line(frame, (xa, y), (xb, y), color, 1, cv2.LINE_AA)

    return frame


def draw_hud(
    frame: np.ndarray,
    *,
    title: str,
    message: str = "",
    fps: float = 0.0,
    extra: str = "",
    progress: Optional[float] = None,
    progress_display: Optional[float] = None,
    step: int = 1,
    timer: str = "",
) -> tuple[np.ndarray, Optional[float]]:
    """Minimal top bar — brand + thin progress only."""
    h, w = frame.shape[:2]
    top = 44 if progress is None else 52
    frame = ui.glass_panel(frame, (0, 0), (w, top), alpha=0.55, radius=0, border=False, accent_top=True)

    cv2.circle(frame, (22, 22), 5, ui.ACCENT, -1, cv2.LINE_AA)
    ui.put_text(frame, title, (36, 28), scale=0.62, color=ui.ACCENT, weight=2)
    if extra:
        ui.put_text(frame, extra[:24], (w - 120, 28), scale=0.48, color=ui.TEXT_MUTED, shadow=False)

    disp = progress_display
    if progress is not None:
        disp = ui.progress_bar(frame, 36, 38, w - 72, 5, progress, display=disp)
    return frame, disp


def draw_framing_link(frame: np.ndarray, dual, w: int, h: int) -> np.ndarray:
    if not (dual.left and dual.right and dual.both_pinching):
        return frame
    a = (int(dual.left.x * (w - 1)), int(dual.left.y * (h - 1)))
    b = (int(dual.right.x * (w - 1)), int(dual.right.y * (h - 1)))
    cv2.line(frame, a, b, ui.ACCENT, 2, cv2.LINE_AA)
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    cv2.circle(frame, mid, 4, ui.ACCENT, -1, cv2.LINE_AA)
    return frame


def draw_help(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 22 * len(lines) + 20
    x1, y1 = 14, h - panel_h - 14
    x2 = min(w - 14, 400)
    frame = ui.glass_panel(frame, (x1, y1), (x2, h - 14), alpha=0.6, radius=12, accent_top=True)
    for i, line in enumerate(lines):
        ui.put_text(frame, line, (x1 + 14, y1 + 28 + i * 22), scale=0.44, color=ui.TEXT)
    return frame


def draw_win(frame: np.ndarray, *, t: Optional[float] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    now = t if t is not None else time.perf_counter()
    pulse = 0.5 + 0.5 * math.sin(now * 2.5)
    cx, cy = w // 2, h // 2
    frame = ui.glass_panel(
        frame, (cx - 220, cy - 55), (cx + 220, cy + 55),
        alpha=0.78, radius=18, accent_top=True,
    )
    cv2.circle(frame, (cx, cy - 6), 54 + int(3 * pulse), ui.SUCCESS, 1, cv2.LINE_AA)
    ui.put_text(frame, "COMPLETE", (cx - 95, cy + 6), scale=0.95, color=ui.SUCCESS, weight=2)
    return frame


def draw_status_chip(frame: np.ndarray, text: str, x: int, y: int, color=ui.ACCENT) -> None:
    # Kept for API compat — unused in lean HUD
    pass
