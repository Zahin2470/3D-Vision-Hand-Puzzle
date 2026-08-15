"""Lean visual overlays — fast drawing, minimal on-screen text."""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from VisionPuzzle.landmarks import (
    CONNECTIONS,
    DEPTH,
    PALM_TRIANGLES,
    TIP_PAIRINGS,
    TIPS,
    WEB_BRIDGES,
)
from VisionPuzzle.leaderboard import AddResult, LeaderboardEntry, format_time
from VisionPuzzle.tracker import HandResult
from VisionPuzzle import ui

# Map MediaPipe landmark indices to finger IDs (0: Thumb, 1: Index, 2: Middle, 3: Ring, 4: Pinky)
LANDMARK_FINGER_MAP: dict[int, int] = {}
for _f_idx, _group in enumerate([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]):
    for _lm_idx in _group:
        LANDMARK_FINGER_MAP[_lm_idx] = _f_idx


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


def _lerp_color(c0, c1, t: float):
    t = max(0.0, min(1.0, t))
    return (int(c0[0] + (c1[0]-c0[0])*t), int(c0[1] + (c1[1]-c0[1])*t), int(c0[2] + (c1[2]-c0[2])*t))


def _draw_web_arc(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    center: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
    sag: float = 0.35,
) -> None:
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    cx = int(mx + (center[0] - mx) * sag)
    cy = int(my + (center[1] - my) * sag)

    pts = np.array([p0, (cx, cy), p1], dtype=np.int32)
    t = np.linspace(0, 1, 12)[:, None]
    curve = (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t**2 * pts[2]
    cv2.polylines(img, [curve.astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_hands(
    frame: np.ndarray,
    hands: list[HandResult],
    *,
    light: bool = False,
    t: Optional[float] = None,
) -> np.ndarray:
    """Cyber HUD Overlay: Holographic triangulation mesh, outer tip shield,

    and spinning reticles on index tip. Re-colors live on theme switch (T key).
    """
    h, w = frame.shape[:2]
    now = t if t is not None else time.perf_counter()
    base, hot = ui.ACCENT, ui.ACCENT_HOT  # Read theme live

    # Theme-shifted per-finger gradients
    base_hsv = cv2.cvtColor(np.uint8([[[*base]]]), cv2.COLOR_BGR2HSV)[0][0]
    hot_hsv = cv2.cvtColor(np.uint8([[[*hot]]]), cv2.COLOR_BGR2HSV)[0][0]

    finger_theme_colors: dict[int, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
    for f_idx in range(5):
        hue_shift = f_idx * 14.0
        b_h = int((float(base_hsv[0]) + hue_shift) % 180)
        h_h = int((float(hot_hsv[0]) + hue_shift) % 180)

        c0 = cv2.cvtColor(np.uint8([[[b_h, base_hsv[1], base_hsv[2]]]]), cv2.COLOR_HSV2BGR)[0][0]
        c1 = cv2.cvtColor(np.uint8([[[h_h, hot_hsv[1], hot_hsv[2]]]]), cv2.COLOR_HSV2BGR)[0][0]
        finger_theme_colors[f_idx] = (
            (int(c0[0]), int(c0[1]), int(c0[2])),
            (int(c1[0]), int(c1[1]), int(c1[2])),
        )

    for hand in hands:
        pts: list[tuple[int, int]] = []
        for lm in hand.landmarks:
            x = int(np.clip(lm[0], 0.0, 1.0) * (w - 1))
            y = int(np.clip(lm[1], 0.0, 1.0) * (h - 1))
            pts.append((x, y))

        if len(pts) < 21:
            continue

        palm_pts = [pts[i] for i in (0, 1, 5, 9, 13, 17)]
        palm_center = (
            int(sum(p[0] for p in palm_pts) / len(palm_pts)),
            int(sum(p[1] for p in palm_pts) / len(palm_pts)),
        )

        # 1. Holographic Palm Triangulation Mesh
        if not light:
            for tri in PALM_TRIANGLES:
                tri_pts = np.array([pts[i] for i in tri], dtype=np.int32)

                # Facet wireframe
                cv2.polylines(frame, [tri_pts], True, base, 1, cv2.LINE_AA)

                # Translucent facet fill
                x_tmin, y_tmin = max(0, int(np.min(tri_pts[:, 0]))), max(0, int(np.min(tri_pts[:, 1])))
                x_tmax, y_tmax = min(w, int(np.max(tri_pts[:, 0])) + 1), min(h, int(np.max(tri_pts[:, 1])) + 1)
                if x_tmax > x_tmin and y_tmax > y_tmin:
                    roi = frame[y_tmin:y_tmax, x_tmin:x_tmax]
                    overlay = roi.copy()
                    shifted_pts = tri_pts - np.array([x_tmin, y_tmin])
                    cv2.fillConvexPoly(overlay, shifted_pts, base, lineType=cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.08, roi, 0.92, 0, dst=roi)

        # 2. Outer Fingertip Perimeter Shield (Tip-to-Tip Arc Boundary)
        for ta, tb in TIP_PAIRINGS:
            _draw_web_arc(frame, pts[ta], pts[tb], palm_center, hot, thickness=1, sag=-0.25)

        # 3. Inter-Finger Web Strands
        for idx, (a, b) in enumerate(WEB_BRIDGES):
            f_a = LANDMARK_FINGER_MAP.get(a, 0)
            f_b = LANDMARK_FINGER_MAP.get(b, 0)
            c0_a, c1_a = finger_theme_colors[f_a]
            c0_b, c1_b = finger_theme_colors[f_b]
            web_color = _lerp_color(c0_a, c1_b, 0.5)
            _draw_web_arc(frame, pts[a], pts[b], palm_center, web_color, thickness=1, sag=0.32)

        # 4. Main Skeletal Connections & Energy Pulses
        for a, b in CONNECTIONS:
            pa, pb = pts[a], pts[b]
            da, db = DEPTH.get(a, 0.0), DEPTH.get(b, 0.0)

            finger_id = LANDMARK_FINGER_MAP.get(b, LANDMARK_FINGER_MAP.get(a))
            if finger_id is not None:
                c0, c1 = finger_theme_colors[finger_id]
                color = _lerp_color(c0, c1, (da + db) / 2.0)
            else:
                color = _lerp_color(base, hot, (da + db) / 2.0)

            cv2.line(frame, pa, pb, (10, 10, 10), 4, cv2.LINE_AA)
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

            phase = (now * 2.5 + da * 0.7) % 1.0
            px = int(pa[0] + (pb[0] - pa[0]) * phase)
            py = int(pa[1] + (pb[1] - pa[1]) * phase)
            cv2.circle(frame, (px, py), 2, hot, -1, cv2.LINE_AA)

        # 5. Joint Nodes, Wrist Web-Shooter & Index Tip HUD Reticle
        for i, (x, y) in enumerate(pts):
            depth = DEPTH.get(i, 0.0)
            finger_id = LANDMARK_FINGER_MAP.get(i)

            if finger_id is not None:
                c0, c1 = finger_theme_colors[finger_id]
                node_color = _lerp_color(c0, c1, depth)
            else:
                node_color = _lerp_color(base, hot, depth)

            if i in TIPS:
                pulse = int(2.5 * math.sin(now * 7.0 + i))
                tip_hot = finger_theme_colors[finger_id][1] if finger_id is not None else hot
                ui.glow_dot(frame, (x, y), 12 + pulse, tip_hot, intensity=0.35)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, tip_hot, -1, cv2.LINE_AA)

                # Concentric HUD Reticles on Index Tip (Landmark 8)
                if i == 8:
                    rot_angle = (now * 140) % 360
                    # Outer target ring
                    cv2.circle(frame, (x, y), 20, hot, 1, cv2.LINE_AA)

                    # Rotating orbital dots
                    for deg in range(0, 360, 45):
                        rad = math.radians(deg + rot_angle)
                        rx = int(x + 14 * math.cos(rad))
                        ry = int(y + 14 * math.sin(rad))
                        if 0 <= rx < w and 0 <= ry < h:
                            cv2.circle(frame, (rx, ry), 1, hot, -1, cv2.LINE_AA)

                    # Target reticle ticks
                    for deg in (0, 90, 180, 270):
                        rad = math.radians(deg - rot_angle * 0.5)
                        x1 = int(x + 16 * math.cos(rad))
                        y1 = int(y + 16 * math.sin(rad))
                        x2 = int(x + 24 * math.cos(rad))
                        y2 = int(y + 24 * math.sin(rad))
                        cv2.line(frame, (x1, y1), (x2, y2), hot, 1, cv2.LINE_AA)

            elif i == 0:
                ui.glow_dot(frame, (x, y), 16, hot, intensity=0.4)
                cv2.circle(frame, (x, y), 7, hot, 2, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, base, -1, cv2.LINE_AA)
            else:
                r = 3 + int(depth * 3)
                cv2.circle(frame, (x, y), r + 1, (10, 10, 10), 1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), r, node_color, -1, cv2.LINE_AA)

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
    """Floating rounded HUD pill, sized to its content — not a full-width
    bar. Reads more like a modern game/OS widget than a status strip."""
    h, w = frame.shape[:2]
    margin = 14
    body_h = 42 if progress is None else 50

    title_w, _ = ui.text_size(title, scale=0.62, weight=2)
    extra = extra[:28]
    extra_w = ui.text_size(extra, scale=0.46, weight=1)[0] if extra else 0
    content_w = 44 + title_w + (28 + extra_w if extra else 0) + 20
    panel_w = int(min(w - margin * 2, max(190, content_w)))

    x1, y1 = margin, margin
    x2, y2 = x1 + panel_w, y1 + body_h

    frame = ui.glass_panel(
        frame, (x1, y1), (x2, y2),
        alpha=0.62, radius=18, border=True, gradient=True, shadow=True,
    )

    cy = y1 + (24 if progress is None else 26)
    dot = (x1 + 20, cy - 6)
    ui.glow_dot(frame, dot, 11, ui.ACCENT, intensity=0.28)
    cv2.circle(frame, dot, 5, ui.ACCENT, -1, cv2.LINE_AA)
    ui.put_text(frame, title, (x1 + 34, cy), scale=0.62, color=ui.ACCENT, weight=2)
    if extra:
        ui.put_text(frame, extra, (x2 - extra_w - 18, cy - 1), scale=0.46, color=ui.TEXT_MUTED, shadow=False)

    disp = progress_display
    if progress is not None:
        disp = ui.progress_bar(frame, x1 + 18, y2 - 13, panel_w - 36, 4, progress, display=disp)
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


def draw_help(frame: np.ndarray, lines: list[str], *, enter_t: float = 1.0) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 22 * len(lines) + 20
    eased = ui.ease_out_cubic(enter_t)
    slide = int((1.0 - eased) * 24)  # starts pushed down, slides up into place
    x1, y1 = 14, h - panel_h - 14 + slide
    x2 = min(w - 14, 400)
    frame = ui.glass_panel(
        frame, (x1, y1), (x2, h - 14 + slide), alpha=0.6 * eased, radius=12, accent_top=True, gradient=True,
    )
    for i, line in enumerate(lines):
        ui.put_text(frame, line, (x1 + 14, y1 + 28 + i * 22), scale=0.44, color=ui.TEXT)
    return frame


def draw_win(frame: np.ndarray, *, t: Optional[float] = None, result: Optional[AddResult] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    now = t if t is not None else time.perf_counter()
    pulse = 0.5 + 0.5 * math.sin(now * 2.5)
    cx, cy = w // 2, h // 2
    extra = 30 if result is not None else 0
    frame = ui.glass_panel(
        frame, (cx - 220, cy - 55 - extra // 2), (cx + 220, cy + 55 + extra // 2),
        alpha=0.78, radius=18, accent_top=True, gradient=True, shadow=True,
    )
    ring_r = 54 + int(3 * pulse)
    ui.glow_dot(frame, (cx, cy - 6 - extra // 2), ring_r + 16, ui.SUCCESS, intensity=0.22)
    cv2.circle(frame, (cx, cy - 6 - extra // 2), ring_r, ui.SUCCESS, 1, cv2.LINE_AA)
    ui.put_text(frame, "COMPLETE", (cx - 95, cy + 6 - extra // 2), scale=0.95, color=ui.SUCCESS, weight=2)

    if result is not None:
        sub = f"Time  {format_time(result.entry.time_seconds)}"
        if result.is_new_best:
            sub += "   ·   NEW BEST!"
        elif result.made_board:
            sub += f"   ·   #{result.rank} best"
        sub_w, _ = ui.text_size(sub, scale=0.5, weight=1)
        badge_color = ui.SUCCESS if result.is_new_best else ui.TEXT_MUTED
        ui.put_text(frame, sub, (cx - sub_w // 2, cy + 40), scale=0.5, color=badge_color, weight=1)

    return frame


def draw_leaderboard(
    frame: np.ndarray,
    entries: list[LeaderboardEntry],
    label: str,
    *,
    highlight: Optional[LeaderboardEntry] = None,
    enter_t: float = 1.0,
) -> np.ndarray:
    """Compact top-times panel, anchored top-right below the HUD bar."""
    h, w = frame.shape[:2]
    rows = max(1, len(entries))
    panel_h = 46 + rows * 24 + 10
    eased = ui.ease_out_cubic(enter_t)
    slide = int((1.0 - eased) * -20)  # starts higher, drops into place
    x2 = w - 14
    x1 = max(14, x2 - 220)
    y1 = 66 + slide
    y2 = y1 + panel_h
    frame = ui.glass_panel(
        frame, (x1, y1), (x2, y2), alpha=0.68 * eased, radius=12, accent_top=True, gradient=True,
    )
    ui.put_text(frame, f"BEST · {label}", (x1 + 14, y1 + 24), scale=0.46, color=ui.ACCENT, weight=2)

    if not entries:
        ui.put_text(frame, "No times yet — go solve one!", (x1 + 14, y1 + 48), scale=0.42, color=ui.TEXT_MUTED)
        return frame

    for i, e in enumerate(entries):
        y = y1 + 48 + i * 24
        is_hl = highlight is not None and e is highlight
        if i == 0:
            color = ui.SUCCESS
        elif is_hl:
            color = ui.ACCENT_HOT
        else:
            color = ui.TEXT
        ui.put_text(frame, f"{i + 1}.", (x1 + 14, y), scale=0.44, color=color)
        ui.put_text(frame, format_time(e.time_seconds), (x1 + 42, y), scale=0.44, color=color)
        if is_hl:
            cv2.circle(frame, (x2 - 16, y - 5), 3, ui.ACCENT_HOT, -1, cv2.LINE_AA)

    return frame


def draw_status_chip(frame: np.ndarray, text: str, x: int, y: int, color=None) -> None:
    # Kept for API compat — unused in lean HUD
    pass
