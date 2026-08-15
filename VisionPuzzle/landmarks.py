"""Hand landmark topology (MediaPipe 21-point hand).

Just topology + index groups here — no baked-in colors. Colors are
computed live in overlay.py from the active theme (ui.ACCENT ->
ui.ACCENT_HOT), so switching themes (T key) re-colors the hand too,
not just the HUD panels.
"""

from __future__ import annotations

WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGER_ORDER = (THUMB, INDEX, MIDDLE, RING, PINKY)
TIPS = (4, 8, 12, 16, 20)

PALM_INDICES = (0, 1, 2, 5, 9, 13, 17)

CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

# Curved inter-finger web connections
WEB_BRIDGES: list[tuple[int, int]] = [
    (2, 5), (6, 10), (10, 14), (14, 18),  # Inner web ring
    (3, 7), (7, 11), (11, 15), (15, 19),  # Outer web ring
]

# 1. Tip-to-Tip pairings for outer perimeter energy shield
TIP_PAIRINGS: list[tuple[int, int]] = [
    (4, 8), (8, 12), (12, 16), (16, 20)
]

# 2. Triangular facets for holographic palm mesh
PALM_TRIANGLES: list[tuple[int, int, int]] = [
    (0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17)
]

DEPTH: dict[int, float] = {WRIST: 0.0}
for _group in FINGER_ORDER:
    for _pos, _idx in enumerate(_group):
        DEPTH[_idx] = (_pos + 1) / len(_group)

LANDMARK_RADIUS: dict[int, int] = {WRIST: 8}
for _group in FINGER_ORDER:
    for _pos, _idx in enumerate(_group):
        LANDMARK_RADIUS[_idx] = 3 + _pos