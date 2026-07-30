"""Hand landmark colors and connection topology (MediaPipe 21-point hand).

Vibrant rainbow theme: each finger gets an evenly-spaced, fully-saturated
hue, so thumb -> pinky sweeps crimson -> orange -> green -> blue -> violet.
Palm connections are pastel tints of the finger they lead into, so the
rainbow fans out softly from a gold wrist anchor instead of competing
with the finger colors for attention.
"""

from __future__ import annotations

import colorsys

# ---------------------------------------------------------------------------
# Finger landmark index groups (MediaPipe Hands)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

FINGER_ORDER = (THUMB, INDEX, MIDDLE, RING, PINKY)

Color = tuple[int, int, int]


def _hsv_to_bgr(h_deg: float, s: float, v: float) -> Color:
    """Convert HSV (hue in degrees) to an OpenCV-style BGR int tuple."""
    r, g, b = colorsys.hsv_to_rgb(h_deg / 360.0, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def _blend(c1: Color, c2: Color, t: float) -> Color:
    """Linearly blend two BGR colors; t=0 -> c1, t=1 -> c2."""
    b = tuple(int(a + (bb - a) * t) for a, bb in zip(c1, c2))
    return (b[0], b[1], b[2])


# ---------------------------------------------------------------------------
# Vibrant rainbow palette
# ---------------------------------------------------------------------------
# Anchor point: warm gold, distinct from every finger hue so the wrist
# always reads as the "root" of the hand.
COLOR_WRIST: Color = (0, 215, 255)  # BGR -> gold

# Hand-picked hues (degrees) rather than a blind even sweep, so adjacent
# fingers stay visually distinct instead of blurring into each other.
_HUE = {THUMB: 355, INDEX: 32, MIDDLE: 145, RING: 205, PINKY: 280}
_SATURATION = 0.82
_VALUE = 0.95

COLOR_THUMB = _hsv_to_bgr(_HUE[THUMB], _SATURATION, _VALUE)
COLOR_INDEX = _hsv_to_bgr(_HUE[INDEX], _SATURATION, _VALUE)
COLOR_MIDDLE = _hsv_to_bgr(_HUE[MIDDLE], _SATURATION, _VALUE)
COLOR_RING = _hsv_to_bgr(_HUE[RING], _SATURATION, _VALUE)
COLOR_PINKY = _hsv_to_bgr(_HUE[PINKY], _SATURATION, _VALUE)

FINGER_COLORS: dict[tuple[int, ...], Color] = {
    THUMB: COLOR_THUMB,
    INDEX: COLOR_INDEX,
    MIDDLE: COLOR_MIDDLE,
    RING: COLOR_RING,
    PINKY: COLOR_PINKY,
}

# Pastel tints used for the palm "root" connections, so the rainbow fans
# out softly from the wrist instead of competing with the fingers.
_PALM_TINT = 0.55  # 0 = full finger color, 1 = white
_WHITE: Color = (255, 255, 255)
COLOR_PALM_THUMB = _blend(COLOR_THUMB, _WHITE, _PALM_TINT)
COLOR_PALM_INDEX = _blend(COLOR_INDEX, _WHITE, _PALM_TINT)
COLOR_PALM_MIDDLE = _blend(COLOR_MIDDLE, _WHITE, _PALM_TINT)
COLOR_PALM_RING = _blend(COLOR_RING, _WHITE, _PALM_TINT)
COLOR_PALM_PINKY = _blend(COLOR_PINKY, _WHITE, _PALM_TINT)

# ---------------------------------------------------------------------------
# Connections: (start, end, color)
# ---------------------------------------------------------------------------
CONNECTIONS: list[tuple[int, int, Color]] = [
    # Palm — each spoke tinted toward the finger it feeds into
    (0, 1, COLOR_PALM_THUMB),
    (0, 5, COLOR_PALM_INDEX),
    (0, 17, COLOR_PALM_PINKY),
    (5, 9, _blend(COLOR_PALM_INDEX, COLOR_PALM_MIDDLE, 0.5)),
    (9, 13, _blend(COLOR_PALM_MIDDLE, COLOR_PALM_RING, 0.5)),
    (13, 17, _blend(COLOR_PALM_RING, COLOR_PALM_PINKY, 0.5)),
    # Thumb
    (1, 2, COLOR_THUMB),
    (2, 3, COLOR_THUMB),
    (3, 4, COLOR_THUMB),
    # Index
    (5, 6, COLOR_INDEX),
    (6, 7, COLOR_INDEX),
    (7, 8, COLOR_INDEX),
    # Middle
    (9, 10, COLOR_MIDDLE),
    (10, 11, COLOR_MIDDLE),
    (11, 12, COLOR_MIDDLE),
    # Ring
    (13, 14, COLOR_RING),
    (14, 15, COLOR_RING),
    (15, 16, COLOR_RING),
    # Pinky
    (17, 18, COLOR_PINKY),
    (18, 19, COLOR_PINKY),
    (19, 20, COLOR_PINKY),
]

LANDMARK_COLOR: dict[int, Color] = {WRIST: COLOR_WRIST}
for _group, _color in FINGER_COLORS.items():
    for _idx in _group:
        LANDMARK_COLOR[_idx] = _color

# ---------------------------------------------------------------------------
# Visual hierarchy: bigger dots / thicker lines toward the fingertips so
# the overlay reads with a sense of depth even on a flat 2D frame.
# ---------------------------------------------------------------------------
LANDMARK_RADIUS: dict[int, int] = {WRIST: 8}
for _group in FINGER_ORDER:
    for _pos, _idx in enumerate(_group):
        # knuckle (pos 0) smallest, fingertip (pos 3) largest
        LANDMARK_RADIUS[_idx] = 4 + _pos

CONNECTION_THICKNESS = 3      # finger segments
PALM_THICKNESS = 2            # palm spokes / arcs