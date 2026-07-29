"""Hand landmark colors and connection topology (MediaPipe 21-point hand)."""

from __future__ import annotations

# Finger landmark index groups (MediaPipe Hands)
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

# BGR colors matching the reference overlay style
COLOR_WRIST = (40, 40, 220)       # red
COLOR_PALM = (160, 160, 160)      # grey palm links
COLOR_THUMB = (180, 200, 220)     # beige / cream
COLOR_INDEX = (220, 80, 180)      # purple
COLOR_MIDDLE = (40, 220, 220)     # yellow
COLOR_RING = (80, 200, 80)        # green
COLOR_PINKY = (255, 120, 40)      # blue (BGR)

FINGER_COLORS = {
    THUMB: COLOR_THUMB,
    INDEX: COLOR_INDEX,
    MIDDLE: COLOR_MIDDLE,
    RING: COLOR_RING,
    PINKY: COLOR_PINKY,
}

# Connections: (start, end, color)
CONNECTIONS: list[tuple[int, int, tuple[int, int, int]]] = [
    # Palm
    (0, 1, COLOR_PALM),
    (0, 5, COLOR_PALM),
    (0, 17, COLOR_PALM),
    (5, 9, COLOR_PALM),
    (9, 13, COLOR_PALM),
    (13, 17, COLOR_PALM),
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

LANDMARK_COLOR: dict[int, tuple[int, int, int]] = {0: COLOR_WRIST}
for group, color in FINGER_COLORS.items():
    for idx in group:
        LANDMARK_COLOR[idx] = color
