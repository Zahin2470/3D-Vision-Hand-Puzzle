"""VisionPuzzle.share — render and save a shareable "I solved it!" card.

Takes the fully-composited winning frame (confetti, COMPLETE banner, and
all), crops it to a clean portrait hero shot, and stamps a branded stats
footer underneath — grid size, difficulty, solve time, and leaderboard
rank if there is one. Saved as a plain PNG in the project's snapshots
folder, ready to drop into a chat or a social post.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from VisionPuzzle import ui
from VisionPuzzle.leaderboard import AddResult, format_time


def build_card(
    win_frame: np.ndarray,
    *,
    board_label: str,
    solve_time: float,
    result: Optional[AddResult] = None,
    card_width: int = 720,
) -> np.ndarray:
    """Compose a shareable card image from the winning frame."""
    h, w = win_frame.shape[:2]

    # Center-crop to a portrait-ish 4:5 hero shot — reads well as a share image.
    target_ratio = 4 / 5
    if w / h > target_ratio:
        crop_w = max(1, int(h * target_ratio))
        x0 = max(0, (w - crop_w) // 2)
        hero = win_frame[:, x0:x0 + crop_w]
    else:
        crop_h = max(1, int(w / target_ratio))
        y0 = max(0, (h - crop_h) // 2)
        hero = win_frame[y0:y0 + crop_h, :]

    scale = card_width / hero.shape[1]
    hero = cv2.resize(hero, (card_width, max(1, int(hero.shape[0] * scale))), interpolation=cv2.INTER_AREA)

    footer_h = 132
    card = np.empty((hero.shape[0] + footer_h, card_width, 3), dtype=np.uint8)
    card[:] = ui.BG
    card[: hero.shape[0]] = hero

    fy = hero.shape[0]
    cv2.line(card, (0, fy), (card_width, fy), ui.ACCENT, 2, cv2.LINE_AA)

    ui.put_text(card, "PUZZLE SOLVED", (24, fy + 36), scale=0.85, color=ui.SUCCESS, weight=2)
    ui.put_text(
        card, f"{board_label}   \u00b7   {format_time(solve_time)}",
        (24, fy + 68), scale=0.55, color=ui.TEXT,
    )

    if result is not None:
        if result.is_new_best:
            tag = "NEW BEST TIME"
        elif result.made_board:
            tag = f"#{result.rank} FASTEST ON THIS BOARD"
        else:
            tag = ""
        if tag:
            ui.put_text(card, tag, (24, fy + 96), scale=0.48, color=ui.ACCENT_HOT)

    stamp = time.strftime("%Y-%m-%d %H:%M")
    stamp_w, _ = ui.text_size(stamp, scale=0.42, weight=1)
    ui.put_text(card, stamp, (card_width - stamp_w - 20, fy + footer_h - 14), scale=0.42, color=ui.TEXT_MUTED)

    return card


def save_card(card: np.ndarray, snapshots_dir: Path) -> Optional[Path]:
    """Write the card to disk as a timestamped PNG. Returns the path on
    success, None (and a console message) if it couldn't be written."""
    try:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = snapshots_dir / f"share_{time.strftime('%Y%m%d_%H%M%S')}.png"
        if cv2.imwrite(str(path), card):
            return path
        print(f"[share] cv2.imwrite failed for {path}")
    except Exception as exc:
        print(f"[share] could not save card: {exc}")
    return None
