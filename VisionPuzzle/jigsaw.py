"""Premium jigsaw board with smooth dual-hand piece control."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from VisionPuzzle import ui


@dataclass
class Piece:
    row: int
    col: int
    image: np.ndarray
    x: float
    y: float
    target_x: float
    target_y: float
    w: int
    h: int
    placed: bool = False
    grab_dx: float = 0.0
    grab_dy: float = 0.0
    holder: Optional[str] = None
    # Visual smooth position (lags slightly behind logical x/y for polish)
    draw_x: float = 0.0
    draw_y: float = 0.0
    lift: float = 0.0  # 0..1 held elevation


@dataclass
class JigsawPuzzle:
    source: np.ndarray
    rows: int
    cols: int
    board_x: int
    board_y: int
    board_w: int
    board_h: int
    pieces: list[Piece] = field(default_factory=list)
    held: dict[str, int] = field(default_factory=dict)
    snap_px: float = 28.0
    completed: bool = False
    last_snap: bool = False

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        rows: int,
        cols: int,
        board_x: int,
        board_y: int,
        board_w: int,
        board_h: int,
    ) -> "JigsawPuzzle":
        ih, iw = image.shape[:2]
        interp = cv2.INTER_AREA if (board_w < iw or board_h < ih) else cv2.INTER_CUBIC
        src = cv2.resize(image, (board_w, board_h), interpolation=interp)
        tw = board_w // cols
        th = board_h // rows
        usable_w, usable_h = tw * cols, th * rows
        src = src[:usable_h, :usable_w]
        board_w, board_h = usable_w, usable_h

        pieces: list[Piece] = []
        for r in range(rows):
            for c in range(cols):
                tile = src[r * th:(r + 1) * th, c * tw:(c + 1) * tw].copy()
                # Premium bevel
                cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (30, 32, 36), 1)
                cv2.line(tile, (1, 1), (tw - 2, 1), (255, 255, 255), 1)
                cv2.line(tile, (1, 1), (1, th - 2), (255, 255, 255), 1)
                pieces.append(
                    Piece(
                        row=r,
                        col=c,
                        image=tile,
                        x=0.0,
                        y=0.0,
                        target_x=float(board_x + c * tw),
                        target_y=float(board_y + r * th),
                        w=tw,
                        h=th,
                    )
                )

        puzzle = cls(
            source=src,
            rows=rows,
            cols=cols,
            board_x=board_x,
            board_y=board_y,
            board_w=board_w,
            board_h=board_h,
            pieces=pieces,
            snap_px=max(24.0, min(tw, th) * 0.32),
        )
        puzzle.shuffle()
        return puzzle

    def shuffle(self) -> None:
        self.completed = False
        self.held.clear()
        self.last_snap = False
        margin = 24
        span_x = max(margin, self.board_x + self.board_w - margin)
        span_y = max(margin, self.board_y + self.board_h - margin)
        for p in self.pieces:
            p.placed = False
            p.holder = None
            p.lift = 0.0
            for _ in range(8):
                p.x = float(random.randint(margin, max(margin, span_x - p.w)))
                p.y = float(random.randint(margin, max(margin, span_y - p.h)))
                if abs(p.x - p.target_x) > self.snap_px or abs(p.y - p.target_y) > self.snap_px:
                    break
            p.draw_x, p.draw_y = p.x, p.y

    @property
    def placed_count(self) -> int:
        return sum(1 for p in self.pieces if p.placed)

    @property
    def total(self) -> int:
        return len(self.pieces)

    def pick(self, hand_key: str, px: float, py: float) -> bool:
        if hand_key in self.held:
            return True
        for i in range(len(self.pieces) - 1, -1, -1):
            p = self.pieces[i]
            if p.placed or p.holder is not None:
                continue
            if p.draw_x <= px <= p.draw_x + p.w and p.draw_y <= py <= p.draw_y + p.h:
                p.x, p.y = p.draw_x, p.draw_y
                p.grab_dx = px - p.x
                p.grab_dy = py - p.y
                p.holder = hand_key
                self.pieces.append(self.pieces.pop(i))
                self.held[hand_key] = len(self.pieces) - 1
                return True
        return False

    def drag(self, hand_key: str, px: float, py: float, frame_w: int, frame_h: int) -> None:
        idx = next((i for i, p in enumerate(self.pieces) if p.holder == hand_key), None)
        if idx is None:
            self.held.pop(hand_key, None)
            return
        self.held[hand_key] = idx
        p = self.pieces[idx]
        target_x = float(np.clip(px - p.grab_dx, 0, max(0, frame_w - p.w)))
        target_y = float(np.clip(py - p.grab_dy, 0, max(0, frame_h - p.h)))
        # Smooth follow — feels premium, less jitter from hand noise
        p.x = ui.smooth_toward(p.x, target_x, 0.55)
        p.y = ui.smooth_toward(p.y, target_y, 0.55)
        # Magnetic assist when close to the correct slot
        dx = p.target_x - p.x
        dy = p.target_y - p.y
        dist = float(np.hypot(dx, dy))
        if dist < self.snap_px * 1.85:
            pull = 0.22 if dist < self.snap_px else 0.12
            p.x += dx * pull
            p.y += dy * pull

    def drop(self, hand_key: str) -> tuple[bool, Optional[tuple[float, float]]]:
        """Returns (snapped, snap_center_xy)."""
        self.last_snap = False
        idx = next((i for i, p in enumerate(self.pieces) if p.holder == hand_key), None)
        self.held.pop(hand_key, None)
        if idx is None:
            return False, None
        p = self.pieces[idx]
        p.holder = None
        if abs(p.x - p.target_x) <= self.snap_px and abs(p.y - p.target_y) <= self.snap_px:
            p.x = p.target_x
            p.y = p.target_y
            p.placed = True
            self.last_snap = True
            self.completed = all(q.placed for q in self.pieces)
            return True, (p.x + p.w * 0.5, p.y + p.h * 0.5)
        return False, None

    def near_slot(self, hand_key: str) -> bool:
        p = next((q for q in self.pieces if q.holder == hand_key), None)
        if p is None:
            return False
        return abs(p.x - p.target_x) <= self.snap_px and abs(p.y - p.target_y) <= self.snap_px

    def piece_center(self, hand_key: str) -> Optional[tuple[float, float]]:
        p = next((q for q in self.pieces if q.holder == hand_key), None)
        if p is None:
            return None
        return (p.draw_x + p.w * 0.5, p.draw_y + p.h * 0.5)

    def tick_visuals(self) -> None:
        """Advance draw positions / lift each frame."""
        for p in self.pieces:
            p.draw_x = ui.smooth_toward(p.draw_x, p.x, 0.45)
            p.draw_y = ui.smooth_toward(p.draw_y, p.y, 0.45)
            want = 1.0 if p.holder else 0.0
            p.lift = ui.smooth_toward(p.lift, want, 0.28)

    def draw_reference(self, frame: np.ndarray) -> np.ndarray:
        """Compact reference thumbnail (no text label)."""
        h, w = frame.shape[:2]
        tw = 110
        th = max(70, int(tw * self.source.shape[0] / max(1, self.source.shape[1])))
        thumb = cv2.resize(self.source, (tw, th), interpolation=cv2.INTER_AREA)
        x1, y1 = w - tw - 16, h - th - 16
        x2, y2 = x1 + tw, y1 + th
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), ui.BG_ELEVATED, -1)
        frame[y1:y2, x1:x2] = thumb
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), ui.ACCENT, 1, cv2.LINE_AA)
        return frame

    def draw(self, frame: np.ndarray, *, show_guides: bool = True) -> np.ndarray:
        self.tick_visuals()
        out = frame
        bx, by, bw, bh = self.board_x, self.board_y, self.board_w, self.board_h

        # Fast board well (ROI darken, no full-frame rounded overlay copies)
        pad = 8
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(out.shape[1], bx + bw + pad), min(out.shape[0], by + bh + pad)
        roi = out[y0:y1, x0:x1]
        dark = np.empty_like(roi)
        dark[:] = (18, 20, 24)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, dst=roi)
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), ui.ACCENT, 1, cv2.LINE_AA)

        # Light ghost
        if by + bh <= out.shape[0] and bx + bw <= out.shape[1]:
            board = out[by:by + bh, bx:bx + bw]
            cv2.addWeighted(self.source, 0.10, board, 0.90, 0, dst=board)

        if show_guides:
            tw, th = bw // self.cols, bh // self.rows
            for r in range(1, self.rows):
                y = by + r * th
                cv2.line(out, (bx, y), (bx + bw, y), ui.STROKE_SOFT, 1, cv2.LINE_AA)
            for c in range(1, self.cols):
                x = bx + c * tw
                cv2.line(out, (x, by), (x, by + bh), ui.STROKE_SOFT, 1, cv2.LINE_AA)

        # Soft shadow for free pieces only (cheap filled rects)
        for p in self.pieces:
            if p.placed:
                continue
            lift_px = int(8 * p.lift)
            sx = int(p.draw_x) + 3 + lift_px // 2
            sy = int(p.draw_y) + 5 + lift_px
            if 0 <= sx < out.shape[1] - p.w and 0 <= sy < out.shape[0] - p.h:
                sh = out[sy:sy + p.h, sx:sx + p.w]
                cv2.addWeighted(sh, 0.75, np.zeros_like(sh), 0.25, 0, dst=sh)

        ordered = sorted(self.pieces, key=lambda q: (0 if q.placed else 1, q.lift))
        for p in ordered:
            x = int(p.draw_x)
            y = int(p.draw_y) - int(6 * p.lift)
            if x < 0 or y < 0 or x + p.w > out.shape[1] or y + p.h > out.shape[0]:
                continue
            out[y:y + p.h, x:x + p.w] = p.image
            if p.placed:
                color, thick = ui.SUCCESS, 1
            elif p.holder:
                color, thick = ui.ACCENT_HOT, 2
            else:
                color, thick = (190, 194, 198), 1
            cv2.rectangle(out, (x, y), (x + p.w - 1, y + p.h - 1), color, thick, cv2.LINE_AA)
            if p.holder and abs(p.x - p.target_x) <= self.snap_px and abs(p.y - p.target_y) <= self.snap_px:
                cv2.rectangle(
                    out, (int(p.target_x), int(p.target_y)),
                    (int(p.target_x) + p.w - 1, int(p.target_y) + p.h - 1),
                    ui.SUCCESS, 2, cv2.LINE_AA,
                )

        return out
