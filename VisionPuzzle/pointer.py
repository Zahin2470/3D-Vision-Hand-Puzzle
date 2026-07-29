"""Dual-hand pointer engine — pinch, tips, and two-corner framing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from VisionPuzzle.tracker import HandResult


@dataclass
class HandPointer:
    """One hand's index cursor + pinch state."""

    handedness: str
    x: float
    y: float
    pinching: bool
    pinch_rising: bool
    pinch_falling: bool
    score: float
    hand: HandResult


@dataclass
class DualPointerState:
    """Combined two-hand interaction state."""

    hands: list[HandPointer] = field(default_factory=list)
    left: Optional[HandPointer] = None
    right: Optional[HandPointer] = None
    both_pinching: bool = False
    both_rising: bool = False   # transitioned into dual-pinch this frame
    both_falling: bool = False  # left dual-pinch this frame

    @property
    def count(self) -> int:
        return len(self.hands)

    def primary(self) -> Optional[HandPointer]:
        """Preferred play hand: currently pinching, else highest score."""
        pinching = [h for h in self.hands if h.pinching]
        if pinching:
            return max(pinching, key=lambda h: h.score)
        return self.hands[0] if self.hands else None


class DualPointerEngine:
    """Tracks up to two hands with stable per-hand pinch hysteresis."""

    PINCH_ON = 0.045
    PINCH_OFF = 0.075

    def __init__(self) -> None:
        self._pinch: dict[str, bool] = {}
        self._was_both = False
        self._smooth: dict[str, tuple[float, float]] = {}

    def update(self, hands: list[HandResult]) -> DualPointerState:
        pointers: list[HandPointer] = []
        seen: set[str] = set()

        # Prefer distinct Left/Right; fall back to spatial order
        ordered = self._order_hands(hands)
        for hand in ordered:
            key = hand.handedness
            # Avoid colliding keys if MediaPipe returns two "Left"
            if key in seen:
                key = f"{key}{hand.track_id}"
            seen.add(key)

            index = hand.landmarks[8]
            thumb = hand.landmarks[4]
            dist = float(np.linalg.norm(index[:2] - thumb[:2]))
            was = self._pinch.get(key, False)
            if was:
                pinching = dist < self.PINCH_OFF
            else:
                pinching = dist < self.PINCH_ON
            rising = pinching and not was
            falling = (not pinching) and was
            self._pinch[key] = pinching

            pointers.append(
                HandPointer(
                    handedness=hand.handedness,
                    x=float(np.clip(index[0], 0.0, 1.0)),
                    y=float(np.clip(index[1], 0.0, 1.0)),
                    pinching=pinching,
                    pinch_rising=rising,
                    pinch_falling=falling,
                    score=hand.score,
                    hand=hand,
                )
            )

        # Per-hand tip smoothing for butter-smooth cursors
        for p in pointers:
            key = p.handedness
            prev = self._smooth.get(key)
            if prev is None:
                self._smooth[key] = (p.x, p.y)
            else:
                # Faster when pinching (drag) so it doesn't feel laggy
                a = 0.55 if p.pinching else 0.38
                sx = prev[0] * (1 - a) + p.x * a
                sy = prev[1] * (1 - a) + p.y * a
                self._smooth[key] = (sx, sy)
                p.x, p.y = sx, sy
        for k in list(self._smooth.keys()):
            if k not in seen and not any(p.handedness == k for p in pointers):
                del self._smooth[k]

        # Drop stale pinch flags
        for k in list(self._pinch.keys()):
            if k not in seen:
                del self._pinch[k]

        left = next((p for p in pointers if p.handedness == "Left"), None)
        right = next((p for p in pointers if p.handedness == "Right"), None)
        # If labels missing, treat first two as corners by x position
        if len(pointers) >= 2 and (left is None or right is None):
            by_x = sorted(pointers, key=lambda p: p.x)
            left, right = by_x[0], by_x[-1]

        both = (
            left is not None
            and right is not None
            and left.pinching
            and right.pinching
        )
        both_rising = both and not self._was_both
        both_falling = (not both) and self._was_both
        self._was_both = both

        return DualPointerState(
            hands=pointers,
            left=left,
            right=right,
            both_pinching=both,
            both_rising=both_rising,
            both_falling=both_falling,
        )

    @staticmethod
    def _order_hands(hands: list[HandResult]) -> list[HandResult]:
        if len(hands) <= 1:
            return list(hands)
        lefts = [h for h in hands if h.handedness == "Left"]
        rights = [h for h in hands if h.handedness == "Right"]
        if lefts and rights:
            return [max(lefts, key=lambda h: h.score), max(rights, key=lambda h: h.score)]
        return sorted(hands, key=lambda h: h.landmarks[0, 0])[:2]
