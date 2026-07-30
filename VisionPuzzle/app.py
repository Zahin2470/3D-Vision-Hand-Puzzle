"""Vision Hand Puzzle — fast dual-hand jigsaw, clean screen, full-res display."""

from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from VisionPuzzle import ui
from VisionPuzzle.effects import Effects
from VisionPuzzle.jigsaw import JigsawPuzzle
from VisionPuzzle.overlay import (
    draw_dual_selection,
    draw_framing_link,
    draw_hand_cursors,
    draw_hands,
    draw_help,
    draw_hud,
    draw_win,
)
from VisionPuzzle.pointer import DualPointerEngine, DualPointerState
from VisionPuzzle.tracker import VisionPuzzleer


class Mode(Enum):
    SELECT = auto()
    PLAY = auto()
    WIN = auto()


class VisionPuzzleApp:
    """Two-hand framing + dual-hand jigsaw assembly."""

    MIN_SEL = 140
    INFER_WIDTH = 960  # tracking resolution (display stays camera-native)

    def __init__(self, camera_index: int = 0) -> None:
        root = Path(__file__).resolve().parent
        model = root / "models" / "hand_landmarker.task"
        self.tracker = VisionPuzzleer(model, max_hands=2)
        self.pointer = DualPointerEngine()
        self.fx = Effects()
        self.camera_index = camera_index
        self.show_help = False
        self._fps = 0.0
        self._frames = 0
        self._fps_t = time.perf_counter()

        self.mode = Mode.SELECT
        self.grid = 4
        self._corner_a: Optional[tuple[float, float]] = None
        self._corner_b: Optional[tuple[float, float]] = None
        self._smooth_a: Optional[tuple[float, float]] = None
        self._smooth_b: Optional[tuple[float, float]] = None
        self._sel_locked = False
        self._framing = False
        self._frozen: Optional[np.ndarray] = None
        self.puzzle: Optional[JigsawPuzzle] = None
        self._progress_disp: Optional[float] = None
        self._play_started: Optional[float] = None
        self._win_celebrated = False
        self._last_live: Optional[np.ndarray] = None

    def run(self) -> int:
        cap = self._open_camera()
        if cap is None:
            print("ERROR: Could not open webcam.")
            return 1

        # Prefer sharp HD for display quality
        for size in ((1920, 1080), (1280, 720)):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            ok, probe = cap.read()
            if ok and probe is not None and probe.shape[1] >= size[0] * 0.75:
                break
        cap.set(cv2.CAP_PROP_FPS, 60)
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        win = "VisionPuzzle Studio"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        ok, first = cap.read()
        if ok and first is not None:
            cv2.resizeWindow(win, first.shape[1], first.shape[0])
        else:
            cv2.resizeWindow(win, 1280, 720)

        self.fx.flash_fade(0.45)
        print("VisionPuzzle Studio — dual-hand frame · SPACE · H help · Q quit")

        try:
            while True:
                ok, live = cap.read()
                if not ok:
                    break
                live = cv2.flip(live, 1)
                # One display copy keeps the camera buffer clean for HD freezes
                frame_base = live.copy()
                self._last_live = frame_base
                h, w = frame_base.shape[:2]
                now = time.perf_counter()

                hands = self.tracker.process(
                    live, mirrored=True, infer_max_width=self.INFER_WIDTH,
                )
                dual = self.pointer.update(hands)

                if self.mode == Mode.SELECT:
                    frame = self._update_select(frame_base, dual, w, h, now)
                else:
                    frame = self._update_play(frame_base, dual, w, h, now)

                frame = ui.vignette(frame, strength=0.18)
                self.fx.update()
                frame = self.fx.draw(frame)

                self._tick_fps()
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
        finally:
            self.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        cap = cv2.VideoCapture(self.camera_index)
        return cap if cap.isOpened() else None

    # ---- SELECT -----------------------------------------------------------

    def _update_select(
        self, live: np.ndarray, dual: DualPointerState, w: int, h: int, now: float,
    ) -> np.ndarray:
        # Live view while framing; only freeze when locked
        if self._sel_locked and self._frozen is not None:
            frame = self._frozen.copy()
        else:
            frame = live

        if dual.left and dual.right and not self._sel_locked:
            if dual.both_pinching:
                self._framing = True
                self._corner_a = (dual.left.x * (w - 1), dual.left.y * (h - 1))
                self._corner_b = (dual.right.x * (w - 1), dual.right.y * (h - 1))
            elif dual.both_falling and self._framing:
                self._framing = False
                if self._selection_valid():
                    self._sel_locked = True
                    self._frozen = live.copy()
                    self.fx.burst(
                        (self._corner_a[0] + self._corner_b[0]) * 0.5,
                        (self._corner_a[1] + self._corner_b[1]) * 0.5,
                        color=ui.ACCENT, n=12,
                    )
                else:
                    self._clear_selection()

        if self._corner_a and self._corner_b:
            self._smooth_a = ui.lerp_point(self._smooth_a, self._corner_a, 0.42)
            self._smooth_b = ui.lerp_point(self._smooth_b, self._corner_b, 0.42)
            frame = draw_dual_selection(
                frame,
                self._smooth_a[0], self._smooth_a[1],
                self._smooth_b[0], self._smooth_b[1],
                locked=self._sel_locked,
                active=self._framing,
                grid=self.grid,
            )

        frame = draw_framing_link(frame, dual, w, h)
        # Dotted only when not framing-heavy; solid is fine for speed in select too
        frame = draw_hands(frame, [p.hand for p in dual.hands], light=True)
        draw_hand_cursors(frame, dual.hands, t=now)

        frame, _ = draw_hud(
            frame,
            title="ZAHIN",
            extra=f"{self.grid}×{self.grid}",
        )
        if self.show_help:
            frame = draw_help(frame, [
                "Both hands pinch to frame",
                "Release to lock · SPACE to create",
                "3 / 4 / 5 grid · C clear · Q quit",
            ])
        return frame

    def _selection_valid(self) -> bool:
        if not self._corner_a or not self._corner_b:
            return False
        x0, x1 = sorted((self._corner_a[0], self._corner_b[0]))
        y0, y1 = sorted((self._corner_a[1], self._corner_b[1]))
        return (x1 - x0) >= self.MIN_SEL and (y1 - y0) >= self.MIN_SEL

    def _clear_selection(self) -> None:
        self._corner_a = None
        self._corner_b = None
        self._smooth_a = None
        self._smooth_b = None
        self._sel_locked = False
        self._framing = False
        self._frozen = None

    def _start_puzzle(self) -> None:
        if not self._selection_valid():
            return
        if self._frozen is None:
            # Capture now if user hit SPACE mid-frame without release-lock
            return
        src = self._frozen
        x0, x1 = sorted((int(self._corner_a[0]), int(self._corner_b[0])))  # type: ignore
        y0, y1 = sorted((int(self._corner_a[1]), int(self._corner_b[1])))  # type: ignore
        y0, y1 = max(0, y0), min(src.shape[0], y1)
        x0, x1 = max(0, x0), min(src.shape[1], x1)
        crop = src[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return

        fh, fw = src.shape[:2]
        max_w, max_h = int(fw * 0.72), int(fh * 0.72)
        aspect = crop.shape[1] / max(1, crop.shape[0])
        board_w = max_w
        board_h = int(board_w / aspect)
        if board_h > max_h:
            board_h = max_h
            board_w = int(board_h * aspect)
        board_w = max(self.grid * 48, board_w)
        board_h = max(self.grid * 48, board_h)
        board_x = (fw - board_w) // 2
        board_y = (fh - board_h) // 2 + 12

        self.puzzle = JigsawPuzzle.from_image(
            crop, self.grid, self.grid, board_x, board_y, board_w, board_h,
        )
        self._progress_disp = 0.0
        self._play_started = time.perf_counter()
        self._win_celebrated = False
        self.fx.flash_fade(0.4)
        self.mode = Mode.PLAY

    # ---- PLAY -------------------------------------------------------------

    def _update_play(
        self, live: np.ndarray, dual: DualPointerState, w: int, h: int, now: float,
    ) -> np.ndarray:
        assert self.puzzle is not None
        frame = live
        # Fast backdrop veil via ROI multiply-ish blend
        cv2.addWeighted(frame, 0.48, np.full_like(frame, ui.BG), 0.52, 0, dst=frame)

        if self.mode == Mode.PLAY:
            for hp in dual.hands:
                key = hp.handedness
                px, py = hp.x * (w - 1), hp.y * (h - 1)
                if hp.pinch_rising:
                    self.puzzle.pick(key, px, py)
                if hp.pinching:
                    self.puzzle.drag(key, px, py, w, h)
                if hp.pinch_falling:
                    snapped, center = self.puzzle.drop(key)
                    if snapped and center is not None:
                        self.fx.burst(center[0], center[1], color=ui.SUCCESS, n=14)
                    if self.puzzle.completed:
                        self.mode = Mode.WIN

        frame = self.puzzle.draw(frame, show_guides=True)
        if self.mode == Mode.PLAY:
            frame = self.puzzle.draw_reference(frame)

        frame = draw_hands(frame, [p.hand for p in dual.hands], light=True)
        draw_hand_cursors(frame, dual.hands, t=now)

        progress = self.puzzle.placed_count / max(1, self.puzzle.total)
        if self.mode == Mode.WIN:
            if not self._win_celebrated:
                self.fx.confetti(w, h, n=48)
                self.fx.flash_fade(0.35)
                self._win_celebrated = True
            frame = draw_win(frame, t=now)

        frame, self._progress_disp = draw_hud(
            frame,
            title="ZAHIN",
            extra=f"{self.puzzle.placed_count}/{self.puzzle.total}",
            progress=progress,
            progress_display=self._progress_disp,
        )
        if self.show_help and self.mode == Mode.PLAY:
            frame = draw_help(frame, [
                "Pinch to lift · release to snap",
                "R reshuffle · N new · Q quit",
            ])
        return frame

    # ---- keys -------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        if key in (ord("3"), ord("4"), ord("5")) and self.mode == Mode.SELECT:
            self.grid = int(chr(key))
        if key in (ord("c"), ord("C")) and self.mode == Mode.SELECT:
            self._clear_selection()
        if key in (ord(" "), 13) and self.mode == Mode.SELECT:
            if self._selection_valid():
                if self._frozen is None and self._last_live is not None:
                    self._frozen = self._last_live.copy()
                self._sel_locked = True
                self._start_puzzle()
        if key in (ord("r"), ord("R")) and self.puzzle is not None:
            self.puzzle.shuffle()
            self._progress_disp = 0.0
            self._play_started = time.perf_counter()
            self._win_celebrated = False
            self.fx.flash_fade(0.3)
            self.mode = Mode.PLAY
        if key in (ord("n"), ord("N")):
            self.puzzle = None
            self._clear_selection()
            self._progress_disp = None
            self._play_started = None
            self._win_celebrated = False
            self.fx.flash_fade(0.3)
            self.mode = Mode.SELECT
        return True

    def _tick_fps(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self._fps = self._frames / (now - self._fps_t)
            self._frames = 0
            self._fps_t = now
