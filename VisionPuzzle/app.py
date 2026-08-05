"""Vision Hand Puzzle — fast dual-hand jigsaw, clean screen, full-res display."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from VisionPuzzle import ui
from VisionPuzzle.audio import AudioManager
from VisionPuzzle.effects import Effects
from VisionPuzzle.jigsaw import JigsawPuzzle
from VisionPuzzle.leaderboard import AddResult, Leaderboard
from VisionPuzzle.savegame import SaveGame
from VisionPuzzle.share import build_card, save_card
from VisionPuzzle.overlay import (
    draw_dual_selection,
    draw_framing_link,
    draw_hand_cursors,
    draw_hands,
    draw_help,
    draw_hud,
    draw_leaderboard,
    draw_win,
)
from VisionPuzzle.pointer import DualPointerEngine, DualPointerState, angle_diff
from VisionPuzzle.tracker import VisionPuzzleer


def _data_root(package_root: Path) -> Path:
    """Where user data (leaderboard, saves, settings, snapshots) lives.

    Running from source, this is right next to the code — easy to poke
    at while developing. Running as a packaged/frozen app (PyInstaller
    etc.), the app bundle itself is often installed somewhere read-only
    (Program Files, /Applications, a Gatekeeper-verified .app), so we
    use the platform's normal per-user data folder instead — the same
    place any other installed desktop app would keep its data.
    """
    if not getattr(sys, "frozen", False):
        return package_root
    app_name = "VisionPuzzle Studio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / app_name
    return Path.home() / ".local" / "share" / app_name.lower().replace(" ", "-")


class Mode(Enum):
    SELECT = auto()
    PLAY = auto()
    WIN = auto()


class VisionPuzzleApp:
    """Two-hand framing + dual-hand jigsaw assembly."""

    MIN_SEL = 140
    INFER_WIDTH = 960  # tracking resolution (display stays camera-native)

    def __init__(self, camera_index: int = 0, initial_image: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parent
        data_root = _data_root(root)
        model = root / "models" / "hand_landmarker.task"
        self.tracker = VisionPuzzleer(model, max_hands=2)
        self.pointer = DualPointerEngine()
        self.fx = Effects()
        self.audio = AudioManager(root / "assets" / "audio")
        self.leaderboard = Leaderboard(data_root / "data" / "leaderboard.json")
        self.show_leaderboard = False
        self._leaderboard_since: Optional[float] = None
        self._last_result: Optional[AddResult] = None
        self.savegame = SaveGame(data_root / "data")
        self._has_save = self.savegame.exists()
        self.camera_index = camera_index
        self._initial_image_path = Path(initial_image) if initial_image else None
        self._uploaded_image: Optional[np.ndarray] = None
        self._using_upload = False
        self._twist_base: dict[str, float] = {}
        self.TWIST_STEP_DEG = 42.0
        self._settings_path = data_root / "data" / "settings.json"
        self._load_settings()
        self._snapshots_dir = data_root / "snapshots"
        self._pending_share_frame = False
        self._last_share_path: Optional[Path] = None
        self._fullscreen = False
        self._win_name = "VisionPuzzle Studio"
        self.show_help = False
        self._help_since: Optional[float] = None
        self._fps = 0.0
        self._frames = 0
        self._fps_t = time.perf_counter()

        self.mode = Mode.SELECT
        self.grid = 4
        self.difficulty = "normal"  # "normal" | "hard" (hard adds piece rotation)
        self.players = 1  # 1 | 2 (2 splits the board: Left hand vs Right hand)
        self._corner_a: Optional[tuple[float, float]] = None
        self._corner_b: Optional[tuple[float, float]] = None
        self._smooth_a: Optional[tuple[float, float]] = None
        self._smooth_b: Optional[tuple[float, float]] = None
        self._sel_locked = False
        self._framing = False
        self._frozen: Optional[np.ndarray] = None
        self.puzzle: Optional[JigsawPuzzle] = None
        self._progress_disp: Optional[float] = None
        self._placed_disp = 0.0
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
        self._win_name = win
        # WINDOW_NORMAL (not AUTOSIZE) is required for the fullscreen (F key)
        # toggle to work reliably. We re-assert the window size below so it
        # still starts locked to the camera's native resolution.
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        ok, first = cap.read()
        if ok and first is not None:
            frame_w, frame_h = first.shape[1], first.shape[0]
            cv2.resizeWindow(win, frame_w, frame_h)
        else:
            frame_w, frame_h = 1280, 720
            cv2.resizeWindow(win, frame_w, frame_h)
        print(f"[display] camera frame size: {frame_w}x{frame_h}")

        if self._initial_image_path is not None:
            canvas = self._load_image_source(self._initial_image_path, frame_w, frame_h)
            if canvas is not None:
                self._uploaded_image = canvas
                self._using_upload = True

        self.fx.flash_fade(0.45)
        self.audio.play_music("select")
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

                if self._pending_share_frame:
                    self._pending_share_frame = False
                    self._build_and_save_share(frame)

                self._tick_fps()
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
        finally:
            self.tracker.close()
            self.audio.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _load_image_source(self, path, target_w: int, target_h: int) -> Optional[np.ndarray]:
        """Read an image file and letterbox it to the camera-frame canvas."""
        img = cv2.imread(str(path))
        if img is None:
            print(f"[upload] could not read image: {path}")
            return None
        return ui.fit_image_to_canvas(img, target_w, target_h)

    def _pick_file_macos(self) -> Optional[str]:
        """Native Cocoa file dialog via AppleScript — no Tcl/Tk involved,
        so it sidesteps the Tk-on-macOS crash entirely."""
        script = 'POSIX path of (choose file with prompt "Choose a puzzle image")'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=300,
            )
            return result.stdout.strip() or None
        except Exception as exc:
            print(f"[upload] macOS file picker failed: {exc}")
            return None

    def _pick_file_windows(self) -> Optional[str]:
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.OpenFileDialog; "
            "$f.Filter = 'Images|*.jpg;*.jpeg;*.png;*.bmp;*.webp'; "
            "if ($f.ShowDialog() -eq 'OK') { Write-Output $f.FileName }"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=300,
            )
            return result.stdout.strip() or None
        except Exception as exc:
            print(f"[upload] Windows file picker failed: {exc}")
            return None

    def _pick_file_linux(self) -> Optional[str]:
        for cmd in (
            ["zenity", "--file-selection", "--title=Choose a puzzle image",
             "--file-filter=Images | *.jpg *.jpeg *.png *.bmp *.webp"],
            ["kdialog", "--getopenfilename", ".", "*.jpg *.jpeg *.png *.bmp *.webp"],
        ):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                path = result.stdout.strip()
                if path:
                    return path
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"[upload] {cmd[0]} failed: {exc}")
        return None

    def _pick_file_tkinter_subprocess(self) -> Optional[str]:
        """Last-resort fallback: run tkinter's file dialog in an isolated
        child process. Some Python/Tcl-Tk builds on macOS crash the whole
        interpreter when a Tk window is created — running it as a
        subprocess means that crash can only take down the child, never
        this app's camera loop."""
        script = (
            "import tkinter as tk\n"
            "from tkinter import filedialog\n"
            "root = tk.Tk()\n"
            "root.withdraw()\n"
            "path = filedialog.askopenfilename(\n"
            "    title='Choose a puzzle image',\n"
            "    filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All files', '*.*')],\n"
            ")\n"
            "print(path)\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip() or None
        except Exception as exc:
            print(f"[upload] file picker subprocess failed: {exc}")
            return None

    def _upload_image(self) -> None:
        """Open a native file picker and swap the puzzle source to it.
        Tries a platform-native dialog first (no Tk dependency at all);
        only falls back to a sandboxed tkinter subprocess if nothing
        native is available."""
        path: Optional[str] = None
        if sys.platform == "darwin":
            path = self._pick_file_macos()
        elif sys.platform == "win32":
            path = self._pick_file_windows()
        elif sys.platform.startswith("linux"):
            path = self._pick_file_linux()
        if not path:
            path = self._pick_file_tkinter_subprocess()
        if not path:
            print("[upload] no image chosen, or no file picker is available here. "
                  "You can always launch with: python main.py --image PATH")
            return

        if self._last_live is None:
            return
        h, w = self._last_live.shape[:2]
        canvas = self._load_image_source(path, w, h)
        if canvas is None:
            return
        self._uploaded_image = canvas
        self._using_upload = True
        self._clear_selection()
        self.fx.flash_fade(0.35)

    def _load_settings(self) -> None:
        if not self._settings_path.is_file():
            return
        try:
            data = json.loads(self._settings_path.read_text(encoding="utf-8"))
            theme = data.get("theme")
            if theme:
                ui.set_theme(theme)
        except Exception as exc:
            print(f"[settings] could not read {self._settings_path}: {exc}")

    def _save_settings(self) -> None:
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._settings_path.write_text(
                json.dumps({"theme": ui.get_theme_name()}), encoding="utf-8",
            )
        except Exception as exc:
            print(f"[settings] could not save {self._settings_path}: {exc}")

    def _build_and_save_share(self, frame: np.ndarray) -> None:
        if self.puzzle is None:
            return
        solve_time = self._last_result.entry.time_seconds if self._last_result else 0.0
        card = build_card(
            frame,
            board_label=self._board_label(),
            solve_time=solve_time,
            result=self._last_result,
        )
        path = save_card(card, self._snapshots_dir)
        if path is not None:
            print(f"[share] saved {path}")
            self._last_share_path = path

    def _enter_t(self, since: Optional[float], now: float, duration: float = 0.18) -> float:
        if since is None:
            return 1.0
        return min(1.0, max(0.0, (now - since) / duration))

    def _board_key(self) -> str:
        """Leaderboard identifier for the current grid + difficulty —
        Hard-mode rotation makes solves slower, so it gets its own board."""
        suffix = "h" if self.difficulty == "hard" else ""
        suffix += "-2p" if self.players == 2 else ""
        return f"{self.grid}{suffix}"

    def _board_label(self) -> str:
        suffix = " · HARD" if self.difficulty == "hard" else ""
        suffix += " · 2P" if self.players == 2 else ""
        src = " · IMG" if self._using_upload else ""
        return f"{self.grid}\u00d7{self.grid}{suffix}{src}"

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
        elif self._using_upload and self._uploaded_image is not None:
            frame = self._uploaded_image.copy()
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
                    self._frozen = (
                        self._uploaded_image.copy()
                        if self._using_upload and self._uploaded_image is not None
                        else live.copy()
                    )
                    self.fx.burst(
                        (self._corner_a[0] + self._corner_b[0]) * 0.5,
                        (self._corner_a[1] + self._corner_b[1]) * 0.5,
                        color=ui.ACCENT, n=12,
                    )
                    self.audio.play_sfx("lock")
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
            extra=self._board_label(),
        )
        theme_label = ui.get_theme_name().upper()
        theme_w, _ = ui.text_size(theme_label, scale=0.42, weight=1)
        ui.chip(frame, theme_label, w - 14 - (theme_w + 28), 14, color=ui.TEXT_MUTED)
        if self._has_save and not self._sel_locked:
            ui.put_text(
                frame, "P  Resume last game", (16, h - 16),
                scale=0.44, color=ui.ACCENT_HOT,
            )
        if not self._sel_locked and not self._framing:
            ui.put_text(
                frame, "U  Upload an image", (16, h - (36 if self._has_save else 16)),
                scale=0.44, color=ui.TEXT_MUTED,
            )
        if self.show_leaderboard:
            frame = draw_leaderboard(
                frame, self.leaderboard.top(self._board_key(), 5), self._board_label(),
                enter_t=self._enter_t(self._leaderboard_since, now),
            )
        if self.show_help:
            frame = draw_help(frame, [
                "Both hands pinch to frame",
                "Release to lock · SPACE to create",
                "3 / 4 / 5 grid · D difficulty · 2 two-player",
                "U upload · C clear · T theme · F fullscreen",
                "M mute · L leaderboard · Q quit",
            ], enter_t=self._enter_t(self._help_since, now))
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
            allow_rotation=(self.difficulty == "hard"),
            two_player=(self.players == 2),
        )
        now = time.perf_counter()
        self.puzzle.start_shatter(now)
        self.fx.burst(board_x + board_w * 0.5, board_y + board_h * 0.5, color=ui.ACCENT, n=30)
        self._progress_disp = 0.0
        self._placed_disp = 0.0
        self._play_started = now
        self._win_celebrated = False
        self._last_result = None
        self.fx.flash_fade(0.4)
        self.audio.play_music("play")
        self.mode = Mode.PLAY

    def _resume_saved_game(self) -> None:
        result = self.savegame.load()
        self._has_save = False
        if result is None:
            return
        puzzle, elapsed = result
        self.puzzle = puzzle
        self.grid = puzzle.rows
        self.difficulty = "hard" if puzzle.allow_rotation else "normal"
        self.players = 2 if puzzle.two_player else 1
        self._progress_disp = puzzle.placed_count / max(1, puzzle.total)
        self._placed_disp = float(puzzle.placed_count)
        self._play_started = time.perf_counter() - elapsed
        self._win_celebrated = False
        self._last_result = None
        self.fx.flash_fade(0.4)
        self.audio.play_music("play")
        self.mode = Mode.PLAY

    def _autosave(self, now: float) -> None:
        """Snapshot the live puzzle so it can be resumed later. Safe to
        call often — no-ops if there's nothing in-progress to save."""
        if self.mode != Mode.PLAY or self.puzzle is None or self.puzzle.completed:
            return
        elapsed = now - (self._play_started or now)
        self.savegame.save(self.puzzle, elapsed_seconds=elapsed)

    # ---- PLAY -------------------------------------------------------------

    def _update_play(
        self, live: np.ndarray, dual: DualPointerState, w: int, h: int, now: float,
    ) -> np.ndarray:
        assert self.puzzle is not None
        frame = live
        # Fast backdrop veil via ROI multiply-ish blend
        cv2.addWeighted(frame, 0.48, np.full_like(frame, ui.BG), 0.52, 0, dst=frame)

        self.puzzle.tick_shatter(now)

        if self.mode == Mode.PLAY and not self.puzzle.shattering:
            for hp in dual.hands:
                key = hp.handedness
                px, py = hp.x * (w - 1), hp.y * (h - 1)
                if hp.pinch_rising:
                    if self.puzzle.pick(key, px, py):
                        self.audio.play_sfx("pinch")
                    self._twist_base[key] = hp.angle_deg
                if hp.pinching:
                    self.puzzle.drag(key, px, py, w, h)
                    if self.puzzle.allow_rotation and key in self.puzzle.held:
                        base = self._twist_base.get(key)
                        if base is None:
                            self._twist_base[key] = hp.angle_deg
                        else:
                            delta = angle_diff(hp.angle_deg, base)
                            if abs(delta) >= self.TWIST_STEP_DEG:
                                step = 90 if delta > 0 else -90
                                if self.puzzle.rotate_piece(key, step):
                                    self.audio.play_sfx("rotate")
                                self._twist_base[key] = hp.angle_deg
                if hp.pinch_falling:
                    self._twist_base.pop(key, None)
                    snapped, center = self.puzzle.drop(key)
                    if snapped and center is not None:
                        self.fx.burst(center[0], center[1], color=ui.SUCCESS, n=14)
                        self.audio.play_sfx("snap")
                        self._autosave(now)
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
                self.audio.play_sfx("win")
                self._win_celebrated = True
                solve_time = now - (self._play_started or now)
                self._last_result = self.leaderboard.add(self._board_key(), solve_time)
                self.savegame.clear()
                self._pending_share_frame = True
                self._last_share_path = None
            frame = draw_win(frame, t=now, result=self._last_result)

        # Count-up micro-animation: the number visibly ticks toward the
        # real value instead of jumping, so each snap feels like it lands.
        self._placed_disp = ui.smooth_toward(self._placed_disp, float(self.puzzle.placed_count), 0.35)
        placed_shown = int(round(self._placed_disp))

        frame, self._progress_disp = draw_hud(
            frame,
            title="ZAHIN",
            extra=f"{placed_shown}/{self.puzzle.total}",
            progress=progress,
            progress_display=self._progress_disp,
        )
        theme_label = ui.get_theme_name().upper()
        theme_w, _ = ui.text_size(theme_label, scale=0.42, weight=1)
        ui.chip(frame, theme_label, w - 14 - (theme_w + 28), 14, color=ui.TEXT_MUTED)
        if self.puzzle.two_player:
            by_owner = self.puzzle.progress_by_owner()
            lp, lt = by_owner.get("Left", (0, 0))
            rp, rt = by_owner.get("Right", (0, 0))
            ui.put_text(
                frame, f"P1 (L)  {lp}/{lt}      P2 (R)  {rp}/{rt}",
                (16, 70), scale=0.42, color=ui.TEXT_MUTED,
            )
        if self.show_leaderboard:
            highlight = self._last_result.entry if self._last_result else None
            frame = draw_leaderboard(
                frame, self.leaderboard.top(self._board_key(), 5), self._board_label(), highlight=highlight,
                enter_t=self._enter_t(self._leaderboard_since, now),
            )
        if self.show_help and self.mode == Mode.PLAY:
            lines = [
                "Pinch to lift · release to snap",
                "R reshuffle · N new · Q quit",
                "M mute/unmute audio · L leaderboard",
            ]
            if self.puzzle is not None and self.puzzle.allow_rotation:
                lines.insert(1, "Twist wrist to rotate · [ / ] also works")
            frame = draw_help(frame, lines, enter_t=self._enter_t(self._help_since, now))
        if self.mode == Mode.WIN and self._last_share_path is not None:
            ui.put_text(
                frame, f"Saved  snapshots/{self._last_share_path.name}", (16, h - 16),
                scale=0.4, color=ui.TEXT_MUTED,
            )
        return frame

    # ---- keys -------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            self._autosave(time.perf_counter())
            return False
        if key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
            if self.show_help:
                self._help_since = time.perf_counter()
        if key in (ord("3"), ord("4"), ord("5")) and self.mode == Mode.SELECT:
            self.grid = int(chr(key))
        if key in (ord("d"), ord("D")) and self.mode == Mode.SELECT:
            self.difficulty = "hard" if self.difficulty == "normal" else "normal"
        if key == ord("2") and self.mode == Mode.SELECT:
            self.players = 2 if self.players == 1 else 1
        if key in (ord("u"), ord("U")) and self.mode == Mode.SELECT:
            self._upload_image()
        if key in (ord("c"), ord("C")) and self.mode == Mode.SELECT:
            self._clear_selection()
        if key in (ord(" "), 13) and self.mode == Mode.SELECT:
            if self._selection_valid():
                if self._frozen is None:
                    if self._using_upload and self._uploaded_image is not None:
                        self._frozen = self._uploaded_image.copy()
                    elif self._last_live is not None:
                        self._frozen = self._last_live.copy()
                self._sel_locked = True
                self._start_puzzle()
        if key in (ord("r"), ord("R")) and self.puzzle is not None:
            self.puzzle.shuffle()
            now = time.perf_counter()
            self.puzzle.start_shatter(now, duration=0.45, max_stagger=0.14)
            self._progress_disp = 0.0
            self._placed_disp = 0.0
            self._play_started = now
            self._win_celebrated = False
            self._last_result = None
            self._pending_share_frame = False
            self._last_share_path = None
            self.fx.flash_fade(0.3)
            self.audio.play_sfx("shuffle")
            self.mode = Mode.PLAY
        if key in (ord("n"), ord("N")):
            self.puzzle = None
            self._clear_selection()
            self._progress_disp = None
            self._play_started = None
            self._win_celebrated = False
            self._last_result = None
            self._pending_share_frame = False
            self._last_share_path = None
            self.savegame.clear()
            self._has_save = False
            self._uploaded_image = None
            self._using_upload = False
            self.fx.flash_fade(0.3)
            self.audio.play_music("select")
            self.mode = Mode.SELECT
        if key in (ord("m"), ord("M")):
            self.audio.toggle_mute()
        if key in (ord("l"), ord("L")):
            self.show_leaderboard = not self.show_leaderboard
            if self.show_leaderboard:
                self._leaderboard_since = time.perf_counter()
        if key in (ord("t"), ord("T")):
            ui.set_theme(ui.next_theme_name())
            self._save_settings()
            self.fx.flash_fade(0.25)
        if key in (ord("f"), ord("F")):
            self._fullscreen = not self._fullscreen
            prop = cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self._win_name, cv2.WND_PROP_FULLSCREEN, prop)
        if key == ord("[") and self.mode == Mode.PLAY and self.puzzle is not None:
            if self.puzzle.rotate_held(-90):
                self.audio.play_sfx("rotate")
        if key == ord("]") and self.mode == Mode.PLAY and self.puzzle is not None:
            if self.puzzle.rotate_held(90):
                self.audio.play_sfx("rotate")
        if key in (ord("p"), ord("P")) and self.mode == Mode.SELECT and self._has_save:
            self._resume_saved_game()
        return True

    def _tick_fps(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self._fps = self._frames / (now - self._fps_t)
            self._frames = 0
            self._fps_t = now
