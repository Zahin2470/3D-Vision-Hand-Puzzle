"""VisionPuzzle.audio — ambient background music + one-shot SFX.

Built on `pygame.mixer`. Designed to fail soft: if pygame isn't
installed, no audio device is available, or an asset file is simply
missing, every method silently no-ops instead of crashing the app.
This lets you drop the game in on a machine with no speakers, or ship
it before you've recorded every cue, without touching this file.

Expected asset layout (relative to VisionPuzzle/assets/audio/):

    music/select.ogg     looping bed for Selection mode
    music/play.ogg       looping bed for Play mode
    sfx/pinch.wav        short tick — pinch engaged
    sfx/lock.wav         selection frame locked
    sfx/snap.wav         piece snapped into its slot
    sfx/shuffle.wav      board reshuffled
    sfx/win.wav          puzzle completed
    sfx/rotate.wav       piece rotated (Hard mode)

Any file you haven't recorded yet is simply skipped — nothing breaks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        print(f"[audio] {msg}")
        _WARNED.add(msg)


class AudioManager:
    """Owns pygame's mixer: one music stream + a pool of SFX sounds."""

    SFX_FILES = {
        "pinch": "pinch.wav",
        "lock": "lock.wav",
        "snap": "snap.wav",
        "shuffle": "shuffle.wav",
        "win": "win.wav",
        "rotate": "rotate.wav",
    }
    MUSIC_FILES = {
        "select": "select.ogg",
        "play": "play.ogg",
    }

    def __init__(self, assets_dir: Optional[Path] = None, *, enabled: bool = True) -> None:
        if assets_dir is None:
            assets_dir = Path(__file__).resolve().parent / "assets" / "audio"
        self.assets_dir = Path(assets_dir)
        self.music_volume = 0.35
        self.sfx_volume = 0.85
        self.muted = False

        self._ok = False
        self._pygame = None
        self._sfx: dict[str, object] = {}
        self._current_track: Optional[str] = None
        self._last_played: dict[str, float] = {}

        if enabled:
            self._init_mixer()
            if self._ok:
                self._load_sfx()

    # -- setup ---------------------------------------------------------------

    def _init_mixer(self) -> None:
        try:
            import pygame  # local import: keeps pygame optional for the rest of the app
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            self._ok = True
        except Exception as exc:  # pragma: no cover - environment dependent
            _warn_once(f"disabled — no audio backend available ({exc})")
            self._ok = False

    def _load_sfx(self) -> None:
        sfx_dir = self.assets_dir / "sfx"
        for name, fname in self.SFX_FILES.items():
            path = sfx_dir / fname
            if not path.is_file():
                continue
            try:
                snd = self._pygame.mixer.Sound(str(path))
                snd.set_volume(self.sfx_volume)
                self._sfx[name] = snd
            except Exception as exc:
                _warn_once(f"could not load sfx '{name}': {exc}")

    # -- music -----------------------------------------------------------------

    def play_music(self, track: str = "select", *, loop: bool = True, fade_ms: int = 700) -> None:
        """Crossfade to a looping music bed. No-op if already playing it."""
        if not self._ok or track == self._current_track:
            return
        fname = self.MUSIC_FILES.get(track)
        if not fname:
            _warn_once(f"unknown music track '{track}'")
            return
        path = self.assets_dir / "music" / fname
        if not path.is_file():
            # Remember the intent so we don't retry every frame, but stay silent.
            self._current_track = track
            return
        try:
            if self._current_track is not None:
                self._pygame.mixer.music.fadeout(fade_ms)
            self._pygame.mixer.music.load(str(path))
            self._pygame.mixer.music.set_volume(0.0 if self.muted else self.music_volume)
            self._pygame.mixer.music.play(loops=-1 if loop else 0, fade_ms=fade_ms)
            self._current_track = track
        except Exception as exc:
            _warn_once(f"could not play music '{track}': {exc}")

    def stop_music(self, *, fade_ms: int = 500) -> None:
        if not self._ok:
            return
        try:
            self._pygame.mixer.music.fadeout(fade_ms)
        except Exception:
            pass
        self._current_track = None

    # -- sfx ---------------------------------------------------------------------

    def play_sfx(self, name: str, *, cooldown: float = 0.06) -> None:
        """Fire a one-shot SFX. `cooldown` guards against re-trigger spam
        (e.g. a pinch flickering across two frames)."""
        if not self._ok or self.muted:
            return
        snd = self._sfx.get(name)
        if snd is None:
            return
        now = time.perf_counter()
        if now - self._last_played.get(name, 0.0) < cooldown:
            return
        self._last_played[name] = now
        try:
            snd.play()
        except Exception:
            pass

    # -- controls ------------------------------------------------------------------

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self._ok:
            try:
                self._pygame.mixer.music.set_volume(0.0 if self.muted else self.music_volume)
            except Exception:
                pass
        return self.muted

    def set_music_volume(self, value: float) -> None:
        self.music_volume = max(0.0, min(1.0, value))
        if self._ok and not self.muted:
            try:
                self._pygame.mixer.music.set_volume(self.music_volume)
            except Exception:
                pass

    def set_sfx_volume(self, value: float) -> None:
        self.sfx_volume = max(0.0, min(1.0, value))
        for snd in self._sfx.values():
            try:
                snd.set_volume(self.sfx_volume)
            except Exception:
                pass

    def close(self) -> None:
        if not self._ok:
            return
        try:
            self._pygame.mixer.music.stop()
            self._pygame.mixer.quit()
        except Exception:
            pass
