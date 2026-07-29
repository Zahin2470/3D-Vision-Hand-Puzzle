"""MediaPipe Hand Landmarker with temporal smoothing for stable tracking."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def get_mediapipe_delegate():
    """Return a MediaPipe delegate that is safe for the current platform."""
    from mediapipe.tasks.python import BaseOptions

    if platform.system() == "Darwin":
        return BaseOptions.Delegate.CPU
    return None


@dataclass
class HandResult:
    """One detected hand."""

    landmarks: np.ndarray  # (21, 3) normalized x, y, z
    handedness: str        # "Left" | "Right" (viewer / mirrored space)
    score: float
    track_id: int = 0


class LandmarkSmoother:
    """Per-hand EMA smoother — cuts jitter without lagging too much."""

    def __init__(self, alpha: float = 0.42) -> None:
        self.alpha = alpha
        self._state: dict[str, np.ndarray] = {}
        self._miss: dict[str, int] = {}

    def reset(self) -> None:
        self._state.clear()
        self._miss.clear()

    def apply(self, key: str, landmarks: np.ndarray) -> np.ndarray:
        prev = self._state.get(key)
        if prev is None:
            smooth = landmarks.copy()
        else:
            # Adaptive blend: jump more when the hand moves fast
            delta = float(np.mean(np.abs(landmarks[:, :2] - prev[:, :2])))
            alpha = min(0.85, self.alpha + delta * 2.5)
            smooth = prev * (1.0 - alpha) + landmarks * alpha
        self._state[key] = smooth
        self._miss[key] = 0
        return smooth

    def mark_seen(self, keys: set[str]) -> None:
        for key in list(self._state.keys()):
            if key not in keys:
                self._miss[key] = self._miss.get(key, 0) + 1
                if self._miss[key] > 12:
                    del self._state[key]
                    self._miss.pop(key, None)


class VisionPuzzleer:
    """Tracks hands from BGR frames using MediaPipe HandLandmarker."""

    def __init__(self, model_path: Path, max_hands: int = 2) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.is_file():
            raise FileNotFoundError(f"Missing hand model: {model_path}")

        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=get_mediapipe_delegate(),
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            # Stricter thresholds → fewer false positives, stickier tracking
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._smoother = LandmarkSmoother(alpha=0.40)
        self._start_ms = int(time.perf_counter() * 1000)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, *, mirrored: bool = True, infer_max_width: int = 960) -> list[HandResult]:
        from mediapipe import Image as MpImage, ImageFormat
        import cv2

        # Infer on a downscaled copy — landmarks stay normalized, display stays full-res
        h, w = frame_bgr.shape[:2]
        if w > infer_max_width:
            scale = infer_max_width / float(w)
            small = cv2.resize(
                frame_bgr,
                (infer_max_width, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)

        ts = int(time.perf_counter() * 1000) - self._start_ms
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(mp_image, ts)

        hands: list[HandResult] = []
        if not result.hand_landmarks:
            self._smoother.mark_seen(set())
            return hands

        seen: set[str] = set()
        for i, lms in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            handedness = "Right"
            score = 0.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                handedness = cat.category_name
                score = float(cat.score)

            # After horizontal flip, MediaPipe labels are camera-space — swap for viewer
            if mirrored:
                handedness = "Left" if handedness == "Right" else "Right"

            key = handedness if handedness not in seen else f"{handedness}_{i}"
            seen.add(key)
            pts = self._smoother.apply(key, pts)
            hands.append(
                HandResult(
                    landmarks=pts,
                    handedness=handedness,
                    score=score,
                    track_id=i,
                )
            )

        self._smoother.mark_seen(seen)
        # Prefer higher-confidence hands first
        hands.sort(key=lambda h: h.score, reverse=True)
        return hands

    def close(self) -> None:
        self._landmarker.close()
