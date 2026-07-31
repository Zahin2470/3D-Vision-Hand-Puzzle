#!/usr/bin/env python3
"""Vision Hand Puzzle - premium dual-hand camera jigsaw.

Run
---
    pip install -r requirements.txt
    python main.py
    python main.py --image path/to/photo.jpg   # skip the camera, use a photo instead

Select
------
Show both hands
Pinch with BOTH hands          Frame opposite corners
Stretch L + R index tips       Resize the crop
Release both                   Lock selection
SPACE / Enter                  Create jigsaw
3 / 4 / 5                      Grid size
D                              Toggle Normal / Hard (Hard adds rotation)
2                              Toggle 1-player / 2-player (split board)
U                              Upload a custom image as the puzzle source
T                              Cycle color theme (dark/light/neon/mono)
C                              Clear

Play
----
Pinch a piece (either hand)    Grab
Both hands at once             Move two pieces
Release near slot              Snap
Twist your wrist               Rotate held piece (Hard mode)
[ / ]                          Rotate held piece (keyboard fallback)
R                              Reshuffle
N                              New capture

H                              Help
M                              Mute / unmute audio
T                              Cycle color theme
L                              Show / hide leaderboard
P                              Resume last save (Selection only)
Q / Esc                        Quit — a solved puzzle auto-saves a share
                               card to VisionPuzzle/snapshots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VisionPuzzle.app import VisionPuzzleApp


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision Hand Puzzle")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to an image to use as the puzzle source instead of framing the camera. "
             "You can still switch sources anytime with the U key.",
    )
    args = parser.parse_args()
    return VisionPuzzleApp(camera_index=args.camera, initial_image=args.image).run()


if __name__ == "__main__":
    raise SystemExit(main())
