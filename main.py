#!/usr/bin/env python3
"""VisionPuzzle Studio — premium dual-hand camera jigsaw.

Run
---
    pip install -r requirements.txt
    python main.py

Select
------
Show both hands
Pinch with BOTH hands          Frame opposite corners
Stretch L + R index tips       Resize the crop
Release both                   Lock selection
SPACE / Enter                  Create jigsaw
3 / 4 / 5                      Grid size
C                              Clear

Play
----
Pinch a piece (either hand)    Grab
Both hands at once             Move two pieces
Release near slot              Snap
R                              Reshuffle
N                              New capture

H                              Help
Q / Esc                        Quit
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VisionPuzzle.app import VisionPuzzleApp


def main() -> int:
    return VisionPuzzleApp(camera_index=0).run()


if __name__ == "__main__":
    raise SystemExit(main())
