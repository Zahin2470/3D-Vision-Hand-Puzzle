"""VisionPuzzle.savegame — save and resume an in-progress puzzle.

Persists everything needed to reconstruct a `JigsawPuzzle` exactly
where you left it: the cropped source image, board geometry, grid
size, every piece's current position/placement, and the elapsed play
time — so the solve-time clock (and leaderboard entry) stays honest
after a resume instead of restarting from zero.

Two files live side by side in the data directory:
    savegame.json          board geometry, piece states, elapsed time
    savegame_source.png    the cropped image the puzzle was built from

Both are written together and both are required to resume; if either
is missing or unreadable, `load()` simply returns None and the app
falls back to a fresh Selection screen.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2


@dataclass
class SaveGame:
    """Reads/writes a single in-progress puzzle slot on disk."""

    data_dir: Path
    meta_path: Path = field(init=False, repr=False)
    image_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.meta_path = self.data_dir / "savegame.json"
        self.image_path = self.data_dir / "savegame_source.png"

    def exists(self) -> bool:
        return self.meta_path.is_file() and self.image_path.is_file()

    # -- writing -----------------------------------------------------------------

    def save(self, puzzle, *, elapsed_seconds: float) -> bool:
        """Snapshot a live JigsawPuzzle. Safe to call often — cheap JSON,
        and the source image only changes size once per puzzle."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.image_path), puzzle.source):
                return False
            pieces = [
                {
                    "row": p.row, "col": p.col,
                    "x": p.x, "y": p.y,
                    "placed": p.placed,
                    "rotation": p.rotation,
                }
                for p in puzzle.pieces
            ]
            payload = {
                "rows": puzzle.rows,
                "cols": puzzle.cols,
                "board_x": puzzle.board_x,
                "board_y": puzzle.board_y,
                "board_w": puzzle.board_w,
                "board_h": puzzle.board_h,
                "allow_rotation": puzzle.allow_rotation,
                "elapsed_seconds": round(max(0.0, elapsed_seconds), 2),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pieces": pieces,
            }
            self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return True
        except Exception as exc:
            print(f"[savegame] could not save: {exc}")
            return False

    # -- reading -----------------------------------------------------------------

    def load(self):
        """Rebuild the puzzle from disk.

        Returns (JigsawPuzzle, elapsed_seconds) on success, None if
        there's nothing to resume or the save is unreadable.
        """
        if not self.exists():
            return None
        from VisionPuzzle.jigsaw import JigsawPuzzle  # local import avoids a module cycle

        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            source = cv2.imread(str(self.image_path))
            if source is None:
                return None

            puzzle = JigsawPuzzle.from_image(
                source,
                int(meta["rows"]), int(meta["cols"]),
                int(meta["board_x"]), int(meta["board_y"]),
                int(meta["board_w"]), int(meta["board_h"]),
                allow_rotation=bool(meta.get("allow_rotation", False)),
            )
            # from_image() shuffles by default — overwrite with the saved
            # positions so the board looks exactly like it did on save.
            by_pos = {(pc["row"], pc["col"]): pc for pc in meta.get("pieces", [])}
            for piece in puzzle.pieces:
                pc = by_pos.get((piece.row, piece.col))
                if pc is None:
                    continue
                piece.x = float(pc["x"])
                piece.y = float(pc["y"])
                piece.draw_x = piece.x
                piece.draw_y = piece.y
                piece.placed = bool(pc["placed"])
                piece.rotation = int(pc.get("rotation", 0))
                piece.holder = None
                piece.lift = 0.0
            puzzle.completed = bool(puzzle.pieces) and all(p.placed for p in puzzle.pieces)
            elapsed = float(meta.get("elapsed_seconds", 0.0))
            return puzzle, elapsed
        except Exception as exc:
            print(f"[savegame] could not load: {exc}")
            return None

    def peek_grid(self) -> Optional[int]:
        """Grid size of the pending save, without doing a full load."""
        if not self.meta_path.is_file():
            return None
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return int(meta.get("rows")) or None
        except Exception:
            return None

    # -- cleanup -----------------------------------------------------------------

    def clear(self) -> None:
        for path in (self.meta_path, self.image_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
