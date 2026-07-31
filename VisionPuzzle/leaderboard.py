"""VisionPuzzle.leaderboard — solve-time tracking, persisted to disk.

Keeps a small, per-grid-size leaderboard (3x3 / 4x4 / 5x5) of the
fastest solves. Entries are stored as plain JSON next to the package
so they're easy to inspect, back up, or hand-edit if you ever want to
reset them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

BoardKey = Union[int, str]


@dataclass
class LeaderboardEntry:
    time_seconds: float
    grid: BoardKey
    date: str  # "%Y-%m-%d %H:%M", local time


@dataclass
class AddResult:
    """What happened when a freshly-solved time was submitted."""

    entry: LeaderboardEntry
    rank: int          # 1-indexed position on this grid's board; 0 if it didn't make the cut
    is_new_best: bool   # rank == 1
    made_board: bool    # rank > 0 (i.e. within max_entries)


@dataclass
class Leaderboard:
    """Fastest-solve tracker, one ranked list per grid size."""

    path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "leaderboard.json")
    max_entries: int = 10
    _boards: dict[BoardKey, list[LeaderboardEntry]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[leaderboard] could not read {self.path}: {exc}")
            return
        for grid_key, entries in raw.items():
            parsed: list[LeaderboardEntry] = []
            for e in entries:
                try:
                    parsed.append(LeaderboardEntry(**e))
                except Exception:
                    continue
            parsed.sort(key=lambda x: x.time_seconds)
            self._boards[grid_key] = parsed[: self.max_entries]

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                str(grid): [asdict(e) for e in entries]
                for grid, entries in self._boards.items()
            }
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[leaderboard] could not save {self.path}: {exc}")

    # -- reading -----------------------------------------------------------------

    def top(self, grid: BoardKey, n: int = 5) -> list[LeaderboardEntry]:
        """Fastest `n` entries for a board key, ascending by time."""
        return list(self._boards.get(grid, []))[:n]

    def best(self, grid: BoardKey) -> Optional[LeaderboardEntry]:
        entries = self._boards.get(grid)
        return entries[0] if entries else None

    # -- writing -----------------------------------------------------------------

    def add(self, grid: BoardKey, time_seconds: float) -> AddResult:
        """Record a solve and return where it landed on the board."""
        entry = LeaderboardEntry(
            time_seconds=round(float(time_seconds), 2),
            grid=grid,
            date=time.strftime("%Y-%m-%d %H:%M"),
        )
        board = self._boards.setdefault(grid, [])
        board.append(entry)
        board.sort(key=lambda x: x.time_seconds)
        rank = board.index(entry) + 1
        made_board = rank <= self.max_entries
        if len(board) > self.max_entries:
            del board[self.max_entries:]
            if rank > self.max_entries:
                made_board = False
                rank = 0
        self._save()
        return AddResult(entry=entry, rank=rank, is_new_best=(rank == 1), made_board=made_board)

    def reset(self, grid: Optional[BoardKey] = None) -> None:
        """Clear one grid's board, or every board if `grid` is None."""
        if grid is None:
            self._boards.clear()
        else:
            self._boards.pop(grid, None)
        self._save()


def format_time(seconds: float) -> str:
    """mm:ss.d — compact solve-time formatting, e.g. '01:04.7'."""
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:04.1f}"
