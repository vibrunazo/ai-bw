"""The strategy selector (ARCHITECTURE.md §6.8; docs/CONTROL.md step 7): which build style to play.

A symbolic decision learned by counting outcomes, the per-game choice the architecture gives the
learning where single decisions drown in chaos (a pair of branches of the same moment differs by sd
848 net value; a build style is worth hundreds over a game). Options: the build styles the macro
model knows (taxonomy clusters, gary/policy/macro.py's style input: 1 standard bio, 2 Factory and
vultures, 3 Factory and Starport, 4 two-Barracks marines) and none (the model unsteered). Context:
the map. The table: each option's games won, lost and still going at the time limit, per map
(TABLE, from eval/vs_rush.py's games: eval/selector_table.py).

Each option's score is a Beta posterior over (won + half the undecided) / games. In play the best
expected score; in training a draw from each (Thompson sampling), so options still in doubt keep
being tried. A map with few games borrows from all maps.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

OPTIONS = [None, 1, 2, 3, 4]
BORROW = 20                          # a map's cell with fewer games than this leans on all maps'


def table_path() -> Path:
    root = Path(os.environ.get("GARY_DATA", Path(__file__).resolve().parents[2] / "data"))
    return root / "interim" / "selector" / "tvz_styles.json"


def map_key(mapinfo) -> str:
    """The same map in any game or replay: its size and start locations (as gary/terrain.py's cache)."""
    return hashlib.sha1(json.dumps([list(mapinfo.size), sorted(map(list, mapinfo.starts))]).encode()).hexdigest()[:16]


def _name(option) -> str:
    return "none" if option is None else str(option)


class StrategySelector:
    def __init__(self, table: dict | None = None):
        self.table = table if table is not None else (json.loads(table_path().read_text(encoding="utf-8"))
                                                      if table_path().exists() else {})

    def counts(self, key: str, option) -> tuple[float, float]:
        """(good, bad): won + half the undecided, lost + half the undecided; this map's, or all maps'
        scaled down to BORROW games where this map has few."""
        def cell(c):
            w, l, t = c.get("won", 0), c.get("lost", 0), c.get("time", 0)
            return w + t / 2, l + t / 2
        here = cell(self.table.get(key, {}).get(_name(option), {}))
        if sum(here) >= BORROW:
            return here
        allg = [cell(m.get(_name(option), {})) for m in self.table.values()]
        g, b = sum(a for a, _ in allg), sum(b for _, b in allg)
        n = g + b
        k = max(0.0, BORROW - sum(here)) / n if n else 0.0
        return here[0] + g * k, here[1] + b * k

    def choose(self, key: str, explore: bool = False, rng: random.Random | None = None):
        """The option: the best expected score, or (explore) the best of a draw from each posterior."""
        rng = rng or random.Random()
        best, best_v = None, -1.0
        for o in OPTIONS:
            g, b = self.counts(key, o)
            v = rng.betavariate(g + 1, b + 1) if explore else (g + 1) / (g + b + 2)
            if v > best_v:
                best, best_v = o, v
        return best

    def add(self, key: str, option, result: str) -> None:
        """One game's outcome: "won", "lost" or "time"."""
        c = self.table.setdefault(key, {}).setdefault(_name(option), {})
        c[result] = c.get(result, 0) + 1

    def save(self) -> None:
        table_path().parent.mkdir(parents=True, exist_ok=True)
        table_path().write_text(json.dumps(self.table, indent=1), encoding="utf-8")
