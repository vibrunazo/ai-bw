"""Map knowledge: what a player knows about a map before the game starts.

Players know the maps they play: where the start locations, bases and naturals are. That's
fair information, so Gary gets it up front (unlike enemy units, which are fogged). Computed at
frame 0 from the map itself.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from gary import commands as C
from gary.env import Game, map_chk

GEYSER = 188
GEYSERS = {GEYSER, 110, 149, 157}    # a geyser, or the Refinery / Extractor / Assimilator on it


@dataclass
class Base:
    tile: tuple[int, int]        # top-left tile of the resource depot
    center: tuple[int, int]      # pixel center of the depot
    minerals: list[tuple[int, int]]
    geysers: list[tuple[int, int]]
    start: bool = False          # a start location


@dataclass
class MapInfo:
    size: tuple[int, int]        # pixels
    bases: list[Base]
    starts: list[tuple[int, int]]
    hash: str | None = None      # the map data's SHA-1, as replays' MapDataHash (ingest/inventory.py), when the
                                 # map file is known (a headless game; not the live client): the macro model's map

    @classmethod
    def from_game(cls, game: Game) -> "MapInfo":
        obs = game.observe()
        size = (obs["map"]["w"], obs["map"]["h"])
        resources = [u for u in obs["units"] if u["type"] in C.MINERAL_FIELDS or u["type"] in GEYSERS]
        bases = [b for b in (_base_for(game, cl) for cl in _clusters(resources)) if b]
        # player start locations have a full base (maps may add observer slots, e.g. by small
        # mineral clusters in the middle): a mineral line and a geyser within 10 tiles
        starts = [(s["x"], s["y"]) for s in game.start_locations()
                  if any(len(b.minerals) >= 6 and b.geysers and math.dist(b.center, (s["x"], s["y"])) < 320 for b in bases)]
        for b in bases:
            b.start = any(math.dist(b.center, s) < 64 for s in starts)
        # start locations are exact: use them as the depot spot of their base
        for s in starts:
            near = min(bases, key=lambda b: math.dist(b.center, s), default=None)
            if near and math.dist(near.center, s) < 320:
                near.center, near.tile, near.start = s, ((s[0] - 64) // 32, (s[1] - 48) // 32), True
        path = getattr(game, "map_path", None)
        return cls(size, bases, starts, hashlib.sha1(map_chk(path)).hexdigest() if path else None)

    def base_near(self, xy: tuple[int, int]) -> Base:
        return min(self.bases, key=lambda b: math.dist(b.center, xy))

    def natural_of(self, main: Base) -> Base:
        """The closest other base (straight line; good enough for standard maps).

        Prefers a non-start base (the natural). On maps whose only bases are start
        locations (small four-corner maps), fall back to the closest other base.
        """
        others = [b for b in self.bases if b is not main and not b.start]
        if not others:
            others = [b for b in self.bases if b is not main]
        if not others:
            raise ValueError("map has no second base to expand to")
        return min(others, key=lambda b: math.dist(b.center, main.center))

    def enemy_starts(self, main: Base) -> list[tuple[int, int]]:
        return [s for s in self.starts if math.dist(s, main.center) > 320]


def _clusters(resources: list[dict]) -> list[list[dict]]:
    """Resources within ~8 tiles of each other belong to one base."""
    clusters: list[list[dict]] = []
    for r in resources:
        joined = [c for c in clusters if any(math.dist((r["x"], r["y"]), (o["x"], o["y"])) < 256 for o in c)]
        merged = [r] + [o for c in joined for o in c]
        clusters = [c for c in clusters if c not in joined] + [merged]
    return [c for c in clusters if sum(1 for u in c if u["type"] in C.MINERAL_FIELDS) >= 4]


def _base_for(game: Game, cluster: list[dict]) -> Base | None:
    """The depot spot closest to its resources (the usual base layout)."""
    cx = sum(u["x"] for u in cluster) / len(cluster)
    cy = sum(u["y"] for u in cluster) / len(cluster)
    best, best_score = None, math.inf
    for ty in range(int(cy // 32) - 12, int(cy // 32) + 12):
        for tx in range(int(cx // 32) - 12, int(cx // 32) + 12):
            if not game.depot_spot_ok(tx, ty):
                continue
            center = (tx * 32 + 64, ty * 32 + 48)
            score = sum(math.dist(center, (u["x"], u["y"])) for u in cluster)
            if score < best_score:
                best, best_score = (tx, ty), score
    if not best:
        return None
    return Base(best, (best[0] * 32 + 64, best[1] * 32 + 48),
                [(u["x"], u["y"]) for u in cluster if u["type"] in C.MINERAL_FIELDS],
                [(u["x"], u["y"]) for u in cluster if u["type"] in GEYSERS])
