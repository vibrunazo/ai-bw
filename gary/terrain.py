"""Map analysis: each start location's main plateau, main ramp, natural and natural choke.

Map knowledge a player has before the game starts (like gary/mapinfo.py), from the terrain
(walkability and ground height per 8x8-pixel minitile: Game.terrain()):

  main plateau    the walkable ground at the start location's height, connected to it
  main ramp       the narrowest stretch of the ground route from the main toward the enemy, near the
                  start: its top (where the route first narrows to it), a spot just inside the main
                  behind it, its bottom (just past it) and its width
  natural         the base nearest the ramp's bottom by ground distance
  natural choke   the narrowest point on the ground route from the natural toward the enemy,
                  between 8 and 20 tiles from it: where a wall or a bunker holds the natural

Routes run through the middle of passages (each step costs more the nearer it is to unwalkable
ground), so the narrowest point along one is the passage's real width, not a wall it brushes.
A main whose way out has no narrow point (wider than RAMP_MAX_WIDTH) near the start is marked
unreliable ("reliable": false).
Widths are the narrowest walkable span there (twice the distance to the nearest unwalkable
minitile). All positions are map pixels. Results are cached per map (data/interim/maps/).

    python -m gary.terrain --map path/to/map.scx     # analysis + a picture to check it
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from gary.env import Game
from gary.mapinfo import MapInfo

MINI = 8                          # pixels per minitile
RAMP_PAST = 4                     # minitiles down the route past the ramp's narrow stretch: its bottom
RAMP_IN = 10                      # minitiles up the route from the ramp's top: the spot inside it...
RAMP_SEARCH = 20                  # ...or up to this many more, to a tile flagged at the main's height
RAMP_MAX_WIDTH = 6 * 32           # a narrowest point wider than this isn't a ramp (pixels)
NATURAL_AREA = 32                 # minitiles from the natural's center that are its own area
CHOKE_SEARCH = 200                # minitiles along the route past that, where its choke is looked for
CHOKE_WITHIN = 20 * 32            # ...and no farther than this from the natural (pixels): its own exit
RAMP_WITHIN = 30 * 32             # a main ramp farther than this from the start is a wrong guess
ANALYSIS_VERSION = 4              # part of the cache key: a change to the analysis makes new files
WALL_COST = 12.0                  # a step's cost: length * (1 + WALL_COST / distance to the nearest wall)


def _cell(p) -> tuple[int, int]:
    return int(p[1]) // MINI, int(p[0]) // MINI


def _px(c) -> tuple[int, int]:
    return int(c[1]) * MINI + MINI // 2, int(c[0]) * MINI + MINI // 2


def _ground_distance(walk: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    """Steps (8-connected, diagonal as 1.4) from the nearest source over walkable minitiles; inf if
    unreachable."""
    h, w = walk.shape
    dist = np.full(walk.shape, np.inf)
    q = collections.deque()
    for s in sources:
        if walk[s]:
            dist[s] = 0.0
            q.append(s)
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.4), (-1, 1, 1.4), (1, -1, 1.4), (1, 1, 1.4)]
    while q:                                     # (approximate: a queue, not a heap; good enough here)
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx, c in steps:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and walk[ny, nx] and d + c < dist[ny, nx]:
                dist[ny, nx] = d + c
                q.append((ny, nx))
    return dist


class CenteredRoutes:
    """Shortest routes over walkable minitiles that keep to the middle of passages (scipy's
    Dijkstra on an 8-connected grid graph)."""

    def __init__(self, walk: np.ndarray, altitude: np.ndarray):
        from scipy.sparse import coo_matrix
        self.shape = walk.shape
        h, w = walk.shape
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, costs = [], [], []
        for dy, dx, length in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4), (1, -1, 1.4)):
            a = walk[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            b = walk[max(0, dy):h - max(0, -dy) or None, max(0, dx):w - max(0, -dx) or None]
            ok = a & b
            ia = idx[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)][ok]
            ib = idx[max(0, dy):h - max(0, -dy) or None, max(0, dx):w - max(0, -dx) or None][ok]
            alt = (altitude.ravel()[ia] + altitude.ravel()[ib]) / 2
            c = length * (1 + WALL_COST / np.maximum(alt, 0.5))
            rows += [ia, ib]
            cols += [ib, ia]
            costs += [c, c]
        n = h * w
        self.graph = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

    def to(self, sources: list[tuple[int, int]]):
        """Distances to the nearest source and each cell's next step toward it."""
        from scipy.sparse.csgraph import dijkstra
        w = self.shape[1]
        dist, pred, _ = dijkstra(self.graph, indices=[y * w + x for y, x in sources], min_only=True,
                                 return_predecessors=True)
        return dist.reshape(self.shape), pred

    def route(self, pred, start: tuple[int, int], max_steps: int) -> list[tuple[int, int]]:
        w = self.shape[1]
        path, cur = [start], start[0] * w + start[1]
        for _ in range(max_steps):
            nxt = pred[cur]
            if nxt < 0:
                break
            cur = int(nxt)
            path.append((cur // w, cur % w))
        return path


def _route(dist: np.ndarray, start: tuple[int, int], max_steps: int) -> list[tuple[int, int]]:
    """Downhill on a distance field from start (toward its sources)."""
    path, cur = [start], start
    h, w = dist.shape
    for _ in range(max_steps):
        y, x = cur
        best = min(((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w), key=lambda c: dist[c])
        if dist[best] >= dist[cur]:
            break
        path.append(best)
        cur = best
    return path


def analyze(game: Game, mapinfo: MapInfo) -> dict:
    walk, height, _ = game.terrain()
    walk = ndimage.binary_opening(walk)          # one-minitile slivers aren't paths
    altitude = ndimage.distance_transform_edt(walk)
    starts = {i: _cell(s) for i, s in enumerate(mapinfo.starts)}
    routes = CenteredRoutes(walk, altitude)
    out = {}
    for i, s in starts.items():
        # the start location's cell may sit under the depot's unwalkable footprint edge: nearest walkable
        if not walk[s]:
            ys, xs = np.nonzero(walk)
            k = np.argmin((ys - s[0]) ** 2 + (xs - s[1]) ** 2)
            s = (int(ys[k]), int(xs[k]))
        others = [_nearest_walkable(walk, o) for j, o in starts.items() if j != i]
        _, pred = routes.to(others)
        level = height[s]
        labels, _ = ndimage.label(walk & (height == level))
        main = labels == labels[s]
        # the main's ramp: the narrowest stretch of the centered route toward the enemy near the start,
        # by walkability alone (ramp tiles' height flags don't follow the slope: the upper part is
        # flagged high ground, and on old maps made with tile editors, e.g. Python, high and low tiles
        # can sit anywhere on it). Its top: where the route first narrows to that width
        out_route = routes.route(pred, s, 4000)
        n = 1 + max((j for j, c in enumerate(out_route) if math.dist(_px(c), mapinfo.starts[i]) <= RAMP_WITHIN), default=0)
        narrowest = min(altitude[c] for c in out_route[:n])
        ramp_width = 2 * MINI * float(narrowest)
        t = next(j for j in range(n) if altitude[out_route[j]] <= narrowest + 1)
        e = next((j for j in range(t, len(out_route)) if altitude[out_route[j]] > narrowest + 2), len(out_route) - 1)
        top = out_route[t]
        bottom = out_route[min(len(out_route) - 1, e + RAMP_PAST)]
        # a hold spot inside, behind the top, on a tile flagged at the main's height (the flags, not
        # the tiles' looks, decide vision: zerg below don't see up there)
        j = max(0, t - RAMP_IN)
        while j > 0 and height[out_route[j]] != level and t - j < RAMP_IN + RAMP_SEARCH:
            j -= 1
        inside = out_route[j]
        # the natural: the non-start base nearest the ramp's bottom by ground
        from_ramp = _ground_distance(walk, [bottom])
        main_base = mapinfo.base_near(mapinfo.starts[i])
        candidates = [b for b in mapinfo.bases if b is not main_base and not b.start] or \
                     [b for b in mapinfo.bases if b is not main_base]
        natural = min(candidates, key=lambda b: from_ramp[_nearest_walkable(walk, _cell(b.center))])
        n_cell = _nearest_walkable(walk, _cell(natural.center))
        route = routes.route(pred, n_cell, NATURAL_AREA + CHOKE_SEARCH)
        ring = [c for c in route if NATURAL_AREA * MINI <= math.dist(_px(c), natural.center) <= CHOKE_WITHIN]
        choke = min(ring or route[NATURAL_AREA:] or route, key=lambda c: altitude[c])
        # trust it where the ramp is a real choke (narrow) near the start
        reliable = ramp_width <= RAMP_MAX_WIDTH and math.dist(_px(top), mapinfo.starts[i]) <= RAMP_WITHIN
        # the other bases by how far a worker walks from the natural (the next one to take: a base
        # beside the main as the crow flies can be a long walk past the natural and through the field)
        from_nat = _ground_distance(walk, [n_cell])
        others = [b for b in mapinfo.bases if b is not main_base and b is not natural]
        walk_to = {tuple(b.center): float(from_nat[_nearest_walkable(walk, _cell(b.center))]) for b in others}
        out[i] = {"start": list(mapinfo.starts[i]), "reliable": bool(reliable),
                  "main_level": int(level), "main_area_tiles": int(main.sum()) // 16,
                  "ramp_top": list(_px(top)), "ramp_bottom": list(_px(bottom)), "ramp_width": round(ramp_width),
                  "ramp_inside": list(_px(inside)),
                  "natural": list(natural.center), "natural_choke": list(_px(choke)),
                  "natural_choke_width": round(2 * MINI * float(altitude[choke])),
                  "bases_by_walk": [list(c) for c in sorted(walk_to, key=walk_to.get) if math.isfinite(walk_to[c])]}
    return {"map": {"w": int(walk.shape[1]) * MINI, "h": int(walk.shape[0]) * MINI}, "starts": out}


def _nearest_walkable(walk: np.ndarray, c: tuple[int, int]) -> tuple[int, int]:
    if walk[c]:
        return c
    ys, xs = np.nonzero(walk)
    k = np.argmin((ys - c[0]) ** 2 + (xs - c[1]) ** 2)
    return int(ys[k]), int(xs[k])


def draw(game: Game, mapinfo: MapInfo, analysis: dict, path: str | Path) -> None:
    """A picture of the analysis to check it: terrain by height, the main plateau tinted, ramps
    (red: top, pink: bottom), naturals (yellow) and their chokes (orange), bases (white)."""
    from PIL import Image, ImageDraw
    walk, height, _ = game.terrain()
    img = np.zeros(walk.shape + (3,), np.uint8)
    shade = {0: 90, 1: 140, 2: 190}
    for lv, v in shade.items():
        img[walk & (height == lv)] = v
    pic = Image.fromarray(img).resize((walk.shape[1] * 2, walk.shape[0] * 2), Image.NEAREST)
    d = ImageDraw.Draw(pic)
    sc = 2 / MINI
    dot = lambda p, r, c: d.ellipse([p[0] * sc - r, p[1] * sc - r, p[0] * sc + r, p[1] * sc + r], fill=c)
    for b in mapinfo.bases:
        dot(b.center, 4, (255, 255, 255))
    for a in analysis["starts"].values():
        dot(a["start"], 7, (60, 120, 255))
        dot(a["ramp_top"], 6, (255, 40, 40))
        dot(a["ramp_bottom"], 4, (255, 140, 160))
        dot(a["natural"], 6, (255, 230, 0))
        dot(a["natural_choke"], 6, (255, 140, 0))
        d.line([a["ramp_top"][0] * sc, a["ramp_top"][1] * sc, a["ramp_bottom"][0] * sc, a["ramp_bottom"][1] * sc],
               fill=(255, 40, 40), width=2)
    pic.save(path)


def map_analysis(game: Game, mapinfo: MapInfo) -> dict:
    """The analysis for a map, cached under data/interim/maps/ by the map's size and start locations
    (the same map in any replay or game gets the same file)."""
    import hashlib
    import os
    root = Path(os.environ.get("GARY_DATA", Path(__file__).resolve().parent.parent / "data"))
    key = hashlib.sha1(json.dumps([ANALYSIS_VERSION, mapinfo.size, sorted(mapinfo.starts)]).encode()).hexdigest()[:16]
    cache = root / "interim" / "maps" / f"{mapinfo.size[0] // 32}x{mapinfo.size[1] // 32}_{key}.json"
    if cache.exists():
        a = json.loads(cache.read_text(encoding="utf-8"))
        a["starts"] = {int(k): v for k, v in a["starts"].items()}
        return a
    a = analyze(game, mapinfo)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(a, indent=1), encoding="utf-8")
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="a .scm/.scx map or a pre-1.18 replay")
    ap.add_argument("--png", help="where to save the picture (default: next to the cache)")
    args = ap.parse_args()
    game = Game.new(args.map, ["T", "Z"], seed=0)
    mapinfo = MapInfo.from_game(game)
    import time
    t0 = time.time()
    a = analyze(game, mapinfo)
    print(f"analyzed in {time.time() - t0:.1f} s")
    for i, s in a["starts"].items():
        print(i, s)
    png = args.png or "map_analysis.png"
    draw(game, mapinfo, a, png)
    print("->", png)


if __name__ == "__main__":
    main()
