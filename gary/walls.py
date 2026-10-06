"""Walls: which units get between two buildings (sim city), and where a base's openings are.

A building takes whole tiles to place, but what units collide with is its collision box from the
game data (units.dat), smaller than its tiles by a different number of pixels on each side. So the
opening between two buildings is the empty tiles between them plus the two facing margins, and a
unit gets through if its own collision box fits: a Supply Depot on top of another leaves 15 pixels
(5 under the upper one, 10 over the lower one) and stops zerglings (16 wide); side by side they
leave 19 and let zerglings and marines through. Pros build their walls from this, and from what
buildings cost and whether they need them anyway (a Terran walls with Supply Depots, Barracks and
an Academy; a Zerg with Evolution Chambers).

    python -m gary.walls --table                 # the margins, and every pair's openings
    python -m gary.walls --check                 # the arithmetic against the engine: buildings
                                                 # placed side by side in an OpenBW game, units
                                                 # sent through the opening
    python -m gary.walls --frontier 0c50ac60     # a home defense's Terran frontier at the takeover

The frontier (frontier()): what separates all of a player's bases (every Command Center is inside)
from the enemy's side, whatever the map: where the natural is the only way into the main, the
natural's choke; where the main has another way in, that too; where the natural is behind the
main, the main's way out. A minimum cut over the 8-pixel minitiles within 24 tiles (by ground) of
the bases: the fewest cells to close so that no ground way leads out to what still connects to the
enemy's start locations (max-flow; obstacles are unwalkable minitiles and the collision boxes of
buildings, resources and neutral map units). Its pieces are then looked at exactly (gaps(), from
right inside to right outside): zergling-tight ones are closed parts of a wall, the rest are the
openings. In the pros' 239 home defenses: bases with only the main, the ramp alone (61 of 123),
another single way out (20) or closed to zerglings altogether (13); with the natural taken, one
opening (57 of 116: the natural's choke, or the way out where the map analysis didn't find it),
two (25), closed altogether in only 2.

The gap finder (gaps(): a window around two points, e.g. inside the natural and past its choke)
works at pixel resolution: obstacles are unwalkable minitiles and the collision boxes of
buildings, mineral fields and geysers; where a unit can stand is where its box clears them all,
so whether a zergling, a marine or an SCV gets from outside in is exact. The openings are found
narrowest first: the widest way in, its narrowest place (by clearance), then that place closed
and the next way in. In the pros' 239 home defenses (at the takeover): no main ramp is walled
or narrowed by buildings (median 56 pixels of clearance, as bare); the natural's choke is
narrowed from a median 176 pixels bare to 74 by the buildings (to half or less in 95 of 114
bases that took it), but closed to zerglings in only 5. Their marines on hold position stand at
such an opening 23% of the time (open ground 34%, by a building 16%, by a bunker 13%, between
buildings 11%, the mineral line 3%).

Everything is read from the installed game's data at run time (Game.unit_type_info). Checked
against the engine (--check; 5 Terran buildings in every pairing and way they touch, 7 units):
where the two collision boxes face each other by 16+ pixels, "opening >= the unit's box" agrees
with the engine in 1,144 of 1,148 cases, to the pixel (an opening exactly the unit's size lets it
through, one pixel less doesn't). The exception: an Academy above a Supply Depot leaves 17 pixels
and still stops zerglings (16 tall). Boxes that barely face each other (corners) are diagonal
openings, where units squeeze past less predictably: there the engine decides.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

TERRAN_BUILDINGS = {106: "Command Center", 109: "Supply Depot", 111: "Barracks", 112: "Academy", 113: "Factory",
                    114: "Starport", 116: "Science Facility", 122: "Engineering Bay", 123: "Armory",
                    124: "Missile Turret", 125: "Bunker"}
TIGHTER_BELOW = {(112, 109)}         # (a, b): b below a stops zerglings though the arithmetic says it doesn't
WALKERS = {37: "Zergling", 0: "Marine", 34: "Medic", 32: "Firebat", 7: "SCV", 41: "Drone", 38: "Hydralisk",
           2: "Vulture", 39: "Ultralisk"}


@dataclass(frozen=True)
class Footprint:
    """A unit type's size: placement tiles and its collision box (pixels from its position)."""
    w: int                   # placement size, pixels (whole tiles for buildings)
    h: int
    left: int                # collision box: x - left .. x + right, y - up .. y + down (inclusive)
    up: int
    right: int
    down: int

    @property
    def box_w(self) -> int:
        return self.left + self.right + 1

    @property
    def box_h(self) -> int:
        return self.up + self.down + 1

    def margins(self) -> tuple[int, int, int, int]:
        """Pixels between the collision box and the edge of its tiles: left, top, right, bottom."""
        return (self.w // 2 - self.left, self.h // 2 - self.up,
                self.w - self.w // 2 - self.right - 1, self.h - self.h // 2 - self.down - 1)


class Footprints:
    def __init__(self, game):
        self.game = game
        self._by_type: dict[int, Footprint] = {}

    def __getitem__(self, unit_type: int) -> Footprint:
        if unit_type not in self._by_type:
            i = self.game.unit_type_info(unit_type)
            self._by_type[unit_type] = Footprint(i["w"], i["h"], i["left"], i["up"], i["right"], i["down"])
        return self._by_type[unit_type]

    def box_at_tile(self, unit_type: int, tile_x: int, tile_y: int) -> tuple[int, int, int, int]:
        """A building's collision box (x0, y0, x1, y1, inclusive pixels) with its top-left tile here."""
        f = self[unit_type]
        cx, cy = tile_x * 32 + f.w // 2, tile_y * 32 + f.h // 2
        return cx - f.left, cy - f.up, cx + f.right, cy + f.down

    def box_at(self, unit_type: int, x: int, y: int) -> tuple[int, int, int, int]:
        """A unit's collision box at its position (x, y)."""
        f = self[unit_type]
        return x - f.left, y - f.up, x + f.right, y + f.down


def opening(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    """Free pixels between two boxes along x and along y (<= 0: they overlap along that axis)."""
    gx = max(b[0] - a[2], a[0] - b[2]) - 1
    gy = max(b[1] - a[3], a[1] - b[3]) - 1
    return gx, gy


def passes(unit: Footprint, gx: int, gy: int) -> bool:
    """Whether a unit fits between two boxes this far apart (opening): through a gap between side
    by side boxes it needs its width, between stacked ones its height, between diagonal ones either."""
    return gx >= unit.box_w or gy >= unit.box_h


def neighbors(fp: Footprints, a: int, b: int):
    """Every way b touches a (no empty tile between them) with their collision boxes facing each
    other, as (side, b's tile offset from a's top-left, opening in pixels): b right of a, and b
    below a. (Tiles touching only at a corner, or boxes that don't face: a diagonal opening.)"""
    fa, fb = fp[a], fp[b]
    aw, ah, bw, bh = fa.w // 32, fa.h // 32, fb.w // 32, fb.h // 32
    for dy in range(-bh + 1, ah):
        gx, gy = opening(fp.box_at_tile(a, 0, 0), fp.box_at_tile(b, aw, dy))
        if gy < 0:
            yield "right", (aw, dy), gx
    for dx in range(-bw + 1, aw):
        gx, gy = opening(fp.box_at_tile(a, 0, 0), fp.box_at_tile(b, dx, ah))
        if gx < 0:
            yield "below", (dx, ah), gy


# --- the gap finder: where units get through a base's buildings and cliffs ------------------------

RESOURCES = {176, 177, 178, 188}      # mineral fields, vespene geyser
GAP_UNITS = (37, 0, 7)                # zergling, marine, SCV: who gets through each opening
MAX_OPENINGS = 6


@dataclass
class Opening:
    """A narrowest place on a way from outside in: its middle (map pixels), its width (pixels a
    unit has there, roughly: the clearance across it), and which unit types fit through it."""
    x: int
    y: int
    width: int
    passes: list[int]


def obstacles(game, obs: dict, fp: Footprints, window: tuple[int, int, int, int]):
    """What ground units collide with in a window of the map (x0, y0, x1, y1 pixels): a boolean
    array [y, x], True where unwalkable terrain (8x8-pixel minitiles) or a building's, mineral
    field's or geyser's collision box is."""
    import numpy as np
    from gary.policy.army import is_building
    x0, y0, x1, y1 = window
    walk = getattr(game, "_walk_cache", None)
    if walk is None:
        walk = game._walk_cache = game.terrain()[0]
    mini = walk[y0 // 8:(y1 + 7) // 8, x0 // 8:(x1 + 7) // 8]
    blocked = np.repeat(np.repeat(~mini, 8, axis=0), 8, axis=1)[y0 % 8:y0 % 8 + (y1 - y0), x0 % 8:x0 % 8 + (x1 - x0)]
    for u in obs["units"]:
        if u["type"] in RESOURCES or u["owner"] == 11 or is_building(u["type"]):
            bx0, by0, bx1, by1 = fp.box_at(u["type"], u["x"], u["y"])
            bx0, by0, bx1, by1 = max(bx0, x0), max(by0, y0), min(bx1, x1 - 1), min(by1, y1 - 1)
            if bx0 <= bx1 and by0 <= by1:
                blocked[by0 - y0:by1 - y0 + 1, bx0 - x0:bx1 - x0 + 1] = True
    return blocked


def fits(blocked, unit: Footprint):
    """Where a unit can stand: True at positions [y, x] where its collision box clears every
    obstacle (and stays in the window)."""
    import numpy as np
    h, w = blocked.shape
    s = np.zeros((h + 1, w + 1), np.int32)
    s[1:, 1:] = blocked.cumsum(0).cumsum(1)
    free = np.zeros((h, w), bool)
    ys, xs = slice(unit.up, h - unit.down), slice(unit.left, w - unit.right)
    y0, y1 = np.arange(h)[ys] - unit.up, np.arange(h)[ys] + unit.down + 1
    x0, x1 = np.arange(w)[xs] - unit.left, np.arange(w)[xs] + unit.right + 1
    total = s[y1][:, x1] - s[y0][:, x1] - s[y1][:, x0] + s[y0][:, x0]
    free[ys, xs] = total == 0
    return free


SNAP = 32                             # a point in an obstacle stands for the free cell this near


def _seed(free, p, snap: int = SNAP):
    """The free cell nearest p (window coordinates [y, x]) within snap pixels, or None."""
    import numpy as np
    y0, x0 = max(0, p[0] - snap), max(0, p[1] - snap)
    ys, xs = np.nonzero(free[y0:p[0] + snap + 1, x0:p[1] + snap + 1])
    if not len(ys):
        return None
    k = int(np.argmin((ys + y0 - p[0]) ** 2 + (xs + x0 - p[1]) ** 2))
    return int(ys[k] + y0), int(xs[k] + x0)


def _narrowest(ridge, side_a, side_b):
    """The narrowest place of a way in: the shortest chain of ridge cells (exactly the bottleneck
    clearance: across the gap, but also along every cliff and building edge at that distance)
    linking a cell next to one side's ground to a cell next to the other's, by ground (through
    ridge cells only: a cliff between them doesn't count as near). Across a short gap it's the gap,
    along a corridor its middle line. None if no chain links them."""
    import numpy as np
    from collections import deque
    from scipy import ndimage
    pieces, _ = ndimage.label(ridge, structure=np.ones((3, 3)))
    touch_a, touch_b = ridge & side_a, ridge & side_b
    ids = (set(np.unique(pieces[touch_a])) & set(np.unique(pieces[touch_b]))) - {0}
    best = None
    h, w = ridge.shape
    for k in ids:
        starts = np.argwhere(touch_a & (pieces == k))
        goal = touch_b & (pieces == k)
        prev = {tuple(c): None for c in starts}
        queue = deque(prev)
        end = None
        while queue:
            c = queue.popleft()
            if goal[c]:
                end = c
                break
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (c[0] + dy, c[1] + dx)
                    if 0 <= n[0] < h and 0 <= n[1] < w and n not in prev and pieces[n] == k:
                        prev[n] = c
                        queue.append(n)
        if end is None:
            continue
        path = []
        while end is not None:
            path.append(end)
            end = prev[end]
        if best is None or len(path) < len(best):
            best = path
    if best is None:
        return None
    found = np.zeros_like(ridge)
    for c in best:
        found[c] = True
    return found


def connected(free, a, b) -> bool:
    from scipy import ndimage
    labels, _ = ndimage.label(free)
    sa, sb = _seed(free, a), _seed(free, b)
    return sa is not None and sb is not None and labels[sa] == labels[sb] != 0


def gaps(game, obs: dict, inside: tuple[int, int], outside: tuple[int, int], margin: int = 12 * 32,
         units=GAP_UNITS) -> dict:
    """The openings between outside and inside (map pixels; e.g. the main's Command Center and its
    ramp's bottom), through the buildings and cliffs in between: for each unit type whether it gets
    in at all ("walled": False if it does), and the openings, narrowest first, each with who fits.
    The window is the box around both points plus margin."""
    import numpy as np
    from scipy import ndimage
    fp = Footprints(game)
    if getattr(game, "_walk_cache", None) is None:
        game._walk_cache = game.terrain()[0]
    H, W = game._walk_cache.shape[0] * 8, game._walk_cache.shape[1] * 8
    x0, y0 = max(0, min(inside[0], outside[0]) - margin), max(0, min(inside[1], outside[1]) - margin)
    x1, y1 = min(W, max(inside[0], outside[0]) + margin), min(H, max(inside[1], outside[1]) + margin)
    blocked = obstacles(game, obs, fp, (x0, y0, x1, y1))
    a, b = (inside[1] - y0, inside[0] - x0), (outside[1] - y0, outside[0] - x0)
    # points on a building (the natural's Command Center...): the open ground nearest them, once
    a, b = _seed(fits(blocked, fp[37]), a, 160) or a, _seed(fits(blocked, fp[37]), b, 160) or b
    free = {t: fits(blocked, fp[t]) for t in units}
    walled = {t: not connected(free[t], a, b) for t in units}
    openings = []
    clear = ndimage.distance_transform_cdt(~blocked, metric="chessboard")
    work = clear.copy()
    for _ in range(MAX_OPENINGS):
        # the widest way in: the largest clearance t with inside and outside connected over >= t
        lo, hi = 0, int(work.max())
        while lo < hi:
            t = (lo + hi + 1) // 2
            if connected(work >= t, a, b):
                lo = t
            else:
                hi = t - 1
        t = lo
        if t < 1:
            break
        # its narrowest place: every way in at t crosses cells of clearance exactly t (above t the
        # two sides aren't connected); the piece of those touching both sides' open ground (each
        # side: the ground above t around its point, or the point's own neighborhood)
        reach, _ = ndimage.label(work >= t)
        if reach[_seed(work >= t, a)] != reach[_seed(work >= t, b)]:
            break
        ridge = (reach == reach[_seed(work >= t, a)]) & (work == t)
        rest, _ = ndimage.label((reach == reach[_seed(work >= t, a)]) & (work > t))

        def side(p):
            near = np.zeros_like(ridge)
            near[max(0, p[0] - 8):p[0] + 9, max(0, p[1] - 8):p[1] + 9] = True
            ids = set(np.unique(rest[near])) - {0}
            return ndimage.binary_dilation(np.isin(rest, list(ids)) | (near & ridge), iterations=1)
        side_a, side_b = side(_seed(work >= t, a)), side(_seed(work >= t, b))
        found = _narrowest(ridge, side_a, side_b)
        if found is None:
            break
        ys, xs = np.nonzero(found)
        cy, cx = int(ys.mean()), int(xs.mean())
        across = ndimage.binary_dilation(found, iterations=2)          # (asymmetric boxes: a pixel or two off)
        openings.append(Opening(cx + x0, cy + y0, 2 * t, [u for u in units if (free[u] & across).any()]))
        if not any(openings[-1].passes):
            break
        work[ndimage.binary_dilation(found, iterations=t + 3)] = 0     # closed: the next way in
    return {"walled": walled, "openings": openings, "window": (x0, y0, x1, y1)}


# --- the frontier: what separates all of a player's bases from the enemy's side ------------------

_source_side = None

FRONTIER_REACH = 24 * 4              # minitiles: the ground this near a base (by ground) is the base's side
FRONTIER_WIDEST = 12 * 4             # minitiles: a way in wider than this at its narrowest isn't a choke
FRONTIER_MAX = 4                     # openings at most


@dataclass
class FrontierOpening:
    """One way in through the frontier: its narrowest place (map pixels), its width there (coarse:
    per 8-pixel minitile), and the exact look from right inside to right outside of it (gaps())."""
    x: int
    y: int
    width: int
    inside: tuple[int, int]
    outside: tuple[int, int]
    exact: dict

    @property
    def closed(self) -> bool:
        """Zerglings don't get through (exact): a closed part of the wall, not a way in."""
        return bool(self.exact.get("walled", {}).get(37))


def frontier(game, obs: dict, bases: list[tuple[int, int]], enemy: list[tuple[int, int]],
             units=GAP_UNITS, reach: int = FRONTIER_REACH) -> dict:
    """The frontier of a player's bases (Command Center positions; all of them are inside) against
    the enemy's side (e.g. the other start locations): the openings that, all closed, cut every
    ground way in, narrowest place first, each with which units get through (exact). Map
    independent: where the natural is the only way to the main, blocking the natural's choke is
    the whole frontier; where the main has another way in, that is an opening too; where the
    natural is behind the main, the main's way out is.

    Coarse over the whole map (8-pixel minitiles: unwalkable, or under a building's or resource's
    collision box): the ground within FRONTIER_REACH of the bases is theirs, the ground beyond it
    that still connects to the enemy's points is outside. Then, as gaps(): the widest way in, its
    narrowest place, closed, the next. "open": True if a way in is wider than FRONTIER_WIDEST at
    its narrowest (no choke to hold)."""
    import numpy as np
    from scipy import ndimage
    from gary.policy.army import is_building
    fp = Footprints(game)
    if getattr(game, "_walk_cache", None) is None:
        game._walk_cache = game.terrain()[0]
    free = game._walk_cache.copy()
    H, W = free.shape
    for u in obs["units"]:
        if u["type"] in RESOURCES or u["owner"] == 11 or is_building(u["type"]):
            x0, y0, x1, y1 = fp.box_at(u["type"], u["x"], u["y"])
            free[max(0, (y0 + 4) // 8):max(0, (y1 - 3) // 8 + 1), max(0, (x0 + 4) // 8):max(0, (x1 - 3) // 8 + 1)] = False

    def cell(p):
        return _seed(free, (min(H - 1, p[1] // 8), min(W - 1, p[0] // 8)), 24)
    clear = ndimage.distance_transform_cdt(free, metric="chessboard")
    # inside: the roomy ground (2+ minitiles from anything) within 4 tiles of each base
    yy, xx = np.mgrid[0:H, 0:W]
    home = np.zeros_like(free)
    for bx, by in bases:
        home |= (yy - by // 8) ** 2 + (xx - bx // 8) ** 2 <= 16 ** 2
    home &= clear >= 2
    if not home.any():
        return {"openings": [], "open": False}
    inside = [tuple(c) for c in np.argwhere(home)]
    near = home.copy()
    eight = np.ones((3, 3), bool)
    for _ in range(reach):                           # the ground within reach (by ground) of a base
        near = ndimage.binary_dilation(near, structure=eight) & free
    far, _ = ndimage.label(free & ~near)
    ids = {far[c] for c in map(cell, enemy) if c and far[c]}
    outside = np.isin(far, list(ids))
    if not outside.any():
        return {"openings": [], "open": False}
    cut, flow = _min_cut(near | home, home, near & ndimage.binary_dilation(outside, structure=eight))
    if cut is None:                                   # a base's own ground touches the outside
        return {"openings": [], "open": True, "cut_width": None}
    pieces, n = ndimage.label(cut, structure=eight)
    side = _source_side & ~cut
    openings = []
    for k in range(1, n + 1):
        piece = pieces == k
        ys, xs = np.nonzero(piece)
        cy, cx = int(ys.mean()), int(xs.mean())
        # right inside and right outside of it: the nearest ground on each side, 2+ minitiles off
        ring = np.zeros_like(free)
        ring[max(0, cy - 24):cy + 25, max(0, cx - 24):cx + 25] = True
        ring &= (near | home) & ~ndimage.binary_dilation(piece, structure=eight, iterations=2) & (clear >= 2)
        pts_in, pts_out = np.argwhere(ring & side), np.argwhere(ring & ~side & ~cut)
        if len(pts_in) and len(pts_out):
            pi = pts_in[np.argmin(((pts_in - (cy, cx)) ** 2).sum(1))]
            po = pts_out[np.argmin(((pts_out - (cy, cx)) ** 2).sum(1))]
            pin, pout = (int(pi[1]) * 8 + 4, int(pi[0]) * 8 + 4), (int(po[1]) * 8 + 4, int(po[0]) * 8 + 4)
            exact = gaps(game, obs, pin, pout, margin=4 * 32, units=units)
        else:
            pin = pout = (cx * 8 + 4, cy * 8 + 4)
            exact = {}
        openings.append(FrontierOpening(cx * 8 + 4, cy * 8 + 4, int(piece.sum()) * 8, pin, pout, exact))
    openings.sort(key=lambda o: o.width)
    return {"openings": openings, "open": flow > FRONTIER_WIDEST, "cut_width": flow * 8}


def _min_cut(region, sources, sinks):
    """The fewest cells of region to close so that no 8-connected way leads from a source cell to a
    sink cell (sources can't be closed): a minimum vertex cut, by max-flow over the cells (each in-
    to-out edge capacity 1). Returns (the cut cells, its size), or (None, None) if a source touches
    a sink. Leaves the source side (cells reachable from the sources without crossing the cut) in
    _source_side."""
    global _source_side
    import numpy as np
    from scipy import sparse
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow
    if (sources & sinks).any():
        return None, None
    h, w = region.shape
    idx = -np.ones(region.shape, np.int64)
    cells = np.argwhere(region)
    n = len(cells)
    idx[cells[:, 0], cells[:, 1]] = np.arange(n)
    INF = 1 << 20
    src, snk = 2 * n, 2 * n + 1
    rows, cols, caps = [], [], []
    inner = np.where(sources[cells[:, 0], cells[:, 1]], INF, 1)
    rows.append(2 * np.arange(n)); cols.append(2 * np.arange(n) + 1); caps.append(inner)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            ny, nx = cells[:, 0] + dy, cells[:, 1] + dx
            ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            j = np.full(n, -1, np.int64)
            j[ok] = idx[ny[ok], nx[ok]]
            m = j >= 0
            rows.append(2 * np.nonzero(m)[0] + 1); cols.append(2 * j[m]); caps.append(np.full(m.sum(), INF))
    s_cells = np.nonzero(sources[cells[:, 0], cells[:, 1]])[0]
    t_cells = np.nonzero(sinks[cells[:, 0], cells[:, 1]])[0]
    rows += [np.full(len(s_cells), src), 2 * t_cells + 1]
    cols += [2 * s_cells, np.full(len(t_cells), snk)]
    caps += [np.full(len(s_cells), INF), np.full(len(t_cells), INF)]
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(caps)
    graph = sparse.csr_matrix((v.astype(np.int32), (r, c)), shape=(2 * n + 2, 2 * n + 2))
    result = maximum_flow(graph, src, snk, method="dinic")
    residual = (graph - result.flow).tocsr()
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reach = np.zeros(2 * n + 2, bool)
    reach[breadth_first_order(residual, src, directed=True, return_predecessors=False)] = True
    cut_cells = reach[2 * np.arange(n)] & ~reach[2 * np.arange(n) + 1]
    cut = np.zeros(region.shape, bool)
    cut[cells[cut_cells, 0], cells[cut_cells, 1]] = True
    _source_side = np.zeros(region.shape, bool)
    on = reach[2 * np.arange(n) + 1]
    _source_side[cells[on, 0], cells[on, 1]] = True
    return cut, int(result.flow_value)





# --- the check against the engine ----------------------------------------------------------------

def _open_area(game, tiles_w: int, tiles_h: int) -> tuple[int, int]:
    """Top-left tile of a buildable, walkable area this big, far from the start locations."""
    walk, _, build = game.terrain()
    th, tw = build.shape
    starts = [(s["x"] // 32, s["y"] // 32) for s in game.start_locations()]
    for ty in range(2, th - tiles_h - 2):
        for tx in range(2, tw - tiles_w - 2):
            if not build[ty:ty + tiles_h, tx:tx + tiles_w].all():
                continue
            if not walk[ty * 4:(ty + tiles_h) * 4, tx * 4:(tx + tiles_w) * 4].all():
                continue
            if min(math.dist((tx + tiles_w / 2, ty + tiles_h / 2), s) for s in starts) < 12:
                continue
            return tx, ty
    raise RuntimeError("no open area on this map")


def engine_passes(map_path, a: int, b: int, side: str, offset: tuple[int, int], walker: int) -> bool:
    """Place a and b in a fresh game (b at offset tiles from a's top-left), send walker from one
    side of their opening to the other, and see whether it went through the opening."""
    from gary import commands as C
    from gary.env import Game
    game = Game.new(map_path, ["T", "Z"], seed=1)
    try:
        obs = game.observe()
        terran = next(p["slot"] for p in obs["players"] if p["race"] == 1)
        zerg = next(p["slot"] for p in obs["players"] if p["race"] == 0)
        fp = Footprints(game)
        tx, ty = _open_area(game, 16, 16)
        ax, ay = tx + 5, ty + 5
        bx, by = ax + offset[0], ay + offset[1]
        for t, x, y in ((a, ax, ay), (b, bx, by)):
            f = fp[t]
            if not game.create_unit(terran, t, x * 32 + f.w // 2, y * 32 + f.h // 2):
                raise RuntimeError(f"couldn't place {t}")
        box_a, box_b = fp.box_at_tile(a, ax, ay), fp.box_at_tile(b, bx, by)
        owner = zerg if walker in (37, 38, 39, 41) else terran
        if side == "right":        # a vertical corridor: from above it to below it
            mid = (box_a[2] + box_b[0]) // 2
            lo, hi_ = max(box_a[1], box_b[1]), min(box_a[3], box_b[3])
            start, goal = (mid, min(box_a[1], box_b[1]) - 64), (mid, max(box_a[3], box_b[3]) + 64)
            corridor = (box_a[2] + 1, lo, box_b[0] - 1, hi_)
        else:                      # a horizontal corridor: from left of it to right of it
            mid = (box_a[3] + box_b[1]) // 2
            lo, hi_ = max(box_a[0], box_b[0]), min(box_a[2], box_b[2])
            start, goal = (min(box_a[0], box_b[0]) - 64, mid), (max(box_a[2], box_b[2]) + 64, mid)
            corridor = (lo, box_a[3] + 1, hi_, box_b[1] - 1)
        tag = game.create_unit(owner, walker, *start)
        if not tag:
            raise RuntimeError("couldn't place the walker")
        game.act(owner, C.select([tag]))
        game.act(owner, C.right_click(*goal))
        prev = start
        for _ in range(300):
            game.step(1)
            u = next((u for u in game.observe()["units"] if u["tag"] == tag), None)
            if u is None:
                return False
            if _cuts(prev, (u["x"], u["y"]), corridor):
                return True
            prev = (u["x"], u["y"])
            if math.dist(prev, goal) < 8:
                return False
        return False
    finally:
        game.close()


def _cuts(p, q, r) -> bool:
    """Whether the segment p-q passes through the rectangle r (x0, y0, x1, y1, inclusive)."""
    for k in range(9):
        x, y = p[0] + (q[0] - p[0]) * k / 8, p[1] + (q[1] - p[1]) * k / 8
        if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            return True
    return False


def check(map_path, pairs, walkers) -> list[dict]:
    """The arithmetic against the engine for these building pairs (every way they touch) and units."""
    from gary.env import Game
    game = Game.new(map_path, ["T", "Z"], seed=1)
    fp = Footprints(game)
    out = []
    for a, b in pairs:
        for side, offset, gap in neighbors(fp, a, b):
            for w in walkers:
                f = fp[w]
                need = f.box_w if side == "right" else f.box_h
                out.append({"a": a, "b": b, "side": side, "offset": offset, "gap": gap, "walker": w,
                            "arithmetic": gap >= need, "need": need})
    game.close()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as pool:
        got = list(pool.map(_engine_job, [(str(map_path), r["a"], r["b"], r["side"], r["offset"], r["walker"]) for r in out]))
    for r, e in zip(out, got):
        r["engine"] = e
    return out


def _engine_job(args) -> bool | str:
    try:
        return engine_passes(*args)
    except Exception as e:  # noqa: BLE001 - a case that can't be set up is reported, not fatal
        return f"error: {e}"


def main() -> None:
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "replays"
                                         / "stardata_tvz_standard_ozp3w.rep"))
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--buildings", nargs="+", type=int, default=[109, 111, 112, 122, 125])
    ap.add_argument("--walkers", nargs="+", type=int, default=[37, 0, 7])
    ap.add_argument("--frontier", metavar="SHA1", help="a home defense (eval/scenarios.py): its Terran's frontier")
    args = ap.parse_args()
    if args.frontier:
        import json
        from eval.scenarios import scenario_game, scenarios_path
        sc = next(json.loads(line) for line in scenarios_path("test").read_text(encoding="utf-8").splitlines()
                  if json.loads(line)["sha1"].startswith(args.frontier))
        game = scenario_game(sc)
        obs = game.observe()
        ccs = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] == sc["terran_slot"] and u["type"] == 106]
        starts = [(s["x"], s["y"]) for s in game.start_locations()]
        r = frontier(game, obs, ccs, [s for s in starts if min(math.dist(s, c) for c in ccs) > 20 * 32])
        print(f"bases at tiles {[(x // 32, y // 32) for x, y in ccs]}; cut {r.get('cut_width')} px"
              + (" (wide open)" if r["open"] else ""))
        for o in r["openings"]:
            e = o.exact.get("openings") or []
            print(f"  tile ({o.x // 32}, {o.y // 32}): {'closed to zerglings' if o.closed else 'opening'}, "
                  f"{o.width} px across (coarse)" + (f"; narrowest {e[0].width} px, gets through: "
                  f"{', '.join(WALKERS.get(u, str(u)) for u in e[0].passes) or 'nobody'}" if e else ""))
    if args.table:
        from gary.env import Game
        game = Game.new(args.map, ["T", "Z"], seed=1)
        fp = Footprints(game)
        print("building           tiles  margins (left top right bottom, px)  cost")
        for t, n in TERRAN_BUILDINGS.items():
            f, i = fp[t], game.unit_type_info(t)
            print(f"{n:18s} {f.w // 32}x{f.h // 32}    {' '.join(f'{m:3d}' for m in f.margins())}"
                  f"                       {i['minerals']}/{i['gas']}")
        print()
        print("pairs: the opening (px) with b right of a | b below a; z: zergling-tight (and so tight for everything"
              " bigger), m: marine-tight only; *: the engine is tighter than the arithmetic (--check)")
        names = {t: n for t, n in TERRAN_BUILDINGS.items() if t in args.buildings}
        print(" " * 18 + "".join(f"{n[:12]:>14s}" for n in names.values()))
        for a, na in names.items():
            cells = []
            for b in names:
                h_ = fp[a].margins()[2] + fp[b].margins()[0]
                v = fp[a].margins()[3] + fp[b].margins()[1]
                tag = lambda o, need_m: "z" if o < 16 else "m" if o < need_m else " "
                star = "*" if (a, b) in TIGHTER_BELOW else " "
                cells.append(f"{h_:3d}{tag(h_, 17)} |{v:3d}{'z' if star == '*' else tag(v, 20)}{star}")
            print(f"{na[:16]:16s}  " + "".join(f"{c:>14s}" for c in cells))
        print()
        print("unit         box (w x h)")
        for t, n in WALKERS.items():
            print(f"{n:12s} {fp[t].box_w} x {fp[t].box_h}")
        game.close()
    if args.check:
        pairs = list(itertools.combinations_with_replacement(args.buildings, 2))
        pairs += [(b, a) for a, b in pairs if a != b]
        rows = check(args.map, pairs, args.walkers)
        bad = [r for r in rows if r["engine"] != r["arithmetic"]]
        print(f"{len(rows)} cases (building pairs x ways they touch x units): the arithmetic agrees with the engine "
              f"in {len(rows) - len(bad)}")
        for r in bad[:40]:
            print(f"  {TERRAN_BUILDINGS.get(r['a'], r['a'])} | {TERRAN_BUILDINGS.get(r['b'], r['b'])} {r['side']} "
                  f"{r['offset']}: opening {r['gap']} px, {WALKERS.get(r['walker'], r['walker'])} needs {r['need']}: "
                  f"arithmetic {'passes' if r['arithmetic'] else 'blocked'}, engine {r['engine']}")


if __name__ == "__main__":
    main()
