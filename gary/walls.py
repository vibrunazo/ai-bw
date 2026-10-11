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

Body blocks (block()): where units on hold position close a choke to a walker (zerglings), from
their collision boxes: the narrowest cross-section that cuts the way, wall to wall, the fewest
units, closed even with any one a few pixels off (gary/agent/block.py: the skill that holds it).

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


def square(size: int) -> Footprint:
    """A size x size unit box (for the widest unit that still gets through somewhere)."""
    return Footprint(size, size, size // 2, size // 2, size - size // 2 - 1, size - size // 2 - 1)


def widest_through(blocked, a, b, lo: int = 8, hi: int = 12 * 32) -> int:
    """The widest square unit (pixels) that still gets from a to b (window cells [y, x]); lo - 1 if
    none does."""
    if not connected(fits(blocked, square(lo)), a, b):
        return lo - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if connected(fits(blocked, square(mid)), a, b):
            lo = mid
        else:
            hi = mid - 1
    return lo


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


# --- body-blocking: units on hold position that close a choke ------------------------------------

BLOCK_ORDER = (32, 34, 7, 0, 7)        # who goes in a block first: firebats, medics, SCVs (BLOCK_SCVS), marines, more SCVs
BLOCK_SCVS = 2                        # ...SCVs before marines (pros hold with an SCV, a marine and an SCV)
BLOCK_ANGLES = 36                     # cross-sections tried at each point (every 5 degrees)
BLOCK_STEP = 4                        # pixels between the points tried along the way through
BLOCK_TRIES = 8                       # the narrowest cross-sections that cut the way, tried with units
BLOCK_CANDIDATES = 120                # ...looked for among this many narrowest
SEED_SNAP = 160                       # a block's points on a building or cliff stand for the open ground this near
BLOCK_SLOP = 8                        # a block stays closed with any one of its units this far off its spot (pixels)
BLOCK_BEHIND = 6                      # pixels between the block's boxes and the ones behind it


def _cross(blocked, c, angle: float, limit: int = 256):
    """The cross-section through c ([y, x] in the window) at this angle: its two ends (the last
    free pixels before an obstacle), and its length; infinite where a side meets no obstacle within
    limit or the window (no wall to close it against: the walker would go around)."""
    h, w = blocked.shape
    d = (math.sin(angle), math.cos(angle))
    ends = []
    for side in (1, -1):
        k = 0
        while True:
            y, x = int(round(c[0] + d[0] * (k + 1) * side)), int(round(c[1] + d[1] * (k + 1) * side))
            if k >= limit or not (0 <= y < h and 0 <= x < w):
                return c, c, math.inf
            if blocked[y, x]:
                break
            k += 1
        ends.append((c[0] + d[0] * k * side, c[1] + d[1] * k * side))
    return ends[0], ends[1], math.dist(ends[0], ends[1])


def _place_along(fp: Footprints, kinds: list[int], a, b) -> list[tuple[int, int, int]]:
    """The units spread evenly from end a to end b of a cross-section ([y, x]): along it each box
    takes its projection, the room left is shared by the gaps (walls included). As (type, x, y),
    window pixels; [] if they don't fit."""
    L = math.dist(a, b) or 1
    d = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
    extent = [fp[t].box_w * abs(d[1]) + fp[t].box_h * abs(d[0]) for t in kinds]
    slack = L - sum(extent)
    if slack < 0:
        return []
    gap, s, out = slack / (len(kinds) + 1), 0.0, []
    for t, e in zip(kinds, extent):
        s += gap + e / 2
        cy, cx = a[0] + d[0] * s, a[1] + d[1] * s           # the box's center; the unit's position from it
        f = fp[t]
        out.append((t, int(round(cx - (f.right - f.left) / 2)), int(round(cy - (f.down - f.up) / 2))))
        s += e / 2
    return out


def _with_boxes(blocked, fp: Footprints, units):
    b = blocked.copy()
    h, w = b.shape
    for t, x, y in units:
        f = fp[t]
        x0, y0, x1, y1 = max(0, x - f.left), max(0, y - f.up), min(w - 1, x + f.right), min(h - 1, y + f.down)
        if b[y0:y1 + 1, x0:x1 + 1].any():
            return None                                  # a box on an obstacle or another box: can't stand there
        b[y0:y1 + 1, x0:x1 + 1] = True
    return b


def _squeezer(f: Footprint) -> Footprint:
    """A walker a pixel smaller on each side: what a block is planned and checked against (between
    boxes that face each other only diagonally the engine squeezes units past less predictably)."""
    return Footprint(f.w, f.h, max(0, f.left - 1), max(0, f.up - 1), max(0, f.right - 1), max(0, f.down - 1))


def closed(game, obs: dict, units, outside: tuple[int, int], inside: tuple[int, int], margin: int = 4 * 32,
           walker: int = 37) -> bool:
    """Whether these units ((type, x, y) map pixels: their collision boxes) with the cliffs and the
    buildings in obs close the way from outside to inside to a walker (zerglings; with a pixel to
    spare on each side, _squeezer())."""
    fp = Footprints(game)
    if getattr(game, "_walk_cache", None) is None:
        game._walk_cache = game.terrain()[0]
    H, W = game._walk_cache.shape[0] * 8, game._walk_cache.shape[1] * 8
    x0, y0 = max(0, min(outside[0], inside[0]) - margin), max(0, min(outside[1], inside[1]) - margin)
    x1, y1 = min(W, max(outside[0], inside[0]) + margin), min(H, max(outside[1], inside[1]) + margin)
    blocked = obstacles(game, obs, fp, (x0, y0, x1, y1))
    walker = _squeezer(fp[walker])
    # the two points on open ground first (one can be on a building: the base's Command Center)
    bare = fits(blocked, walker)
    a = _seed(bare, (inside[1] - y0, inside[0] - x0), SEED_SNAP)
    b = _seed(bare, (outside[1] - y0, outside[0] - x0), SEED_SNAP)
    if a is None or b is None:
        return False
    h, w = blocked.shape
    for t, x, y in units:
        bx0, by0, bx1, by1 = fp.box_at(t, x - x0, y - y0)
        blocked[max(0, by0):max(0, min(h, by1 + 1)), max(0, bx0):max(0, min(w, bx1 + 1))] = True
    return not connected(fits(blocked, walker), a, b)


def block(game, obs: dict, outside: tuple[int, int], inside: tuple[int, int], pool: dict[int, int],
          through: list[tuple[int, int]] | None = None, margin: int = 6 * 32, walker: int = 37) -> dict | None:
    """Where units on hold position close the way from outside to inside (map pixels; e.g. the main
    ramp's bottom and a spot inside the main) to a walker (zerglings), from a pool of unit types
    ({type: count}): pros hold a ramp with an SCV, a marine and an SCV, or with one zealot, because
    they know where each one has to stand. Cross-sections (wall to wall) are tried at points along
    the way (`through`, a polyline from outside to inside; default the straight line), narrowest
    first; on each the fewest units (BLOCK_ORDER) whose boxes, spread evenly along it, leave the
    walker no way through (exact, as gaps(): the walker's box against the cliffs, buildings and
    their boxes), even with any one of them BLOCK_SLOP pixels off its spot along it. Ties go to the cross-section nearest inside (on the top of a ramp, not its
    bottom). None if nothing in the pool closes it.

    {"units": [(type, x, y)...], "ends": (two map points), "behind": (x, y): where others stand
    right behind the block, on the inside, "width": the cross-section's length}"""
    import numpy as np
    fp = Footprints(game)
    if getattr(game, "_walk_cache", None) is None:
        game._walk_cache = game.terrain()[0]
    H, W = game._walk_cache.shape[0] * 8, game._walk_cache.shape[1] * 8
    path = [outside] + list(through or []) + [inside]
    x0, y0 = max(0, min(p[0] for p in path) - margin), max(0, min(p[1] for p in path) - margin)
    x1, y1 = min(W, max(p[0] for p in path) + margin), min(H, max(p[1] for p in path) + margin)
    blocked = obstacles(game, obs, fp, (x0, y0, x1, y1))
    a, b = (inside[1] - y0, inside[0] - x0), (outside[1] - y0, outside[0] - x0)
    free = fits(blocked, fp[walker])
    a, b = _seed(free, a, SEED_SNAP), _seed(free, b, SEED_SNAP)
    if a is None or b is None or not connected(free, a, b):
        return None
    # the points tried: along the polyline, every BLOCK_STEP pixels, each with its narrowest cross-section
    tried = []
    walked = 0.0
    for p, q in zip(path, path[1:]):
        n = max(1, int(math.dist(p, q) // BLOCK_STEP))
        for k in range(n):
            c = (p[1] + (q[1] - p[1]) * k / n - y0, p[0] + (q[0] - p[0]) * k / n - x0)
            walked += math.dist(p, q) / n
            if blocked[int(c[0]), int(c[1])]:
                continue
            ea, eb, L = min((_cross(blocked, c, math.pi * i / BLOCK_ANGLES) for i in range(BLOCK_ANGLES)),
                            key=lambda r: r[2])
            if L < math.inf:
                tried.append((L, -walked, ea, eb))
    tried.sort()
    # only cross-sections that cut the way: closed along their whole length, the walker can't get
    # through (a narrow place beside an island isn't one: it goes around)
    walker_free = lambda bl: fits(bl, _squeezer(fp[walker]))
    cuts = []
    for L, at, ea, eb in tried[:BLOCK_CANDIDATES]:
        line = blocked.copy()
        n = int(L) + 1
        for k in range(n + 1):
            y, x = int(round(ea[0] + (eb[0] - ea[0]) * k / n)), int(round(ea[1] + (eb[1] - ea[1]) * k / n))
            line[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
        if not connected(walker_free(line), a, b):
            cuts.append((L, at, ea, eb))
        if len(cuts) >= BLOCK_TRIES:
            break
    order, scvs = [], pool.get(7, 0)
    for k, t in enumerate(BLOCK_ORDER):
        n = pool.get(t, 0) if t != 7 else (min(scvs, BLOCK_SCVS) if k == BLOCK_ORDER.index(7) else max(0, scvs - BLOCK_SCVS))
        order += [t] * n
    best = None
    for L, at, ea, eb in cuts:
        for n in range(1, len(order) + 1):
            if best is not None and (n > len(best[0]) or (n == len(best[0]) and at >= best[1])):
                break
            units = _place_along(fp, order[:n], ea, eb)
            boxes = _with_boxes(blocked, fp, units) if units else None
            if boxes is None or connected(walker_free(boxes), a, b):
                continue
            # it has to stay closed with any one unit a few pixels off its spot (clicks scatter)
            d = ((eb[0] - ea[0]) / (L or 1), (eb[1] - ea[1]) / (L or 1))
            robust = True
            for i, (t, x, y) in enumerate(units):
                for k in (BLOCK_SLOP, -BLOCK_SLOP):
                    moved = units[:i] + [(t, int(round(x + d[1] * k)), int(round(y + d[0] * k)))] + units[i + 1:]
                    mb = _with_boxes(blocked, fp, moved)
                    if mb is not None and connected(walker_free(mb), a, b):
                        robust = False
                        break
                if not robust:
                    break
            if robust:
                best = (units, at, ea, eb, L)
                break
    if best is None:
        return None
    units, _, ea, eb, L = best
    # behind it: from the block's middle toward inside, past the deepest box
    my, mx = (ea[0] + eb[0]) / 2, (ea[1] + eb[1]) / 2
    nd = (eb[1] - ea[1], -(eb[0] - ea[0]))                # a normal to the cross-section ([y, x])
    nl = math.hypot(*nd) or 1
    nd = (nd[0] / nl, nd[1] / nl)
    if (a[0] - my) * nd[0] + (a[1] - mx) * nd[1] < 0:
        nd = (-nd[0], -nd[1])
    depth = max(max(fp[t].box_w, fp[t].box_h) for t, _, _ in units)
    k = depth / 2 + BLOCK_BEHIND + 10
    return {"units": [(t, x + x0, y + y0) for t, x, y in units],
            "ends": ((int(ea[1]) + x0, int(ea[0]) + y0), (int(eb[1]) + x0, int(eb[0]) + y0)),
            "behind": (int(mx + nd[1] * k) + x0, int(my + nd[0] * k) + y0), "width": round(L)}


# --- the frontier: what separates all of a player's bases from the enemy's side ------------------

_source_side = None

FRONTIER_REACH = 24 * 4              # minitiles: the ground this near a base (by ground) is the base's side
FRONTIER_WIDEST = 12 * 4             # minitiles: a way in wider than this at its narrowest isn't a choke
FRONTIER_MAX = 4                     # openings at most
WALL_ELSEWHERE = 6                   # frontier(buildable=...): a cut minitile off buildable ground costs this many


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
    ends: tuple = ()                 # the cut's two ends (map pixels): where it meets what bounds the opening
    cells: tuple = ()                # the cut's minitiles ([y, x]): what a wall across it has to cover

    @property
    def closed(self) -> bool:
        """Zerglings don't get through (exact): a closed part of the wall, not a way in."""
        return bool(self.exact.get("walled", {}).get(37))


def frontier(game, obs: dict, bases: list[tuple[int, int]], enemy: list[tuple[int, int]],
             units=GAP_UNITS, reach: int = FRONTIER_REACH, buildable=None, level: bool = False) -> dict:
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
    its narrowest (no choke to hold).

    buildable (minitiles, bool): where a wall can stand. Given, the cut keeps to that ground where it
    can (crossing other ground costs WALL_ELSEWHERE cells): the line a wall can close, not the
    narrowest one (Colosseum II's naturals narrow on a ramp, unbuildable; the pros wall a little
    inside it).

    level: the bases' side keeps to their ground height (the main's plateau): the cut is where it
    drops away, the top of the main's ramp, not a narrower choke further out."""
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
    ground = free
    if level:
        height = game.terrain()[1]
        ground = free & np.isin(height, [int(height[by // 8, bx // 8]) for bx, by in bases])
    for _ in range(reach):                           # the ground within reach (by ground) of a base
        near = ndimage.binary_dilation(near, structure=eight) & ground
    far, _ = ndimage.label(free & ~near)
    ids = {far[c] for c in map(cell, enemy) if c and far[c]}
    outside = np.isin(far, list(ids))
    if not outside.any():
        return {"openings": [], "open": False}
    cost = None if buildable is None else np.where(buildable, 1, WALL_ELSEWHERE)
    cut, flow = _min_cut(near | home, home, near & ndimage.binary_dilation(outside, structure=eight), cost)
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
        far1 = (ys[np.argmax((ys - cy) ** 2 + (xs - cx) ** 2)], xs[np.argmax((ys - cy) ** 2 + (xs - cx) ** 2)])
        k2 = np.argmax((ys - far1[0]) ** 2 + (xs - far1[1]) ** 2)
        ends = ((int(far1[1]) * 8 + 4, int(far1[0]) * 8 + 4), (int(xs[k2]) * 8 + 4, int(ys[k2]) * 8 + 4))
        openings.append(FrontierOpening(cx * 8 + 4, cy * 8 + 4, int(piece.sum()) * 8, pin, pout, exact, ends,
                                        tuple(map(tuple, np.argwhere(piece)))))
    openings.sort(key=lambda o: o.width)
    width = int(cut.sum())                           # (minitiles; the flow, with no costs)
    return {"openings": openings, "open": width > FRONTIER_WIDEST, "cut_width": width * 8,
            "inside": side | cut}                   # (minitiles: the bases' side of the cut, and the cut)


WALL_GAP = 4 * 32                     # an opening narrower than this, beside a building, is a wall gap
GAP_INSIDE = 48                       # the post at a wall gap: this far inside it


def wall_gaps(game, obs: dict, slot: int, fr: dict) -> list[dict]:
    """The wall gaps of a frontier (frontier()): openings zerglings get through, narrower than
    WALL_GAP, with one of the player's buildings a side of them. Each: "at" (its narrowest place),
    "width", and "post" (GAP_INSIDE inside it: where units hold it)."""
    from gary.policy.army import is_building
    fp = Footprints(game)
    built = [fp.box_at(u["type"], u["x"], u["y"]) for u in obs["units"]
             if u["owner"] == slot and is_building(u["type"]) and u["type"] != 106]
    by = lambda o, b: max(b[0] - o.x, o.x - b[2], 0) ** 2 + max(b[1] - o.y, o.y - b[3], 0) ** 2 <= (o.width / 2 + 16) ** 2
    out = []
    for f in (fr or {}).get("openings", []):
        if f.closed or not f.exact.get("openings"):
            continue
        o = f.exact["openings"][0]
        if o.width < WALL_GAP and 37 in o.passes and any(by(o, b) for b in built):
            d = math.dist((o.x, o.y), f.inside) or 1
            out.append({"at": (o.x, o.y), "width": o.width,
                        "post": (o.x + (f.inside[0] - o.x) / d * GAP_INSIDE, o.y + (f.inside[1] - o.y) / d * GAP_INSIDE)})
    return out


def outside(fr: dict, p) -> bool:
    """Whether a map point is on the enemy's side of a frontier (frontier()["inside"] is the other)."""
    ins = fr["inside"]
    return not ins[min(ins.shape[0] - 1, int(p[1]) // 8), min(ins.shape[1] - 1, int(p[0]) // 8)]


def _min_cut(region, sources, sinks, cost=None):
    """The fewest cells of region to close so that no 8-connected way leads from a source cell to a
    sink cell (sources can't be closed): a minimum vertex cut, by max-flow over the cells (each in-
    to-out edge capacity 1, or cost[cell]). Returns (the cut cells, its cost), or (None, None) if a
    source touches a sink. Leaves the source side (cells reachable from the sources without
    crossing the cut) in _source_side."""
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
    inner = np.where(sources[cells[:, 0], cells[:, 1]], INF, 1 if cost is None else cost[cells[:, 0], cells[:, 1]])
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
