"""Sim city (part of the executor, docs/CONTROL.md): where the buildings Gary needs anyway go, so that
they narrow its frontier.

The frontier is what separates all of Gary's bases from the enemy's side (gary/walls.py: the
natural's choke where the natural is the only way into the main, the main's way out while Gary has
only the main...). When the executor is about to place a Supply Depot, a Barracks or an Academy,
it asks here first: is there a spot by one of the frontier's openings that makes it narrower,
while a door stays for Gary's own army (a 32 x 32 unit, a vulture or a tank, still gets through:
pros leave their natural's choke about 2 tiles wide and hold it, they rarely close it), and it
grows the wall from an edge (zergling-tight against a cliff or a building: a building in the
middle of the gap would leave two openings)? The best such spot leaves the narrowest opening, checked exactly (the building's collision box added to
the opening's pixel-exact obstacles), and not while zerg fighters are near the opening (the SCV
would die there, and a depot that isn't finished is a supply block). Openings already narrower than TARGET px aren't walled
further, nor are openings wider than WIDEST (no choke to wall). Otherwise the executor's usual spot.
min_bases (option simcity_nat: 2): wall only once Gary has its natural (the pros never wall their
main's ramp in TvZ, they wall their natural).
"""

from __future__ import annotations

import math

from gary import terran as T

WALL_TYPES = {T.DEPOT, T.RAX, 112}       # Supply Depot, Barracks, Academy: built anyway, wall well
DOOR = 32                                # a unit this wide (vulture, tank) still gets through
TARGET = 48                              # an opening this narrow is walled enough
WIDEST = 10 * 32                         # an opening wider than this (coarse) isn't walled
REACH = 48                               # candidates: the building's box this near the cut (past its half-width)
TOUCH = 15                               # ...and this near a cliff or a building (zergling-tight)
UNSAFE = 10 * 32                         # no wall building while zerg fighters are this near the opening


class SimCity:
    def __init__(self, g, min_bases: int = 1):
        self.g = g
        self.min_bases = min_bases                   # wall only with this many Command Centers
        self._for, self._frontier = None, None

    def spot(self, unit_type: int, builder: int) -> tuple[int, int] | None:
        """A spot (top-left tile) for this building that narrows the frontier, or None."""
        if unit_type not in WALL_TYPES:
            return None
        if sum(1 for u in self.g.hi.observe()["units"] if u["owner"] == self.g.slot and u["type"] == T.CC) < self.min_bases:
            return None
        from gary.walls import Footprints, obstacles, square, widest_through, connected, fits
        g = self.g
        game = g.hi.game
        obs = g.hi.observe()
        fr = self._frontier_now(obs)
        if not fr or not fr["openings"]:
            return None
        fp = Footprints(game)
        size = fp[unit_type]
        best = None
        from gary.agent.fight import enemy_fighters
        zerg = enemy_fighters(obs, g.slot)
        for o in fr["openings"]:
            if o.closed or o.width > WIDEST or not o.exact or o.inside == o.outside:
                continue
            if any(math.dist((e["x"], e["y"]), (o.x, o.y)) < UNSAFE for e in zerg):
                continue                                      # not while the zerg are there: the builder would die
            # the window: the whole cut and the points right inside and outside of it, 6 tiles around
            pts = [*o.ends, o.inside, o.outside, (o.x, o.y)]
            x0, y0 = max(0, min(p[0] for p in pts) - 192), max(0, min(p[1] for p in pts) - 192)
            x1, y1 = max(p[0] for p in pts) + 192, max(p[1] for p in pts) + 192
            blocked = obstacles(game, obs, fp, (x0, y0, x1, y1))
            x1, y1 = x0 + blocked.shape[1], y0 + blocked.shape[0]
            for rt, rx, ry in g.reserved:                    # buildings ordered but not started: there too
                bx0, by0, bx1, by1 = fp.box_at_tile(rt, rx, ry)
                blocked[max(0, by0 - y0):max(0, by1 - y0 + 1), max(0, bx0 - x0):max(0, bx1 - x0 + 1)] = True
            a, b = (o.inside[1] - y0, o.inside[0] - x0), (o.outside[1] - y0, o.outside[0] - x0)
            now = widest_through(blocked, a, b)
            if now <= TARGET:
                continue
            narrow = (o.x, o.y)                               # the cut's middle (the bases' side is known there)
            reach = o.width / 2 + REACH
            # the opening's sides: the obstacles that bound it (cliffs, and wall buildings tight
            # against them), as groups of obstacles closer than TOUCH to each other
            import numpy as np
            from scipy import ndimage
            groups, _ = ndimage.label(ndimage.binary_dilation(blocked, iterations=TOUCH // 2))
            yy, xx = np.ogrid[0:blocked.shape[0], 0:blocked.shape[1]]
            ids = set()
            for ex, ey in o.ends:                             # the obstacles at the cut's two ends
                disk = (yy - (ey - y0)) ** 2 + (xx - (ex - x0)) ** 2 <= 24 ** 2
                ids |= set(np.unique(groups[disk & blocked])) - {0}
            sides = np.isin(groups, list(ids))
            ex_ = [p[0] for p in (*o.ends, narrow)]
            ey_ = [p[1] for p in (*o.ends, narrow)]
            for tx in range(min(ex_) // 32 - 5, max(ex_) // 32 + 5):
                for ty in range(min(ey_) // 32 - 5, max(ey_) // 32 + 5):
                    bx0, by0, bx1, by1 = fp.box_at_tile(unit_type, tx, ty)
                    if min(max(bx0 - p[0], p[0] - bx1, 0) ** 2 + max(by0 - p[1], p[1] - by1, 0) ** 2
                           for p in (*o.ends, narrow)) > REACH ** 2 and                             max(bx0 - narrow[0], narrow[0] - bx1, 0) ** 2 + max(by0 - narrow[1], narrow[1] - by1, 0) ** 2 > reach ** 2:
                        continue
                    cx, cy = tx * 32 + size.w // 2, ty * 32 + size.h // 2
                    if not fr["inside"][min(cy // 8, fr["inside"].shape[0] - 1), min(cx // 8, fr["inside"].shape[1] - 1)]:
                        continue                              # on the enemy's side of the frontier
                    if any(abs(tx - rx) < T.SIZE[rt][0] and abs(ty - ry) < T.SIZE[rt][1] for rt, rx, ry in g.reserved):
                        continue
                    if not g.executor.fits(unit_type, tx, ty, builder):
                        continue
                    # it grows the wall from a side of the opening, zergling-tight against it (a
                    # building in the middle of the gap would leave two openings to hold)
                    ex0, ey0, ex1, ey1 = bx0 - x0 - TOUCH, by0 - y0 - TOUCH, bx1 - x0 + TOUCH, by1 - y0 + TOUCH
                    if not (sides & blocked)[max(0, ey0):max(0, ey1 + 1), max(0, ex0):max(0, ex1 + 1)].any():
                        continue
                    trial = blocked.copy()
                    trial[max(0, by0 - y0):max(0, by1 - y0 + 1), max(0, bx0 - x0):max(0, bx1 - x0 + 1)] = True
                    if not connected(fits(trial, square(DOOR)), a, b):
                        continue                              # no door left
                    left = widest_through(trial, a, b, lo=DOOR)
                    score = (left, math.dist((cx, cy), narrow))
                    if left < now - 4 and (best is None or score < best[0]):
                        best = (score, (tx, ty))
        if best is not None:
            g.say(f"sim city: {T.NAMES.get(unit_type, unit_type)} in the wall, leaving {best[0][0]} px")
        return best[1] if best else None

    def _frontier_now(self, obs: dict):
        """The frontier of Gary's bases now (kept until its buildings change)."""
        from gary.walls import frontier
        g = self.g
        mine = [u for u in obs["units"] if u["owner"] == g.slot]
        key = frozenset((u["tag"], u["type"]) for u in mine if u["type"] in T.SIZE) | frozenset(g.reserved)
        if key != self._for:
            ccs = [(u["x"], u["y"]) for u in mine if u["type"] == T.CC]
            self._for = key
            self._frontier = frontier(g.hi.game, obs, ccs, list(g.enemy_starts)) if ccs and g.enemy_starts else None
        return self._frontier
