"""Sim city (part of the executor, docs/CONTROL.md): where the buildings Gary needs anyway go, so that
they wall its frontier.

The frontier is what separates Gary's bases, and the natural it will take, from the enemy's side
(gary/walls.py: the natural's way out where the natural is the only way into the main, the main's
own way out where it isn't...), found here along buildable ground without creep: the line a wall
can close, not necessarily the narrowest one (Colosseum II's naturals narrow on an unbuildable
ramp; the pros wall just inside it, two Supply Depots and a Barracks from cliff to cliff). When the
executor is about to place a Supply Depot, a Barracks or an Academy, it asks here first: the spot
that leaves the least of that line zerglings get across, checked exactly (collision boxes are
smaller than their tiles, by a different margin on each side: a wall tight against a cliff one
way leaves a gap the other), with a door a `door`-pixel unit still gets through (32: a vulture or
a tank; Gary can't lift a Barracks to open a closed wall; door 0 closes it). Lines longer than WIDEST aren't walled (no
choke: Fighting Spirit's naturals), and nothing is placed while zerg fighters are near (the SCV
would die there, and a depot that isn't finished is a supply block). Otherwise the executor's
usual spot.

Which maps and starts pros wall (TvZ, 22 maps): Colosseum II at every start, Neo Medusa and
Longinus at some starts; at most 13% of games elsewhere (docs/PROGRESS.md).
"""

from __future__ import annotations

import math

from gary import terran as T

WALL_TYPES = {T.DEPOT, T.RAX, 112}       # Supply Depot, Barracks, Academy: built anyway, wall well
DOOR = 32                                # the open part a wall leaves (pixels; 0: closed)
WIDEST = 10 * 32                         # a frontier line longer than this isn't walled
UNSAFE = 10 * 32                         # no wall building while zerg fighters are this near the line
AWAY = 3 * 32                            # whether zerglings get through: from this far inside the line to this far outside


class SimCity:
    def __init__(self, g, door: int = DOOR):
        self.g = g
        self.door = door
        self._for, self._frontier = None, None
        self._buildable = None

    def spot(self, unit_type: int, builder: int) -> tuple[int, int] | None:
        """A spot (top-left tile) for this building in the wall, or None."""
        if unit_type not in WALL_TYPES:
            return None
        from gary.walls import Footprints, connected, fits, obstacles, square
        from gary.agent.fight import enemy_fighters
        import numpy as np
        g = self.g
        obs = g.hi.observe()
        fr = self._frontier_now(obs)
        if not fr or not fr["openings"]:
            return None
        game = g.hi.game
        fp = Footprints(game)
        size, ling = fp[unit_type], fp[37]
        zerg = enemy_fighters(obs, g.slot)
        best = None
        for o in fr["openings"]:
            if o.closed or o.width > WIDEST:
                continue
            if any(math.dist((e["x"], e["y"]), (o.x, o.y)) < UNSAFE for e in zerg):
                continue
            # the window around the line, with the buildings ordered but not started in it
            # the ground well inside and well outside of it (right by the line, a wall building can
            # cover the point, and it would snap to the wrong side)
            d = math.dist(o.inside, o.outside) or 1
            ux, uy = (o.inside[0] - o.outside[0]) / d, (o.inside[1] - o.outside[1]) / d
            pin = (int(o.inside[0] + ux * AWAY), int(o.inside[1] + uy * AWAY))
            pout = (int(o.outside[0] - ux * AWAY), int(o.outside[1] - uy * AWAY))
            ys, xs = [c[0] for c in o.cells], [c[1] for c in o.cells]
            x0 = max(0, min(min(xs) * 8, pin[0], pout[0]) - 128)
            y0 = max(0, min(min(ys) * 8, pin[1], pout[1]) - 128)
            x1, y1 = max(max(xs) * 8, pin[0], pout[0]) + 136, max(max(ys) * 8, pin[1], pout[1]) + 136
            blocked = obstacles(game, obs, fp, (x0, y0, x1, y1))
            for rt, rx, ry in g.reserved:
                bx0, by0, bx1, by1 = fp.box_at_tile(rt, rx, ry)
                blocked[max(0, by0 - y0):max(0, by1 - y0 + 1), max(0, bx0 - x0):max(0, bx1 - x0 + 1)] = True
            a, b = (pin[1] - y0, pin[0] - x0), (pout[1] - y0, pout[0] - x0)
            line = np.zeros(blocked.shape, bool)              # the line, a minitile thicker each side
            for cy, cx in o.cells:
                line[max(0, cy * 8 - 8 - y0):cy * 8 + 16 - y0, max(0, cx * 8 - 8 - x0):cx * 8 + 16 - x0] = True

            def leak(grid) -> int:
                """How much of the line zerglings still get across (pixels where one can stand on it)."""
                free = fits(grid, ling)
                return int((free & line).sum()) if connected(free, a, b) else 0

            now = leak(blocked)
            if now == 0 or (self.door and not connected(fits(blocked, square(self.door)), a, b)):
                continue                                      # walled (as far as it goes)
            end = min(o.ends, key=lambda p: (p[1], p[0]))     # (ties: the wall grows from here)
            for tx in range(min(xs) // 4 - size.w // 32, max(xs) // 4 + 1):
                for ty in range(min(ys) // 4 - size.h // 32, max(ys) // 4 + 1):
                    bx0, by0, bx1, by1 = fp.box_at_tile(unit_type, tx, ty)
                    if not line[max(0, by0 - y0):max(0, by1 - y0 + 1), max(0, bx0 - x0):max(0, bx1 - x0 + 1)].any():
                        continue
                    if any(abs(tx - rx) < g.race.sizes[rt][0] and abs(ty - ry) < g.race.sizes[rt][1] for rt, rx, ry in g.reserved):
                        continue
                    if not g.executor.fits(unit_type, tx, ty, builder):
                        continue
                    trial = blocked.copy()
                    trial[max(0, by0 - y0):max(0, by1 - y0 + 1), max(0, bx0 - x0):max(0, bx1 - x0 + 1)] = True
                    if self.door and not connected(fits(trial, square(self.door)), a, b):
                        continue                              # (the door stays)
                    left = leak(trial)
                    if left >= now:
                        continue
                    cx, cy = tx * 32 + size.w // 2, ty * 32 + size.h // 2
                    score = (left, math.dist((cx, cy), end))
                    if best is None or score < best[0]:
                        best = (score, (tx, ty))
        if best is None:
            return None
        g.say(f"sim city: {g.race.names.get(unit_type, unit_type)} in the wall"
              + (", closing it" if best[0][0] == 0 else ""))
        return best[1]

    def _frontier_now(self, obs: dict):
        """The frontier of Gary's bases and its natural (kept until its buildings change), along
        ground a wall can stand on."""
        from gary.walls import frontier
        g = self.g
        mine = [u for u in obs["units"] if u["owner"] == g.slot]
        key = frozenset((u["tag"], u["type"]) for u in mine if u["type"] in g.race.sizes) | frozenset(g.reserved)
        if key != self._for:
            bases = [(u["x"], u["y"]) for u in mine if u["type"] in g.race.town_halls]
            if g.natural is not None and not any(math.dist(b, g.natural.center) < 4 * 32 for b in bases):
                bases.append(g.natural.center)                         # (where it will expand)
            self._for = key
            self._frontier = frontier(g.hi.game, obs, bases, list(g.enemy_starts), buildable=self.buildable()) \
                if bases and g.enemy_starts else None
        return self._frontier

    def buildable(self):
        """Per minitile: buildable terrain without the map's creep (None where the env can't tell)."""
        if self._buildable is None:
            import numpy as np
            ex = self.g.executor
            terrain = ex.terrain()
            if terrain is None:
                self._buildable = False
            else:
                ok = terrain[0] if ex._map_creep is None else terrain[0] & ~ex._map_creep
                self._buildable = np.repeat(np.repeat(ok, 4, 0), 4, 1)
        return self._buildable if self._buildable is not False else None

