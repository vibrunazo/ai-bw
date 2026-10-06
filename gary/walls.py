"""Walls: which units get between two buildings (sim city).

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
    args = ap.parse_args()
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
