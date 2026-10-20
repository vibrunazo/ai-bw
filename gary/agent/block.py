"""Choke blocks: units on hold position that close a choke with their bodies, as pros do.

The skill (ChokeBlock) holds one choke: any way in, given as where the enemy comes from (outside),
what it protects (inside) and points along the way between (through). Where each unit has to
stand so that the walker (zerglings by default; any unit type: its collision box decides) can't
physically get through is planned to the pixel from the units' collision boxes, the cliffs and the
buildings (gary/walls.py block(): the narrowest cross-section, the fewest units, closed even with
any one of them a few pixels off). The units that may block are the fighters near the choke
(Choke.units_at, units_reach), and workers from a mineral line when allowed (Choke.workers_from,
workers). Each blocker walks to the block, is placed on its spot with a precise right-click and
holds position (it attacks from there and nothing pushes it); when all hold and the block still
isn't closed (checked exactly: clicks scatter), the one farthest off is placed again, until the
enemy is there. The others wait
a little further in until it stands, then hold right behind it. With enemy fighters past it the
block is lost: the skill lets go (its workers fight, then mine again once it's calm).

Who asks for a block is a higher decision: g.blocks, name -> Choke (fight.py BlockRule holds every
one asked for). Chokes from the map: main_ramp(), natural_choke(); and the ways into any group of
bases (base_ways(): walls.frontier(), e.g. a late expansion's, to keep small harass out).

TODO (placement, later): place blockers the way pros do, by pressing them against the wall and
each other rather than clicking exact pixels. A right-click within a unit's own box targets the
unit itself, so a click can't nudge it a few pixels, and clicks scatter a few pixels anyway. So:
plan a packed line from one wall (walls.block(): the units touching, the slack at the far wall),
right-click each unit a little into the wall or past the last one placed, so the unit stops pressed
against it (spam the click while a group compresses), then hold position; closed() checks the
result as now. It doesn't depend on click accuracy, and a group can be pressed in at once rather
than placed one by one (the natural's 6-10 unit blocks were too slow: closed 42% in the drill).
Measured with eval/block_drill.py (--chokes ramp natural; in git before the refactor's Phase 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gary import terran as T
from gary.agent.intents import Intent

SLACK = 40                    # a unit this close to a spot is there (the blockers' gathering)
BACK_SLACK = 2 * 32           # the others this close to theirs are there (a clump)
GATHER = 4 * 32               # blockers farther from their spots walk to the block together first
ON_SPOT = 8                   # a blocker this close to its spot (pixels) is on it
REPLACE = 3                   # a holding blocker placed again this many times at most (the block not closed)
WAIT = 2 * 32                 # the others wait this much further in until the block stands
COMING = 8 * 32               # enemy fighters this close to the block: blockers hold where they are
LOST = 2 * 32                 # enemy fighters this much nearer the inside than the block: it's lost
HOLD_TTL = 24 * 2             # the units holding it are the skill's this long, renewed every turn
REACH = {0: 160, 32: 48}      # an enemy this close (+32) to a unit reaches it (marine, firebat; else 32)
HOLD_ORDERS, MOVE_ORDER = {107, 177}, 6     # (HoldPosition, MedicHoldPosition), Move
HARVEST_ORDERS = set(range(79, 91))
WORKERS_FROM = 12 * 32        # workers this close to Choke.workers_from can go


@dataclass
class Choke:
    """A way in to close: where the enemy comes from, what's behind it, and who may close it."""
    name: str
    outside: tuple[int, int]                  # map pixels: a point the enemy comes from
    inside: tuple[int, int]                   # ...one it must not reach
    through: list = field(default_factory=list)   # ...points along the way between (e.g. a ramp's top)
    units_at: tuple[int, int] = (0, 0)        # fighters within units_reach of this point may block
    units_reach: int = 16 * 32
    workers_from: tuple[int, int] | None = None   # workers from near this point (a mineral line) may too
    workers: int = 0                          # ...this many at most
    walker: int = 37                          # what it keeps out (zergling)


def _analysis(g):
    from gary.terrain import map_analysis
    a = map_analysis(g.hi.game, g.map)
    return a["starts"][min(range(len(g.map.starts)), key=lambda k: math.dist(g.map.starts[k], g.main.center))]


def main_ramp(g, workers: int = 4, walker: int = 37) -> Choke | None:
    """The main's ramp (gary/terrain.py), held from its top by the fighters in the main and SCVs from
    its mineral line; None where the map analysis isn't reliable."""
    try:
        r = _analysis(g)
    except Exception:                         # (no terrain in this game)
        return None
    if not r["reliable"]:
        return None
    top = tuple(r["ramp_top"])
    return Choke("main ramp", tuple(r["ramp_bottom"]), tuple(r["ramp_inside"]), [top], g.main.center,
                 int(math.dist(top, g.main.center)) + SLACK, g.main.center, workers, walker)


def natural_choke(g, workers: int = 0, walker: int = 37, beyond: int = 6 * 32) -> Choke | None:
    """The natural's choke (gary/terrain.py: the narrowest place on its way out toward the enemy),
    held by the fighters at the natural; outside: beyond the choke, away from the natural."""
    try:
        r = _analysis(g)
    except Exception:
        return None
    nat, ch = tuple(r["natural"]), tuple(r["natural_choke"])
    d = math.dist(nat, ch) or 1
    out = (int(ch[0] + (ch[0] - nat[0]) / d * beyond), int(ch[1] + (ch[1] - nat[1]) / d * beyond))
    return Choke("natural", out, nat, [ch], nat, int(d) + 6 * 32, nat if workers else None, workers, walker)


def base_ways(g, obs: dict, bases: list[tuple[int, int]], name: str = "base", workers: int = 0,
              walker: int = 37) -> list[Choke]:
    """The ways into a group of bases (Command Center positions; e.g. a late expansion): each
    opening of walls.frontier() against the enemy's start locations that the walker still gets
    through, as a Choke held by the fighters near it ([] where the bases are open ground)."""
    from gary.walls import frontier
    enemy = [s for s in g.map.starts if math.dist(s, g.main.center) > 32 * 32] or g.map.starts
    fr = frontier(g.hi.game, obs, bases, enemy)
    if fr.get("open"):
        return []
    out = []
    for k, o in enumerate(fr["openings"]):
        if o.closed:
            continue
        base = min(bases, key=lambda b: math.dist(b, (o.x, o.y)))
        out.append(Choke(f"{name} {k}", tuple(o.outside), tuple(o.inside), [(o.x, o.y)], (o.x, o.y),
                         12 * 32, base if workers else None, workers, walker))
    return out


class ChokeBlock:
    """The skill holding one Choke (see the module's doc). step(): at most one order per turn
    (True if it gave one); units: what it holds; rally: right behind the block."""

    def __init__(self, g, choke: Choke, source: str = "rule:block"):
        self.g = g
        self.choke = choke
        self.source = source
        self.plan = None                      # walls.block()'s plan, once
        self.slot_of: dict[int, int] = {}     # blocker tag -> its spot (index in the plan)
        self.workers: set[int] = set()        # workers the block took
        self.fetch: Intent | None = None      # the order sending workers to it (whichever the clicks take)
        self.told: dict[int, tuple[str, int]] = {}
        self.replaced: dict[int, int] = {}    # blocker tag -> times placed again on a block not closed
        self.units: set[int] = set()
        self.rally = None
        self.lost = False

    def release(self, mine: list[dict]) -> None:
        g = self.g
        g.leases.release(self.units)
        if self.workers:                      # to the pull rule: they fight, then mine once it's calm
            g.pulled |= self.workers & {u["tag"] for u in mine}
            g.calm_since = g.hi.frame
        self.units, self.rally, self.workers, self.fetch, self.slot_of = set(), None, set(), None, {}

    def _closed(self, obs: dict, blockers: list[dict]) -> bool:
        from gary.walls import closed
        c = self.choke
        seen = {"units": [u for u in obs["units"] if u["owner"] in (self.g.slot, 11)]}
        return closed(self.g.hi.game, seen, [(u["type"], u["x"], u["y"]) for u in blockers], c.outside, c.inside,
                      walker=c.walker)

    def _middle(self):
        if self.plan:
            (ax, ay), (bx, by) = self.plan["ends"]
            return (ax + bx) / 2, (ay + by) / 2
        return self.choke.through[0] if self.choke.through else self.choke.inside

    def step(self, obs: dict, mine: list[dict], fighters: list[dict]) -> bool:
        g, c = self.g, self.choke
        now = g.hi.frame
        mid = self._middle()
        # enemy fighters past it, nearer what it protects than the block is: lost
        if any(math.dist((e["x"], e["y"]), c.inside) < math.dist(mid, c.inside) - LOST for e in fighters):
            self.lost = True
            self.release(mine)
            return False
        by_tag = {u["tag"]: u for u in mine}
        army = [u for u in mine if u["type"] in g.army_types and u["completed"]
                and math.dist((u["x"], u["y"]), c.units_at) <= c.units_reach]
        if not army and not self.workers:
            self.release(mine)
            return False
        aside = g.task_aside
        busy = aside is not None and aside.intent is not None and aside.intent.source == self.source
        # the workers the last fetch took (whichever its clicks got), once it's done; one still
        # mining was never sent (the clicks took another, or none): fetched again
        if self.fetch is not None and not (busy and aside.intent is self.fetch):
            took = [t for t in self.fetch.units if t in by_tag][:self.fetch.params.get("need", 99)]
            self.workers |= set(took)         # (an extra one goes back to mining once idle)
            for t in took:
                self.told[t] = ("move", now)
            self.fetch = None
        self.workers = {t for t in self.workers if t in by_tag and by_tag[t]["order"] not in HARVEST_ORDERS}
        if self.plan is None:                 # the plan, once: the fighters here, and the workers allowed
            from gary.walls import block
            pool = {}
            for u in army:
                pool[u["type"]] = pool.get(u["type"], 0) + 1
            if c.workers and c.workers_from is not None:
                pool[g.race.worker] = c.workers
            seen = {"units": [u for u in obs["units"] if u["owner"] in (g.slot, 11)]}
            self.plan = block(g.hi.game, seen, c.outside, c.inside, pool, through=c.through, walker=c.walker) or \
                {"units": [], "behind": c.inside, "ends": (c.inside, c.inside)}
            mid = self._middle()
        spots = self.plan["units"]
        # each spot keeps its unit while it lives; an empty one: the nearest free unit of its type
        self.slot_of = {t: i for t, i in self.slot_of.items() if t in by_tag}
        taken = set(self.slot_of.values())
        for i, (kind, x, y) in enumerate(spots):
            if i in taken:
                continue
            pool = [u for u in (army if kind != g.race.worker else [by_tag[t] for t in self.workers]) if u["type"] == kind
                    and u["tag"] not in self.slot_of]
            if pool:
                u = min(pool, key=lambda u: math.dist((u["x"], u["y"]), (x, y)))
                self.slot_of[u["tag"]] = i
        blockers = [by_tag[t] for t in self.slot_of]
        spot = lambda u: spots[self.slot_of[u["tag"]]][1:]
        back = [u for u in army if u["tag"] not in self.slot_of]
        behind = self.plan["behind"]
        formed = bool(blockers) and len(blockers) == len(spots) and all(
            math.dist((u["x"], u["y"]), spot(u)) <= ON_SPOT and u.get("order") in HOLD_ORDERS for u in blockers)
        if spots and not formed:              # out of the blockers' way until the block stands
            dn = math.dist(mid, behind) or 1
            behind = (int(behind[0] + (behind[0] - mid[0]) / dn * WAIT), int(behind[1] + (behind[1] - mid[1]) / dn * WAIT))
        self.units = {u["tag"] for u in army} | self.workers | (set(self.fetch.units) if self.fetch else set())
        self.rally = behind
        g.leases.take(self.units, Intent("HOLD", list(self.units), None, {}, self.source, ttl=HOLD_TTL))
        told = self.told
        reached = lambda u: any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) <= REACH.get(u["type"], 32) + 32
                                for e in fighters)
        coming = any(math.dist((e["x"], e["y"]), mid) < COMING for e in fighters)
        fresh = lambda u, kind, wait=24: told.get(u["tag"], ("", -10 ** 9))[0] != kind or now - told[u["tag"]][1] >= wait
        off = lambda u: math.dist((u["x"], u["y"]), spot(u))
        walking = lambda u: u.get("order") == MOVE_ORDER
        sent = lambda u, kind: told.get(u["tag"], ("",))[0] == kind and walking(u)

        def order(kind: str, group: list[dict], key: tuple, target, fetch: bool = False, **params) -> bool:
            for u in group:
                told[u["tag"]] = (kind, now)
            tags = [u["tag"] for u in group]
            intent = Intent(kind.upper() if kind != "place" else "MOVE", tags, target,
                            {"key": key, "tags": tags, "center": (group[0]["x"], group[0]["y"]), "exact": True,
                             "any_scv": fetch, **params}, self.source)
            if fetch:
                self.fetch = intent
            g.motor.start(intent)
            return True

        # 1. an enemy on a unit that isn't holding: hold (it shoots, from where it is)
        hit = [u for u in blockers + back if reached(u) and u.get("order") not in HOLD_ORDERS and fresh(u, "hold")][:12]
        if hit:
            return order("hold", hit, ("hold",), None)
        if busy:                              # its last order still being given: let it finish
            return False
        # 2. workers for the block's worker spots: any from the mineral line, to the block (one order)
        need = sum(1 for kind, _, _ in spots if kind == g.race.worker) - len(self.workers)
        if need > 0 and not coming and c.workers_from is not None:
            free = sorted((u for u in mine if u["type"] == g.race.worker and u["completed"] and u["tag"] != g.scout
                           and u["tag"] not in g.pulled and u["tag"] not in self.workers
                           and u["tag"] not in g.block_units and u["order"] in HARVEST_ORDERS | {3}
                           and math.dist((u["x"], u["y"]), c.workers_from) < WORKERS_FROM),
                          key=lambda u: math.dist((u["x"], u["y"]), mid))[:need]
            if free:
                return order("move", free, ("move", int(self.plan["behind"][0]), int(self.plan["behind"][1])),
                             self.plan["behind"], fetch=True, need=need, exclude=sorted(self.workers))
        # 3. blockers far from their spots: to them (one order)
        away = [u for u in blockers if off(u) > GATHER and not reached(u) and fresh(u, "move", 48) and not sent(u, "move")]
        if away:
            p = (sum(spot(u)[0] for u in away) // len(away), sum(spot(u)[1] for u in away) // len(away))
            return order("move", away[:12], ("move", p[0], p[1]), p)
        # 4. the others: out of the way until the block stands, then right behind it (one order)
        far = [u for u in back if math.dist((u["x"], u["y"]), behind) > BACK_SLACK and not reached(u)
               and fresh(u, "move", 48) and not sent(u, "move")][:12]
        if far:
            return order("move", far, ("move", int(behind[0]), int(behind[1])), behind)
        # 5. on its spot (or the enemy here): hold position, each as it gets there
        settle = [u for u in blockers if (off(u) <= ON_SPOT or coming) and not walking(u)] + \
                 [u for u in back if math.dist((u["x"], u["y"]), behind) <= BACK_SLACK]
        settle = [u for u in settle if u.get("order") not in HOLD_ORDERS and fresh(u, "hold")][:12]
        if settle:
            return order("hold", settle, ("hold",), None)
        # 6. near its spot but off it: placed there with a precise right-click (before the enemy is there)
        place = [u for u in blockers if not coming and not reached(u) and ON_SPOT < off(u) <= GATHER
                 and not sent(u, "place") and fresh(u, "place", 12)]
        if place:
            return self._place(order, max(place, key=off), spot)
        # 7. all holding on their spots yet not closed (exactly: clicks scatter a few pixels): the one
        # farthest from its spot placed again, a few times at most each
        if blockers and len(blockers) == len(spots) and not coming and all(
                u.get("order") in HOLD_ORDERS for u in blockers) and not self._closed(obs, blockers):
            again = [u for u in blockers if off(u) >= 1 and self.replaced.get(u["tag"], 0) < REPLACE and not reached(u)]
            if again:
                u = max(again, key=off)
                self.replaced[u["tag"]] = self.replaced.get(u["tag"], 0) + 1
                return self._place(order, u, spot)
        return False

    @staticmethod
    def _place(order, u: dict, spot) -> bool:
        x, y = spot(u)
        return order("place", [u], ("place", x, y), (x, y))


# --- what a decision to block sees -----------------------------------------------------------------

FEATURES = ["minutes", "bases", "enemy_near", "enemy_close", "enemy_in", "enemy_dist", "own_army", "own_at_choke",
            "workers", "bunkers_at", "loaded_at", "bunkers_else", "loaded_else", "width"]
NEAR, CLOSE = 12 * 32, 6 * 32        # enemy fighters this close to the choke (pixels)


def features(g, obs: dict, choke: Choke, width: float = 0.0) -> dict:
    """What Gary sees that bears on closing this choke now, from its own view (fogged): the enemy's
    fighters near it (supply), how close, any already past it; its own fighters (in reach, at the
    choke), its workers, its bunkers at the choke and elsewhere with the marines it put in them,
    its bases, the game time, and the choke's width."""
    from gary.policy.army import is_building, is_worker, supply_x2
    mid = choke.through[0] if choke.through else choke.inside
    d_mid = math.dist(mid, choke.inside)
    enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
             and not is_worker(u["type"]) and not is_building(u["type"])]
    mine = [u for u in obs["units"] if u["owner"] == g.slot]
    sup = lambda us: sum(supply_x2(u["type"]) for u in us) / 2
    dist = lambda u, p: math.dist((u["x"], u["y"]), p)
    army = [u for u in mine if u["type"] in g.army_types and u["completed"]]
    bunkers = [u for u in mine if u["type"] == T.BUNKER and u["completed"]]
    at = [b for b in bunkers if dist(b, mid) < CLOSE]
    loaded = lambda bs: sum(len(g.loaded.get(b["tag"], ())) for b in bs)
    return {"minutes": obs["frame"] / 24 / 60,
            "bases": sum(1 for u in mine if u["type"] in g.race.town_halls),
            "enemy_near": sup([e for e in enemy if dist(e, mid) < NEAR]),
            "enemy_close": sup([e for e in enemy if dist(e, mid) < CLOSE]),
            "enemy_in": sup([e for e in enemy if dist(e, choke.inside) < d_mid - LOST]),
            "enemy_dist": min((dist(e, mid) for e in enemy), default=NEAR * 2) / 32,
            "own_army": sup([u for u in army if dist(u, choke.units_at) <= choke.units_reach]),
            "own_at_choke": sup([u for u in army if dist(u, mid) < CLOSE]),
            "workers": sum(1 for u in mine if u["type"] == g.race.worker),
            "bunkers_at": len(at), "loaded_at": loaded(at),
            "bunkers_else": len(bunkers) - len(at), "loaded_else": loaded(bunkers) - loaded(at),
            "width": width / 32}
