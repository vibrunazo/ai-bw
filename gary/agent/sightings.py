"""What a Gary saw and lost between its decisions, in the game's records (gary/decision_log.py).

A decision records what Gary saw at the moment it decided; nothing recorded what it saw in between,
so "it saw 12 zerglings at its natural at 6:02 and kept the move at 6:05" couldn't be read from the
records, nor "the marine died alone at 7:40, held by the army layer, the hands on a depot". These
records fill that in, read with eval/why.py, for Gary-T (side "T", from the arbiter) and Gary-Z (side
"Z", from its act loop). Three kinds:

  seen     enemy fighters in sight: on contact (none in sight for 10 s before), when 2+ supply of
           them is new (not seen in the last 30 s), else every 10 s while in sight: how many, their
           supply and types, where (tiles), by which of the player's bases, how far from its nearest
           building, and whether they are closing on it (tiles gained in the last 5 s; negative:
           leaving); an enemy building the first time it is in sight (what: "building")
  belief   once a minute, with the belief model on (Gary-T): the enemy army supply it expects, its
           50th and 80th percentiles, and the most seen at once in the last 3 minutes
  death    one of the player's fighters gone: what, where, by which base, how many own fighters were
           within 6 tiles and how far the nearest (alone: none), enemy fighters within 8 tiles, which
           source held its lease (Gary-T), and who had the hands. Marines Gary sent into a bunker, or
           standing by one, aren't deaths.

Looks every half second at what the player sees (hi.observe(): fogged, from its reaction time ago);
no action. A look that finds time gone backwards (a branch restored, eval/pairs.py) starts over.
"""

from __future__ import annotations

import math
from collections import deque

from gary import terran as T
from gary import zerg as Z
from gary.decision_log import LOG

EVERY = 12                     # frames between looks (half a second)
AGAIN = 10 * 24                # a group in sight is recorded again this often
CONTACT_GAP = 10 * 24          # nothing in sight this long, then something: contact
NEW_WINDOW = 30 * 24           # a tag not seen this long counts as new
NEW_SUPPLY = 2                 # new supply (x2) that earns a record: 2 zerglings, a hydralisk
CLOSING_OVER = 5 * 24          # closing speed measured over this long
BELIEF_EVERY = 60 * 24
ALONE = 6 * 32                 # own fighter within this: not alone (eval/losses.py's measure)
ENEMY_NEAR = 8 * 32
BY_BUNKER = 3 * 32             # a marine gone this close to an own bunker was loaded, not killed
NEUTRAL = 11


def _name(t: int) -> str:
    return T.NAMES.get(t) or Z.NAMES.get(t) or str(t)


class Sightings:
    def __init__(self, hi, side: str, mapinfo, home: tuple[int, int] | None, fighter, loaded=None, leases=None,
                 belief=None):
        """hi: the player's interface (hi.slot its slot); mapinfo: the map (None: no bases named); home: its
        main's center (px); fighter(unit): one of its army units (not a worker); loaded(): bunker tag -> tags
        inside (Gary-T); leases(): its Leases (Gary-T); belief(): (BeliefModel, BeliefMemory) or None."""
        self.hi, self.side, self.map = hi, side, mapinfo
        self.fighter, self.loaded, self.leases, self.belief = fighter, loaded, leases, belief
        self.main = mapinfo.base_near(home) if mapinfo is not None and home is not None else None
        self.natural = mapinfo.natural_of(self.main) if self.main is not None else None
        self._reset()

    @classmethod
    def for_gary(cls, g) -> "Sightings":
        """Gary-T's: its army types, bunker loads, leases and the army layer's belief model (set later)."""
        from gary.policy.army import is_worker
        return cls(g.hi, g.hi.side, g.map, g.main.center if g.main is not None else None,
                   lambda u: u["type"] in g.army_types and not is_worker(u["type"]),
                   loaded=lambda: g.loaded, leases=lambda: g.leases,
                   belief=lambda: (getattr(g.army, "belief", None), getattr(g.army, "belief_mem", None)))

    def _reset(self) -> None:
        self.next_look = 0
        self.last_frame = -1
        self.seen_at: dict[int, int] = {}                 # enemy fighter tag -> frame last in sight
        self.last_in_sight = -10 ** 6                     # frame enemy fighters were last in sight
        self.last_record = -10 ** 6
        self.centers: deque[tuple[int, float]] = deque()  # (frame, tiles to the nearest own building)
        self.buildings: set[int] = set()                  # enemy building tags seen
        self.fighters: dict[int, tuple[int, int, int]] = {}   # own fighter tag -> (type, x, y) last seen
        self.enemies: list[tuple[int, int]] = []          # enemy fighters' positions at the last look
        self.supply_seen: deque[tuple[int, int]] = deque()    # (frame, enemy supply x2 in sight)
        self.next_belief = BELIEF_EVERY
        self.broken = False

    def look(self, obs: dict | None = None) -> None:
        """Every half second: what the player sees now (obs, or hi.observe(): fogged, from its reaction time ago)."""
        if self.broken:
            return
        hi = self.hi
        if hi.frame < self.last_frame:                    # a branch restored the game: start over
            self._reset()
        self.last_frame = hi.frame
        if hi.frame < self.next_look:
            return
        self.next_look = hi.frame + EVERY
        try:
            self._look(obs if obs is not None else hi.observe())
        except Exception as e:                            # (never costs a game: say so once, stop looking)
            self.broken = True
            LOG.note(self.side, int(hi.frame), f"sightings stopped: {type(e).__name__}: {e}")

    def _look(self, obs: dict) -> None:
        frame = obs["frame"]
        from gary.policy.army import is_building, is_worker, supply_x2
        slot = self.hi.slot
        mine, theirs = [], []
        for u in obs["units"]:
            if u["owner"] == slot:
                mine.append(u)
            elif u["owner"] != NEUTRAL and ("visible_to" not in u or (u["visible_to"] >> slot) & 1):
                theirs.append(u)
        own_buildings = [(u["x"], u["y"]) for u in mine if is_building(u["type"])]
        own_fighters = [u for u in mine if u["completed"] and self.fighter(u)]
        enemy_fighters = [u for u in theirs if supply_x2(u["type"]) > 0 and not is_worker(u["type"])
                          and not is_building(u["type"])]
        self._deaths(frame, mine, own_fighters)
        self._buildings(frame, theirs, own_buildings, is_building)
        self._fighters(frame, enemy_fighters, own_buildings, supply_x2)
        self._belief(frame, own_fighters, mine, supply_x2, is_worker)
        self.enemies = [(u["x"], u["y"]) for u in enemy_fighters]
        self.fighters = {u["tag"]: (u["type"], u["x"], u["y"]) for u in own_fighters}

    # --- where ---------------------------------------------------------------------------------------

    def _near(self, x: int, y: int) -> str:
        if self.map is None:
            return "?"
        base = self.map.base_near((x, y))
        if math.dist((x, y), base.center) > 12 * 32:
            return "field"
        return "main" if base is self.main else "natural" if base is self.natural else f"base {base.tile[0]},{base.tile[1]}"

    @staticmethod
    def _tiles_to(points, x: int, y: int) -> float | None:
        if not points:
            return None
        return round(min(math.dist((x, y), p) for p in points) / 32, 1)

    # --- the records ---------------------------------------------------------------------------------

    def _fighters(self, frame: int, enemy: list[dict], own_buildings: list, supply_x2) -> None:
        if not enemy:
            self.supply_seen.append((frame, 0))
            return
        new = sum(supply_x2(u["type"]) for u in enemy if frame - self.seen_at.get(u["tag"], -10 ** 6) > NEW_WINDOW)
        for u in enemy:
            self.seen_at[u["tag"]] = frame
        contact = frame - self.last_in_sight > CONTACT_GAP
        self.last_in_sight = frame
        supply = sum(supply_x2(u["type"]) for u in enemy)
        self.supply_seen.append((frame, supply))
        while self.supply_seen and frame - self.supply_seen[0][0] > 180 * 24:
            self.supply_seen.popleft()
        cx = sum(u["x"] for u in enemy) / len(enemy)
        cy = sum(u["y"] for u in enemy) / len(enemy)
        dist = self._tiles_to(own_buildings, int(cx), int(cy))
        if dist is not None:
            self.centers.append((frame, dist))
            while self.centers and frame - self.centers[0][0] > CLOSING_OVER:
                self.centers.popleft()
        if not (contact or new >= NEW_SUPPLY or frame - self.last_record >= AGAIN):
            return
        self.last_record = frame
        types: dict[str, int] = {}
        for u in enemy:
            types[_name(u["type"])] = types.get(_name(u["type"]), 0) + 1
        r = {"f": int(frame), "kind": "seen", "side": self.side, "what": "contact" if contact else "fighters",
             "n": len(enemy), "supply": supply / 2, "types": types, "at": [int(cx // 32), int(cy // 32)],
             "near": self._near(int(cx), int(cy))}
        if dist is not None:
            r["dist"] = dist
            if len(self.centers) > 1 and frame - self.centers[0][0] >= CLOSING_OVER // 2:
                r["closing"] = round(self.centers[0][1] - dist, 1)    # tiles gained on the player's buildings
        if new >= NEW_SUPPLY and not contact:
            r["new"] = new / 2
        LOG.records.append(r)

    def _buildings(self, frame: int, theirs: list[dict], own_buildings: list, is_building) -> None:
        for u in theirs:
            if not is_building(u["type"]) or u["tag"] in self.buildings:
                continue
            self.buildings.add(u["tag"])
            LOG.records.append({"f": int(frame), "kind": "seen", "side": self.side, "what": "building",
                                "type": _name(u["type"]), "done": bool(u.get("completed", True)),
                                "at": [u["x"] // 32, u["y"] // 32], "near": self._near(u["x"], u["y"]),
                                "dist": self._tiles_to(own_buildings, u["x"], u["y"])})

    def _deaths(self, frame: int, mine: list[dict], own_fighters: list[dict]) -> None:
        if not self.fighters:
            return
        alive = {u["tag"] for u in mine}
        loads = self.loaded() if self.loaded is not None else None
        loaded = set().union(*loads.values()) if loads else set()
        bunkers = [(u["x"], u["y"]) for u in mine if u["type"] == T.BUNKER]
        was = [(x, y) for _, x, y in self.fighters.values()]
        leases = self.leases() if self.leases is not None else None
        for tag, (t, x, y) in self.fighters.items():
            if tag in alive or tag in loaded:
                continue
            if t == T.MARINE and bunkers and min(math.dist((x, y), b) for b in bunkers) <= BY_BUNKER:
                continue                                  # (into a bunker by the game's own hand)
            others = [math.dist((x, y), p) for p in was if p != (x, y)]
            within = sum(1 for d in others if d <= ALONE)
            r = {"f": int(frame), "kind": "death", "side": self.side, "type": _name(t), "at": [x // 32, y // 32],
                 "near": self._near(x, y), "with": within, "alone": within == 0,
                 "enemy_near": sum(1 for p in self.enemies if math.dist((x, y), p) <= ENEMY_NEAR),
                 "hands": LOG.owner.get(self.side, "?")}
            if others:
                r["nearest"] = round(min(others) / 32, 1)
            held = leases.held.get(tag) if leases is not None else None
            if held:                                      # the last source that ordered it (its lease may be over)
                r["held_by"] = f"{held[0]} ({held[1]}{'' if held[2] > frame else ', lease over'})"
            LOG.records.append(r)

    def _belief(self, frame: int, own_fighters: list[dict], mine: list[dict], supply_x2, is_worker) -> None:
        if frame < self.next_belief or self.belief is None:
            return
        self.next_belief = frame + BELIEF_EVERY
        model, mem = self.belief() or (None, None)
        if model is None or mem is None:
            return
        x = mem.vector(frame, sum(supply_x2(u["type"]) for u in own_fighters),
                       sum(1 for u in mine if is_worker(u["type"])))
        out = model.predict(x)
        r = {"f": int(frame), "kind": "belief", "side": self.side, "supply": round(out["supply"] / 2, 1),
             "seen_3min": max((s for _, s in self.supply_seen), default=0) / 2}
        for k in ("q50", "q80"):
            if k in out:
                r[k] = round(out[k] / 2, 1)
        drones = out["counts"].get(Z.DRONE)
        if drones is not None:
            r["drones"] = round(drones, 1)
        LOG.records.append(r)
