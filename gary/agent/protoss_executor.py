"""The Protoss executor (docs/REFACTOR.md Phase 4): what only Protoss hands do, over the race-neutral
Executor. A probe places a building and leaves (it warps in alone: nothing to resume, no add-ons); units
are trained from their buildings, as the Terran's are; a building other than a Nexus, Pylon or
Assimilator stands only in a Pylon's power field, which the game's placement check holds (can_place:
no Gateway spot beside a Nexus without a Pylon), so the hands' part is the Pylons: one ahead of
production for supply, and one when a powered building found no powered spot. An Archon or Dark Archon the
macro decides is two of its Templar merged: the two with least energy (the pros merge Templar spent on
storms), the second nearest the first, selected exactly and T (R) pressed."""

from __future__ import annotations

import math

from gary import commands as C
from gary import protoss as P
from gary.agent.executor import Executor
from gary.agent.macro import BUILD, EXPAND, RESEARCH, TRAIN, UPGRADE, WAIT
from gary.agent.gary import Gary, Task
from gary.agent.intents import Intent

UNPOWERED = {P.NEXUS, P.PYLON, P.ASSIMILATOR}    # what stands without a Pylon's power
POWER_SHORT_FOR = 20 * 24                        # a powered build that found no powered spot calls a Pylon this long
MERGING = {105, 106, 183}                        # ArchonWarp, CompletingArchonSummon, DarkArchonMeld: merging already
MERGE_FOR = 6 * 24                               # a merge not given by then is given up (its decision taken back)


class ProtossExecutor(Executor):
    build_order = C.ORDER_PLACE_PROTOSS_BUILDING

    def __init__(self, g: Gary):
        super().__init__(g)
        self.power_short = -10 ** 9                  # when a powered building last found no powered spot
        g.task_runners["merge"] = self._task_merge

    # --- what a decision means for Protoss hands ---------------------------------------------------

    def carry_out(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        """A command to a building (a unit, research, an upgrade), a build task for a probe, WAIT (its
        building is busy, a Pylon is coming), or None (Gary can't)."""
        g = self.g
        have = {u["type"] for u in done}
        if act == TRAIN and uid in P.MERGE:
            return self._merge_plan(uid, mine, done)
        if act == TRAIN and uid in g.race.producer:
            if me["supply_used"] + g.race.supply[uid] > me["supply_max"]:
                return self.wait("supply: a Pylon is coming") if self._pylon_coming(mine) else self.no("supply")
            b = self.free_building(obs, mine, done, g.race.producer[uid], depth=self.train_depth)
            return self._command_or_wait(b, done, g.race.producer[uid], act, uid, g.race.costs[uid])
        if act == BUILD and uid == g.race.gas_building:
            spot = self.free_geyser(obs, mine)
            return {"kind": "build", "where": spot[0], "data": {"spot": spot[1]}, "cost": g.race.costs[uid]} \
                if spot else self.no("no free geyser")
        if act == BUILD and uid in g.race.sizes and uid != g.race.town_hall:
            if uid not in UNPOWERED and P.PYLON not in have:
                return self.wait("no Pylon's power yet: one is coming") if self._pylon_coming(mine) \
                    else self.no("no Pylon's power")
            return {"kind": "build", "where": "main", "cost": g.race.costs[uid]}
        if act == EXPAND and uid == g.race.town_hall:
            base = self.free_base(obs)
            return {"kind": "build", "where": "natural" if base is g.natural else base,
                    "cost": g.race.costs[g.race.town_hall]} if base else self.no("no free base")
        if act == RESEARCH and uid in g.race.research and uid not in g.researched:
            where, cost = g.race.research[uid]
            b = self.free_building(obs, mine, done, where)
            return self._command_or_wait(b, done, where, act, uid, cost)
        if act == UPGRADE and uid in g.race.upgrade:
            where, cost, levels = g.race.upgrade[uid]
            level = g.upgraded[uid]
            if level >= levels:
                return self.no("done already")
            need = P.UPGRADE_LEVEL_2_NEEDS.get(uid)
            if level >= 1 and need is not None and need not in have:
                return self.no(f"no {P.NAMES[need]} for its next level")
            b = self.free_building(obs, mine, done, where)
            cost = (cost[0] + 75 * level, cost[1] + 75 * level)
            return self._command_or_wait(b, done, where, act, uid, cost)
        if act == RESEARCH and uid in g.researched:
            return self.no("done already")
        return self.no("not a decision Gary-P carries out")

    def _merge_plan(self, uid: int, mine: list[dict], done: list[dict]) -> dict | None:
        g = self.g
        kind = P.MERGE[uid]
        free = sorted((u for u in done if u["type"] == kind and u["order"] not in MERGING
                       and g.unselectable.get(u["tag"], 0) <= g.hi.frame), key=lambda u: u["energy"])
        if len(free) < 2:
            coming = sum(1 for u in mine if u["type"] == kind) + self.in_production(mine)[kind] >= 2
            return self.wait(f"two {P.NAMES[kind]} to merge: coming") if coming else self.no(f"no two {P.NAMES[kind]}")
        first = free[0]
        second = min(free[1:], key=lambda u: math.dist((u["x"], u["y"]), (first["x"], first["y"])))
        return {"kind": "merge", "id": uid, "tags": [first["tag"], second["tag"]], "cost": (0, 0)}

    def _task_merge(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Exactly the two Templar selected (the motor's clicks), then the merge key. Given up, the decision taken
        back, when one is gone or MERGE_FOR passed (the dataset counts the Archons that formed)."""
        g, d = self.g, t.data
        units = [u for u in mine if u["tag"] in d["tags"]]
        d.setdefault("key", ("merge",))
        if len(units) < 2 or g.hi.frame - t.started > MERGE_FOR:
            self.undecided(TRAIN, d["id"])
            g.task = None
            return
        if not g.motor._exact(obs, units, set(obs["selection"]), set(d["tags"]), t):
            if g.task is None:                           # (its clicks didn't take them)
                self.undecided(TRAIN, d["id"])
            return
        if not g.hi.merge_archon(dark=d["id"] == P.DARK_ARCHON).accepted:
            return                                       # (no APM left: again next turn)
        g.leases.take(d["tags"], Intent("MERGE", list(d["tags"]), None, {}, "hands:merge"))
        g.say(f"merge: 2 {P.NAMES[P.MERGE[d['id']]]} -> {P.NAMES[d['id']]}")
        g.task = None

    def _pylon_coming(self, mine: list[dict]) -> bool:
        g = self.g
        return any(u["type"] == P.PYLON and not u["completed"] for u in mine) or \
            any(ut == P.PYLON for ut, *_ in g.reserved) or g.building(P.PYLON)

    def supply_due(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> tuple[int, str, bool] | None:
        """A Pylon ahead of production (option depots: while each producer has less than 2 supply left, as the
        Terran's depot), or for power: a powered building found no powered spot in the last POWER_SHORT_FOR."""
        g = self.g
        if self._pylon_coming(mine) or g.macro.plan(BUILD, P.PYLON, obs, mine, done, me) in (None, WAIT):
            return None
        producers = sum(1 for u in done if u["type"] in g.race.production)
        if me["supply_max"] < 200 and me["supply_max"] - me["supply_used"] < 2 * producers:
            return BUILD, f"rule: a Pylon ahead of production ({me['supply_max'] - me['supply_used']} left, {producers} producers)", \
                False
        if g.hi.frame - self.power_short < POWER_SHORT_FOR:
            return BUILD, "rule: a Pylon for power (a building found no powered spot)", True
        return None

    def fail_build(self, t: Task, why: str = "") -> None:
        """A powered building with no spot: no Pylon's field had room for it (the game's check holds power)."""
        if t.unit_type not in UNPOWERED and why == "no spot for it":
            self.power_short = self.g.hi.frame
        super().fail_build(t, why)
