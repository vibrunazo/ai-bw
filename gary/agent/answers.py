"""Answers to the Zerg's plan (TODO 7): what Gary has seen tells which tech the Zerg went, and the
macro gets the pros' answer to lurkers as goals, in the learned macro's format (layer L2, on top of the
model, as the natural's bunker is).

The model plays a generic game: against Zerg v3's three plans (40 games) the same Engineering Bay,
turrets and Comsat whatever it faced, Siege Mode never, a Science Facility at 11:24 in two thirds
of the games. Pro Terrans (build orders, TvZ games of 12 minutes or more, the Zerg's tech by 9:00):

  against lurkers (1,321 games, Lurker Aspect at 5:42): Siege Mode at 8:42 in 96%, a Science
  Facility at 9:12 in 95%, Irradiate at 11:36; turrets 2.4 by 10:00 (Gary decided 11.7)
  against mutalisks (3,221 games, a Spire at 5:06): turrets 6.9 by 10:00, 8.7 by 12:00; a
  Science Facility at 10:12 in 83%, Irradiate at 11:24

What tells the plan: a Spire or a mutalisk seen (mutalisks); a lurker or its egg seen, or a
Hydralisk Den with a Lair and no Spire (lurkers). The Zerg's tech comes after Gary's scout has left
(~3:00), so with a Comsat and nothing known by TECH_SCAN, a scan of the Zerg's main (as pros look
at 6:00-7:00), again at TECH_RESCAN if still nothing.
"""

from __future__ import annotations

import math

from gary import terran as T
from gary.agent.gary import Gary
from gary.agent.intents import Intent

TRAIN, BUILD, RESEARCH = 0, 3, 4
SIEGE_MODE, IRRADIATE = 5, 7
LAIR, HIVE, SPIRE, GREATER_SPIRE, DEN = 132, 133, 141, 137, 135
MUTALISK, LURKER, LURKER_EGG = 43, 103, 97
TECH_SCAN = int(6.5 * 60 * 24)            # nothing known by then: a scan of their main
TECH_RESCAN = int(8.5 * 60 * 24)
SCAN_ENERGY = 50
ASK_AGAIN = 30 * 24                       # an answer's goal not carried out is asked again after this

# the answers, in order (each: what, how many in all; the macro's plan() skips what can't be made yet)
LURKER_ANSWER = [(BUILD, T.FACTORY, 1), (BUILD, T.MACHINE_SHOP, 1), (RESEARCH, SIEGE_MODE, 1),
                 (TRAIN, T.TANK, 2), (BUILD, T.STARPORT, 1), (BUILD, T.CONTROL_TOWER, 1),
                 (BUILD, T.SCIENCE_FACILITY, 1), (TRAIN, T.VESSEL, 2), (RESEARCH, IRRADIATE, 1), (TRAIN, T.TANK, 4)]
# (against mutalisks, none: the model's own turrets are already about the pros' (10.9 by 10:00, pros 6.9),
# and an answer of an Engineering Bay, turrets, a Starport, a Science Facility and vessels on top lost 3
# more of 432 (the replayed pros on Heartbreak Ridge, Zerg v3's zergling pressure that adds mutalisks)
# and won no more against the mutalisk plan (15 of 32, the same))
LURKER_TURRETS_PER_BASE = 1               # against lurkers, turrets for detection only (pros 2.4 by 10:00)


class Answers:
    def __init__(self, g: Gary):
        self.g = g
        self.seen: dict[int, int] = {}       # zerg type -> frame first seen
        self.plan: str | None = None         # "mutas", "lurkers"
        self.scans = 0
        self.asked: dict[tuple[int, int], int] = {}   # answer -> frame last asked

    def observe(self, obs: dict) -> None:
        g, f = self.g, self.g.hi.frame
        for u in obs["units"]:
            if u["owner"] not in (g.slot, 11):
                self.seen.setdefault(u["type"], f)
        plan = self.plan
        if SPIRE in self.seen or GREATER_SPIRE in self.seen or MUTALISK in self.seen:
            plan = "mutas"
        elif LURKER in self.seen or LURKER_EGG in self.seen or \
                (DEN in self.seen and (LAIR in self.seen or HIVE in self.seen)):
            plan = "lurkers"
        if plan != self.plan:
            self.plan = plan
            g.say(f"reads the zerg's plan: {plan} (seen: {', '.join(str(t) for t in sorted(self.seen))})")

    def tech_scan(self, mine: list[dict]) -> bool:
        """With nothing known by TECH_SCAN (or TECH_RESCAN): a scan of their main. True if Gary acted."""
        g, f = self.g, self.g.hi.frame
        if self.plan is not None or self.scans >= 2 or f < (TECH_SCAN, TECH_RESCAN)[self.scans]:
            return False
        main = getattr(g.army, "enemy_main", None) or (g.enemy_starts[0] if len(g.enemy_starts) == 1 else None)
        comsat = next((u for u in mine if u["type"] == T.COMSAT and u["completed"] and (u.get("energy") or 0) >= SCAN_ENERGY), None)
        if main is None or comsat is None:
            return False
        self.scans += 1
        x, y = g.map.base_near(main).center
        g.motor.start(Intent("SCAN", [], (x, y), {"comsat": comsat["tag"], "x": int(x), "y": int(y)}, "macro:answers"))
        g.say(f"scans the zerg's main for its tech ({x}, {y})")
        return True

    def allows(self, act: int, uid: int, mine: list[dict]) -> bool:
        """What the model asks that the answer rules out: against lurkers, turrets past one a base."""
        if self.plan == "lurkers" and act == BUILD and uid == T.TURRET:
            bases = sum(1 for u in mine if u["type"] == T.CC)
            return self.g.macro.count(mine, T.TURRET) < LURKER_TURRETS_PER_BASE * max(1, bases)
        return True

    def next_goal(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        """The first part of the answer Gary lacks and can make (or wait for), as a macro goal."""
        g, f, mac = self.g, self.g.hi.frame, self.g.macro
        answer = LURKER_ANSWER if self.plan == "lurkers" else []
        for act, uid, n in answer:
            have = (1 if uid in g.researched else 0) if act == RESEARCH else mac.count(mine, uid)
            if have >= n or f - self.asked.get((act, uid), -ASK_AGAIN) < ASK_AGAIN:
                continue
            plan = mac.plan(act, uid, obs, mine, done, me)
            if not isinstance(plan, dict) or plan.get("kind") == "wait":
                continue                     # (not possible, or not now: a research underway, a busy Factory)
            self.asked[(act, uid)] = f
            name = (T.TECH_NAMES.get(uid) if act == RESEARCH else T.NAMES.get(uid)) or str(uid)
            return {"act": act, "id": uid, "p": 1.0, "since": f, "name": f"{name} (against {self.plan})"}
        return None
