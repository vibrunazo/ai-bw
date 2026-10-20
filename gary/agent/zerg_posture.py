"""The Zerg's posture (docs/REFACTOR.md Phase 2): NormalPosture plus Gary-Z's alarm, as the posture contract
the Terran early-pool posture uses (posture.build, and filters the learned macro reads).

The alarm: before ALARM_UNTIL, the enemy fighters seen in the last minute, away from the enemy's town halls it
knows of (its home army isn't a threat), outweigh the Zerg's army by ALARM_MARGIN. Gary-Z's model kept droning
at 60-70% odds with 20 marines in sight and 6 army at 6:00-7:30, and Gary-T won 7 of 8 by a push into 0-6
army: pros in that state make zerglings and sunkens, and their data holds few such states. While it is on:

  build "alarm"   a sunken at the natural per 8 supply of threat (ALARM_SUNKENS at most: a creep colony,
                  then its sunken), else the likeliest army unit the model gives that can be made now; when
                  none can, the learned macro goes on, with
  no_workers      no drones
  one_off_min_p   tech, expansions, research and upgrades only at ALARM_MIN_P odds or more
"""

from __future__ import annotations

import math

from gary import zerg as Z
from gary.agent.army_rules import EnemyMemory
from gary.agent.gary import Gary
from gary.agent.macro import BMORPH, BUILD, MORPH, WAIT, describe
from gary.agent.posture import NormalPosture
from gary.decision_log import LOG

ALARM_UNTIL = 12 * 60 * 24
ALARM_MARGIN = 16                # x2: 8 supply
ALARM_SUNKENS = 3
ALARM_HOME = 16 * 32
ALARM_MIN_P = 0.30


class ZergPosture(NormalPosture):
    def __init__(self, g: Gary):
        super().__init__(g)
        self.memory = EnemyMemory(g)
        self.alarm = False
        self.threat = 0
        self.no_workers = False
        self.one_off_min_p: float | None = None

    def update(self) -> None:
        from gary.policy.army import supply_x2
        g = self.g
        if g.enemy is None:
            return
        obs = g.hi.observe()
        self.memory.look(obs)
        halls = [(x, y) for x, y, t in self.memory.buildings.values() if t in g.enemy.town_halls]
        self.threat = sum(v[1] for v in self.memory.fighters.values()
                          if not any(math.dist(v[2:4], h) < ALARM_HOME for h in halls))
        own = sum(supply_x2(u["type"]) for u in obs["units"] if u["owner"] == g.slot and u["type"] in Z.ARMY and u["completed"])
        alarm = g.hi.frame < ALARM_UNTIL and self.threat >= own + ALARM_MARGIN
        if alarm != self.alarm:
            self.alarm = alarm
            g.say(f"alarm {'on' if alarm else 'off'}: {self.threat / 2:.0f} enemy supply seen, {own / 2:.0f} army")
            LOG.decision(g.hi.side, "posture", g.hi.frame, "alarm on" if alarm else "alarm off",
                         inputs={"threat": self.threat / 2, "army": own / 2},
                         by=f"rule: the enemy fighters seen in a minute outweigh the army by {ALARM_MARGIN // 2}")
        self.build = "alarm" if alarm else None
        self.no_workers = alarm
        self.one_off_min_p = ALARM_MIN_P if alarm else None


class AlarmDefense:
    """The alarm's build: sunkens at the natural, then army units; nothing to do now hands back to the learned
    macro (falls_through), under the posture's filters."""
    label = "alarm"
    falls_through = True

    def __init__(self, g: Gary):
        self.g = g

    def first(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> bool:
        return False

    def next_goal(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        g = self.g
        mac = g.macro
        threat = getattr(g.posture, "threat", 0)
        want = min(ALARM_SUNKENS, threat // 16 + 1)
        sunkens = sum(1 for u in mine if u["type"] == Z.SUNKEN) + mac.in_production(mine)[Z.SUNKEN]
        creeps_done = [u for u in done if u["type"] == Z.CREEP]
        creep_on_way = any(u["type"] == Z.CREEP and not u["completed"] for u in mine) or \
            any(ut == Z.CREEP for ut, *_ in g.reserved) or g.building(Z.CREEP)
        steps = []
        if creeps_done and sunkens < want:
            steps.append((BMORPH, Z.SUNKEN))
        if not creep_on_way and sunkens + len(creeps_done) < want:
            steps.append((BUILD, Z.CREEP))
        x = mac.tracker.features(obs, mac.in_production(mine))
        probs, _ = mac.model.probs(x)
        decisions = mac.model.spec.decisions
        army = sorted((k for k, (a, u) in enumerate(decisions) if a == MORPH and u in (Z.ZERGLING, Z.HYDRALISK, Z.MUTALISK)),
                      key=lambda k: -probs[0, k])
        steps += [decisions[k] for k in army]
        for act, uid in steps:
            plan = mac.plan(act, uid, obs, mine, done, me)
            if plan is None or plan is WAIT:
                continue
            k = decisions.index((act, uid)) if (act, uid) in decisions else None
            return {"act": act, "id": uid, "p": float(probs[0, k]) if k is not None else 1.0, "since": g.hi.frame,
                    "name": describe(g.race, act, uid)}
        return None
