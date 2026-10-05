"""The posture (layer L1, docs/CONTROL.md): the plan's mode, and what it asks of the other layers.

Every provider exposes the same constraints, which the other layers read:

  allowed          unit types the macro may train or build (None: anything); nothing else is
                   planned, no research or expansion either
  no_expand        no expansion
  build            a scripted build the macro switches to (by name), or None for the learned macro
  army_home        the army model doesn't send the army anywhere
  preload_bunkers  marines go into bunkers before the enemy is there

  normal         the usual plan: no constraints
  early_pool/v1  v0.9's answer to an early Spawning Pool (#2): noticed from a pool before 3:00
                 with one hatchery known, zerglings before 4:00, or 6 at once before 5:00. Then
                 the one-base defense build, with a bunker on the main's level by the top of its
                 ramp (gary/terrain.py), marines rallied behind it, until the bunker stands with 4
                 marines and no zerglings are at home; then the learned macro again, but the army
                 stays home (no expansion, bunkers loaded early) until it's 1.5x the biggest zerg
                 army seen
"""

from __future__ import annotations

import math

from gary import terran as T
from gary.agent.fight import enemy_fighters, near_home
from gary.agent.gary import Gary
from gary.policy.army import supply_x2
from gary.terrain import map_analysis


class NormalPosture:
    allowed = None
    no_expand = False
    build = None
    army_home = False
    preload_bunkers = False

    def __init__(self, g: Gary):
        self.g = g

    def update(self) -> None:
        pass


POOL, LING = 142, 37
HATCHERIES = {131, 132, 133}
POOL_BY = 3 * 60 * 24                # a Spawning Pool seen before 3:00 with one hatchery is an early pool...
LINGS_BY = 4 * 60 * 24               # ...and so are zerglings before 4:00
MANY_LINGS, MANY_LINGS_BY = 6, 5 * 60 * 24   # ...or this many at once before 5:00
RUSH_MARINES = 4                     # in or at the finished bunker: the rush build can end...
CALM = 20 * 24                       # ...once no zerg fighter was at home for this long
HOLD_ARMY, HOLD_RATIO = 6, 1.5       # Gary holds until its army supply is this, and this many times
                                     # the biggest zerg army seen at once
DEFENSE = {T.SCV, T.MARINE, T.DEPOT, T.RAX, T.BUNKER}


class EarlyPoolPosture:
    def __init__(self, g: Gary):
        self.g = g
        self.rush = False                # an early pool was seen: the rush build
        self.hold = False                # ...and the army stays home
        self.rush_over = False
        self.hatcheries_seen: set[int] = set()
        self.most_enemy = 0.0            # the biggest zerg army (supply) seen at once
        analysis = map_analysis(g.hi.game, g.map)["starts"]
        mine = min(analysis.values(), key=lambda a: math.dist(a["start"], g.main.center), default=None)
        if mine and mine["reliable"] and math.dist(mine["start"], g.main.center) < 64:
            # the bunker goes just inside the main, a little up from the ramp's top
            top, bottom = mine["ramp_top"], mine["ramp_bottom"]
            d = math.dist(top, bottom) or 1
            self.bunker_spot = (int(top[0] + (top[0] - bottom[0]) / d * 48), int(top[1] + (top[1] - bottom[1]) / d * 48))
        else:                            # no trusted ramp: between the main and its natural
            self.bunker_spot = (int(g.main.center[0] * 0.6 + g.natural.center[0] * 0.4),
                                int(g.main.center[1] * 0.6 + g.natural.center[1] * 0.4))

    @property
    def allowed(self):
        return DEFENSE if self.rush else None

    @property
    def no_expand(self) -> bool:
        return self.hold

    @property
    def build(self):
        return "one_base_defense" if self.rush else None

    @property
    def army_home(self) -> bool:
        return self.hold

    @property
    def preload_bunkers(self) -> bool:
        return self.hold

    def update(self) -> None:
        """Notice the rush, and when it's over."""
        g = self.g
        hi = g.hi
        if hi.pending or self.rush_over:
            return
        obs = hi.observe()
        enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11)]
        self.hatcheries_seen |= {u["tag"] for u in enemy if u["type"] in HATCHERIES}
        lings = sum(1 for u in enemy if u["type"] == LING)
        fighters = enemy_fighters(obs, g.slot)
        self.most_enemy = max(self.most_enemy, sum(supply_x2(u["type"]) for u in fighters) / 2)
        if not self.rush and not self.hold and ((hi.frame <= POOL_BY and any(u["type"] == POOL for u in enemy)
                                                 and len(self.hatcheries_seen) <= 1) or
                                                (hi.frame <= LINGS_BY and lings) or
                                                (hi.frame <= MANY_LINGS_BY and lings >= MANY_LINGS)):
            self.rush = self.hold = True
            # marines gather a few tiles behind the bunker, toward the main (not on its spot,
            # where they'd keep the SCV from placing it)
            bx, by = self.bunker_spot
            mx, my = g.main.center
            d = math.dist((bx, by), (mx, my)) or 1
            g.rally = (int(bx + (mx - bx) / d * 4 * 32), int(by + (my - by) / d * 4 * 32))
            g.say("early pool: bunker at the ramp, marines first, no expansion or tech yet")
        if self.hold:
            mine = [u for u in obs["units"] if u["owner"] == g.slot]
            if near_home(mine, fighters):
                g.calm_since = hi.frame
            bunker = any(u["type"] == T.BUNKER and u["completed"] for u in mine)
            tags = {u["tag"] for u in mine}
            inside = set().union(set(), *(m for b, m in g.loaded.items() if b in tags)) - tags
            marines = sum(1 for u in mine if u["type"] == T.MARINE) + len(inside)   # the ones in the bunker aren't seen
            army = sum(supply_x2(u["type"]) for u in mine if u["type"] in g.army_types) / 2 + len(inside)
            if self.rush and bunker and marines >= RUSH_MARINES and hi.frame - g.calm_since >= CALM:
                self.rush = False
                g.say(f"bunker up with {marines} marines, no zerglings at home: gas and tech again, still home")
            if not self.rush and army >= max(HOLD_ARMY, HOLD_RATIO * self.most_enemy):
                self.hold, self.rush_over = False, True
                g.say(f"the early pool is held: army {army:.0f} vs the zerg's {self.most_enemy:.0f} at most, back to the usual plan")
