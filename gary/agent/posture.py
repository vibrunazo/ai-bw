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
                 with one hatchery known, zerglings before 4:00, or 6 at once before 5:00; or
                 assumed when no zerg building has been seen by 2:20 (on a 4-player map the scout
                 can still be on its way: against a pro-timed 9-pool, 8 of v0.11's 24 losses). Then
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
from gary.policy.army import is_building, supply_x2


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
UNKNOWN_BY = 140 * 24                # ...and with no Zerg building seen by 2:20, play it as one (the
                                     # scout is still looking: a 9-pool's zerglings come at 3:00; at
                                     # 2:45, 2 of 48 9-pools were seen too late)
POOL_DONE_BY = 170 * 24              # a finished pool seen by 2:50, or zerglings by 3:00 (POOL_BY): an early pool
LATE_CALL_OFF_BY = 240 * 24          # v5: an alarm from nothing seen is called off on two hatcheries until 4:00
RELEASE_PX = 12 * 32                 # v5: bunkers this close to the main's Command Center are let out
POOL_SURE_BY = 160 * 24              # v3: ...by 2:40: one even with a second hatchery
POOL_EARLY = 120 * 24                # a pool seen by 2:00 is an early pool (hatch firsts' pools are
                                     # ordered at 2:05, median); one seen later only with one hatchery
                                     # once the scout has passed the Zerg's natural
RUSH_MARINES = 4                     # in or at the finished bunker: the rush build can end...
CALM = 20 * 24                       # ...once no zerg fighter was at home for this long
HOLD_ARMY, HOLD_RATIO = 6, 1.5       # Gary holds until its army supply is this, and this many times
                                     # the biggest zerg army seen at once
DEFENSE = {T.SCV, T.MARINE, T.DEPOT, T.RAX, T.BUNKER}
BUNKER_FROM_CC = 3 * 32              # v2: the rush bunker this far from the main's CC, toward its way out
BUNKER_INSIDE = 64                   # the rush bunker: this far inside the main's way out
SWITCH_FROM, SWITCH_UNTIL = 110 * 24, 250 * 24   # the learned switch is asked from 1:50 to 4:10...
SWITCH_EVERY = 10 * 24                           # ...every 10 s


class EarlyPoolPosture:
    bunker_by_cc = False                 # v2: the rush bunker by the main's Command Center, not its way out
    pool_sure = False                    # v3: a pool finished by 2:40 is an early pool, natural or not
    late_call_off = False                # v5: an alarm from nothing seen is called off until 4:00
    release_bunkers = False              # v5: ...and when the rush is over, the main's bunker is let out
    unknown_alarm = False
    unknown_by = UNKNOWN_BY              # v4: nothing seen by then is an early pool (v4: 2:50)
    force_at: int | None = None          # training (train/early_switch.py): the rush defense from this frame
    switch = None                        # a learned switch: facts (gary/policy/plan.py) -> go into the rush
                                         # defense now; asked every SWITCH_EVERY from SWITCH_FROM to SWITCH_UNTIL

    def __init__(self, g: Gary):
        self.g = g
        self.rush = False                # an early pool was seen: the rush build
        self.hold = False                # ...and the army stays home
        self.rush_over = False
        self.hatcheries_seen: set[int] = set()
        self.most_enemy = 0.0            # the biggest zerg army (supply) seen at once
        self.base_seen = False           # any zerg building seen
        self.pool_seen = False           # a Spawning Pool seen (the scout saw it, then left)
        self.early_pool_sure = False     # v3: a finished pool by 2:40, or zerglings by 3:00
        self.early_pool_seen = False     # a finished pool by 2:50, or zerglings by 3:00: early, whatever else
        way_out = self._main_way_out() or (    # (no choke: between the main and its natural)
            int(g.main.center[0] * 0.6 + g.natural.center[0] * 0.4), int(g.main.center[1] * 0.6 + g.natural.center[1] * 0.4))
        if self.bunker_by_cc:
            # against a fast pool, pros' one-base bunker stands by the Command Center (60 of 88 by
            # 3:30; 18 at the main's way out): marines in it cover the mineral line, SCVs repair it
            # from where they mine (Gary's at the way out, 16-20 tiles off, fell to 10 zerglings)
            d = math.dist(way_out, g.main.center) or 1
            self.bunker_spot = (int(g.main.center[0] + (way_out[0] - g.main.center[0]) / d * BUNKER_FROM_CC),
                                int(g.main.center[1] + (way_out[1] - g.main.center[1]) / d * BUNKER_FROM_CC))
        else:
            self.bunker_spot = way_out

    def _main_way_out(self):
        """Where the rush bunker goes: BUNKER_INSIDE inside the main's way out (the frontier of the
        main's plateau, from map knowledge: right on any map, where the ramp analysis can misread
        one: Heartbreak Ridge), or None if the main has no choke."""
        from gary.walls import frontier
        g = self.g
        if not g.enemy_starts or getattr(g.hi.game, "terrain", None) is None:
            return None
        full = g.hi.game.observe()
        fr = frontier(g.hi.game, {**full, "units": [u for u in full["units"] if u["owner"] == 11]},
                      [g.main.center], list(g.enemy_starts), level=True)
        if not fr["openings"] or fr["open"]:
            return None
        o = min(fr["openings"], key=lambda o: math.dist((o.x, o.y), g.main.center))
        d = math.dist(o.inside, o.outside) or 1
        return (int(o.x + (o.inside[0] - o.outside[0]) / d * BUNKER_INSIDE),
                int(o.y + (o.inside[1] - o.outside[1]) / d * BUNKER_INSIDE))

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
        # an early pool even if it expands too (a 12-pool and a hatchery: two hatcheries seen):
        # a pool finished by 2:50 was started by 2:00 (50 s to build; hatch firsts' at 2:05), and
        # zerglings by 3:00
        if (hi.frame <= POOL_DONE_BY and any(u["type"] == POOL and u["completed"] for u in enemy)) or                 (hi.frame <= POOL_BY and any(u["type"] == LING for u in enemy)):
            self.early_pool_seen = True
        # v3: ...and sure, with a natural or not, if finished by 2:40 (pro TvZ: 98% of pool firsts'
        # pools, 0.7% of hatch firsts'; by 2:50 16% of hatch firsts') or zerglings by 3:00
        if self.pool_sure and ((hi.frame <= POOL_SURE_BY and any(u["type"] == POOL and u["completed"] for u in enemy))
                               or (hi.frame <= POOL_BY and any(u["type"] == LING for u in enemy))):
            self.early_pool_sure = True
        self.base_seen = self.base_seen or any(is_building(u["type"]) for u in enemy)
        self.pool_seen = self.pool_seen or any(u["type"] == POOL for u in enemy)
        lings = sum(1 for u in enemy if u["type"] == LING)
        fighters = enemy_fighters(obs, g.slot)
        self.most_enemy = max(self.most_enemy, sum(supply_x2(u["type"]) for u in fighters) / 2)
        # a pool with one hatchery is an early pool if seen by 2:00, later once the scout has passed
        # the Zerg's natural (a hatch first's pool is there by 2:05 too: judged at the main alone,
        # 9 of 29 hatch firsts on Python looked like early pools; waiting for the natural always,
        # 9-pools were seen at 3:04 instead of 1:55 and 7 of 48 lost)
        looked = getattr(g.army, "scout_done", True) or hi.frame <= POOL_EARLY
        # (zerglings count only while Gary hasn't seen a second hatchery: a hatch first's first ones
        # come at 3:20-5:00, and called an early pool they held back Gary's tech by a minute)
        one_base = len(self.hatcheries_seen) <= 1
        # (the pool remembered: the scout sees it while circling the main, and the scouting is done
        # only once it has passed the natural, out of sight of the pool)
        evidence = ((hi.frame <= POOL_BY and self.pool_seen and one_base and looked) or
                    (hi.frame <= POOL_BY and self.early_pool_sure) or
                    (hi.frame <= LINGS_BY and lings and one_base) or
                    (hi.frame <= MANY_LINGS_BY and lings >= MANY_LINGS and one_base))
        unknown = self.unknown_by <= hi.frame <= POOL_BY and not self.base_seen
        if not self.rush and not self.hold and self._switched(obs):
            evidence = self.early_pool_seen = True        # (decided: not called off by a second hatchery)
        if not self.rush and not self.hold and (evidence or unknown):
            self.rush = self.hold = True
            self.unknown_alarm = not evidence    # only nothing seen yet: maybe a hatch first after all
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
            late = self.late_call_off and self.unknown_alarm and hi.frame <= LATE_CALL_OFF_BY
            if (self.rush or late) and len(self.hatcheries_seen) >= 2 and (hi.frame <= POOL_BY or late)                     and not near_home(mine, fighters) and not self.early_pool_seen:
                # a second hatchery by 3:00, no zerg at home: a hatch first (the alarm came from not
                # having seen the base yet: the scout of a four-player map). Back to the usual plan
                # (in 116 of 144 games against replayed pros Gary had played rush defense, its tech a
                # minute late; the rush build held until 6 army and 20 s of calm, often 4:15)
                self.rush = self.hold = False
                self.rush_over = late
                g.say("two hatcheries, no zerg at home: not an early pool, back to the usual plan")
                self._release(mine)
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
                self._release(mine)

    def _switched(self, obs: dict) -> bool:
        """The rush defense decided outside the rules: forced from force_at (training), or by the
        learned switch, asked every SWITCH_EVERY in its window (before the rush was ever over)."""
        f = self.g.hi.frame
        if self.rush_over:
            return False
        if self.force_at is not None and f >= self.force_at:
            self.force_at = None
            return True
        if self.switch is None or not SWITCH_FROM <= f <= SWITCH_UNTIL or f < getattr(self, "next_switch", 0):
            return False
        self.next_switch = f + SWITCH_EVERY
        from gary.policy.plan import facts
        return bool(self.switch(facts(self, obs)))

    def _release(self, mine: list[dict]) -> None:
        """v5: the rush is over: the bunkers by the main's Command Center let their marines out and
        aren't kept manned any more (4 marines sat in one while zerglings took the ramp)."""
        if not self.release_bunkers:
            return
        g = self.g
        main = g.main.center
        for u in mine:
            if u["type"] == T.BUNKER and math.dist((u["x"], u["y"]), main) < RELEASE_PX:
                g.release_bunkers.add(u["tag"])


class EarlyPoolPostureV2(EarlyPoolPosture):
    """v1 with the rush bunker by the main's Command Center (where pros put theirs against a fast pool).
    It doesn't hold the main's choke: the rush dummy's 7:00 wave of 26 zerglings went through in 3
    of 48 games. A second bunker at the way out once the rush is held, while Gary stays home, saved
    2 of those but cost more against the replayed pros (8 lost of 144, 3 without): not kept."""
    bunker_by_cc = True


class EarlyPoolPostureV4(EarlyPoolPostureV2):
    """v3 that calls nothing seen an early pool only at 2:50, not 2:20: pros don't see the Zerg by
    2:20 on 4-player maps either (Terran pros first see a Zerg building at 2:36 on Fighting Spirit,
    median; 2:59 at the 75th percentile), and v3 played rush defense against 19 of 30 hatch firsts
    there. Early pools are caught otherwise: the pool remembered, finished by 2:40, zerglings."""
    pool_sure = True
    unknown_by = 170 * 24


class EarlyPoolPostureV5(EarlyPoolPostureV4):
    """v4 whose alarm from nothing seen yet is called off on two hatcheries until 4:00 (not 3:00: the
    scout of a four-player map found a hatch first at 3:13 and Gary stayed in rush defense), and
    whose main's bunker lets its marines out once the rush is over (4 marines sat in it while
    zerglings took the ramp)."""
    late_call_off = True
    release_bunkers = True


FLOOD_FROM, FLOOD_UNTIL = 210 * 24, 8 * 60 * 24   # v6: a zergling mass seen between 3:30 and 8:00...
FLOOD_LINGS = 6                      # ...this many zerglings in sight at once (at least; pros' Terrans
                                     # see 6 by 5:00 in a median game: the army ratio tells a flood)...
FLOOD_RATIO = 1.5                    # ...outweighing Gary's army (supply) this many times
FLOOD_MEMORY = 90 * 24               # ...the most seen at once over this long
FLOOD_OVER = 1.5                     # it's over: Gary's army this many times that, and calm at home for CALM


class EarlyPoolPostureV6(EarlyPoolPostureV5):
    """v5 that also answers a zergling mass on its way (Zerg v2's all-in; its zerglings gather in
    front of Gary's natural): FLOOD_LINGS+ zerglings in sight at once, outweighing Gary's army
    FLOOD_RATIO times, between 3:30 and 8:00 (what the watching scout sees, army.py watch). Then the
    ling_flood_defense build (macro.py), no expansion, the army home, bunkers manned; until Gary's
    army outweighs the most seen over 90 s FLOOD_OVER times and its bases are calm. A stand-in for a
    strategy layer that picks a proven build when it sees one coming."""

    def __init__(self, g: Gary):
        super().__init__(g)
        self.flood = False
        self.ling_marks: list[tuple[int, int]] = []     # (frame, zerglings in sight)

    @property
    def build(self):
        return "ling_flood_defense" if self.flood and not self.rush else super().build

    @property
    def allowed(self):
        return None if self.flood and not self.rush else super().allowed

    @property
    def no_expand(self) -> bool:
        return self.flood or super().no_expand

    @property
    def army_home(self) -> bool:
        return self.flood or super().army_home

    @property
    def preload_bunkers(self) -> bool:
        return self.flood or super().preload_bunkers

    def update(self) -> None:
        super().update()
        g = self.g
        hi = g.hi
        if hi.pending:
            return
        obs = hi.observe()
        f = hi.frame
        lings = sum(1 for u in obs["units"] if u["owner"] not in (g.slot, 11) and u["type"] == LING)
        self.ling_marks = [(t, n) for t, n in self.ling_marks if f - t <= FLOOD_MEMORY] + [(f, lings)]
        most = max(n for _, n in self.ling_marks)
        mine = [u for u in obs["units"] if u["owner"] == g.slot]
        tags = {u["tag"] for u in mine}
        inside = set().union(set(), *(m for b, m in g.loaded.items() if b in tags)) - tags
        army = sum(supply_x2(u["type"]) for u in mine if u["type"] in g.army_types) / 2 + len(inside)
        if not self.flood:
            if FLOOD_FROM <= f <= FLOOD_UNTIL and most >= FLOOD_LINGS and most / 2 >= FLOOD_RATIO * army:
                self.flood = True
                g.say(f"zergling flood: {most} zerglings seen, army {army:.0f}: barracks, academy, stim; home")
        else:
            if near_home(mine, enemy_fighters(obs, g.slot)):
                g.calm_since = f
            if army >= FLOOD_OVER * most / 2 and f - g.calm_since >= CALM:
                self.flood = False
                g.say(f"the zergling flood is held: army {army:.0f} vs {most} zerglings, back to the usual plan")


class EarlyPoolPostureV3(EarlyPoolPostureV2):
    """v2 that also calls a pool first that takes its natural an early pool: a pool finished by 2:40
    (on Heartbreak Ridge, pools at 1:36 with a hatchery at the natural too: the zerglings came at 3:00
    to Gary's 1 marine)."""
    pool_sure = True


class LearnedPosture(EarlyPoolPostureV6):
    """The posture's mode from the plan policy (gary/policy/plan.py: normal, rush, hold, flood),
    asked every second; the rules (v6) keep the memory it decides on (what was seen, the calm,
    the most zerglings and army seen) and, with no model, the mode itself. record (a list): the
    facts and the mode the rules chose, every second (train/plan.py distills the policy from it).
    temperature > 0 draws the mode (training); every decision is logged with its probability."""

    def __init__(self, g: Gary, model=None, temperature: float = 0.0, record: list | None = None, seed: int = 0):
        import numpy as np
        super().__init__(g)
        self.model = model
        self.temperature = temperature
        self.record = record
        self.rng = np.random.default_rng(seed)
        self.mode = "normal"
        self.next_decide = 0
        self.decisions: list[dict] = []         # (frame, facts vector, mode, probability): training

    def _rules_mode(self) -> str:
        return "flood" if self.flood and not self.rush else "rush" if self.rush else "hold" if self.hold else "normal"

    def _set(self, mode: str) -> None:
        g = self.g
        before = self.mode
        self.rush, self.hold, self.flood = mode == "rush", mode in ("rush", "hold"), mode == "flood"
        self.mode = mode
        if mode == before:
            return
        g.say(f"plan: {before} -> {mode}")
        if mode == "rush":                       # (as the rules: marines gather behind the bunker)
            bx, by = self.bunker_spot
            mx, my = g.main.center
            d = math.dist((bx, by), (mx, my)) or 1
            g.rally = (int(bx + (mx - bx) / d * 4 * 32), int(by + (my - by) / d * 4 * 32))
        if before in ("rush", "hold") and mode == "normal":     # (as the rules: the rush is over)
            self._release([u for u in g.hi.observe()["units"] if u["owner"] == g.slot])

    def update(self) -> None:
        from gary.policy.plan import decide, facts, vector
        g = self.g
        hi = g.hi
        prev = self.mode
        if self.model is not None:               # (the rules see the mode the policy set)
            self.rush, self.hold, self.flood = prev == "rush", prev in ("rush", "hold"), prev == "flood"
        super().update()
        rules = self._rules_mode()
        # asked every second, and at once when the rules would change mode (so that, following them,
        # it changes when they do)
        if hi.pending or (hi.frame < self.next_decide and rules == prev):
            if self.model is not None:
                self.rush, self.hold, self.flood = prev == "rush", prev in ("rush", "hold"), prev == "flood"
            else:
                self.mode = rules
            return
        self.next_decide = hi.frame + 24
        x = vector(facts(self, hi.observe()), prev, rules)
        if self.record is not None:
            self.record.append((x, rules))
        if self.model is None:
            self.mode = rules
            return
        mode, p = decide(self.model, x, self.rng, self.temperature)
        self.decisions.append({"frame": hi.frame, "x": x, "mode": mode, "p": p})
        self._set(mode)
