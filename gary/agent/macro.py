"""Macro decisions (layer L2, docs/CONTROL.md): what to build, train, research next.

  learned           the macro model (gary/policy/macro.py, ~14,000 pro TvZ games; Gary v0.2+):
                    whenever Gary is free to decide, the likeliest decision it can carry out now
                    (requirements built, a building free, a base free), then wait for the money.
                    Steerable by build style. What the posture forbids (L1) it never plans.
  one_base_defense  a scripted build a posture can switch to (v0.9's answer to an early pool): a
                    bunker at the ramp first, then depots ahead of production, marines and SCVs
                    nonstop, a second Barracks on spare money. It gives goals in the learned
                    macro's format, carried out the same way.
  ling_flood_defense  another (posture v6's answer to a zergling mass on its way): Barracks to 3, an
                    Academy and Stim, medics and firebats among the marines, depots ahead; SCVs
                    from what's left. (A stand-in for a strategy layer that picks a proven build:
                    the learned macro, seeing the zerglings, makes marines but adds no Barracks
                    and no Academy.)

The executor (L2b) carries the decisions out: a build task or a command task.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from gary import terran as T
from gary.agent.gary import Gary, Task
from gary.decision_log import LOG, top
from gary.policy.attribution import explain_macro
from gary.policy.macro import ACT_NAMES, MacroModel, MacroTracker

TRAIN, MORPH, BMORPH, BUILD, RESEARCH, UPGRADE, EXPAND = range(7)   # (ACT_NAMES' order)
TOWN_HALLS = {106, 131, 132, 133, 154}
WALK_AHEAD = 6                  # seconds: a building's worker sets out this much income before the money is there
MINING = 1.0                    # minerals a second an SCV mines (about)
GOAL_TIMEOUT = 24 * 30          # a decision Gary can't afford or do within 30 s is dropped
BANK_ESCAPE = 400               # minerals over which the waiting favorite no longer holds the macro (bank_escape)
DICE_FLOOR = 0.01               # decoder "dice": decisions less likely than this are never rolled
HOLD_FOR = 30 * 24              # option hold: a roll waited for this long at most
MIN_P = 0.05                    # never take a decision the model gives less than this: unlikely
                                # choices lead into states pros rarely reach, where it knows less
WAIT = {"kind": "wait"}         # plan(): possible, but not right now
ONE_OFF_BUILDINGS = {T.REFINERY, T.ACADEMY, T.EBAY, T.FACTORY, T.STARPORT, T.ARMORY, T.SCIENCE_FACILITY, T.COMSAT,
                     T.MACHINE_SHOP, T.CONTROL_TOWER, T.COVERT_OPS, T.PHYSICS_LAB}   # odds_mass: built once (or so)
# Zerg's one-offs (Gary-Z's mass): a tech building, Lair / Hive / Greater Spire, research, an upgrade, a hatchery
# or an expansion; and the buildings that open a tech path (option tech_min_p)
Z_ONE_OFF_BUILDINGS = {135, 141, 138, 136, 140, 139, 131}    # Den, Spire, Queen's Nest, Defiler Mound, Ultralisk Cavern,
                                                             # Evolution Chamber, Hatchery
Z_ONE_OFF_MORPHS = {132, 133, 137}                           # Lair, Hive, Greater Spire
TECH_PATH = {"Z": {135, 141, 138, 136, 140}}
FIRST_OPENING = 6 * 60 * 24            # until then the first producer's rule saves for one it can't build yet
FIRST_PRODUCER = {"T": T.RAX, "Z": 142}  # the building that makes the army first: by rax_by (option), else one is built
PACE_MAX = 10 * 24              # option pace: the longest wait for the next decision
QUERY_EVERY = 12                # frames between the macro's looks at the model
PARK_AHEAD = 100                # option park: a building's worker sets out this many minerals short of its cost
NAT_BUNKER_ALONG = 0.45         # the natural's bunker: this far from its Command Center toward its way out
NAT_BUNKER_TILES = 5            # ...or this far toward the enemy where the natural has no choke
NAT_BUNKER_RETRY = 20 * 24


def describe(race, act: int, uid: int) -> str:
    """A decision's name in this race's (gary/race.py) words."""
    if act == TRAIN:
        return race.names.get(uid, f"unit {uid}")
    if act == RESEARCH:
        return race.tech_names.get(uid, f"tech {uid}")
    if act == UPGRADE:
        return race.upgrade_names.get(uid, f"upgrade {uid}")
    if act == EXPAND:
        return "expand"
    return race.names.get(uid, f"{ACT_NAMES[act]} {uid}")


class LearnedMacro:
    def __init__(self, g: Gary, model: MacroModel, style: int | None = None, builds: dict | None = None,
                 natural_bunker: bool = False, rax_by: int = 90 * 24,
                 decoder: str = "favorite", seed: int = 0, temperature: float = 1.0, park: bool = False,
                 pace: bool = False, tech_min_p: float | None = None, hatch_mass: float | None = None,
                 wait_min_p: float | None = None):
        self.g = g
        self.temperature = temperature                      # decoder "sample": the draw's temperature
        self.dice_t = 1.0                                   # decoder "dice": its odds raised to 1 / this (option dice07)
        self.draws: list[dict] = []                         # decoder "sample", or log: every decision (training)
        self.log = False                                    # training: log the favorite's decisions too
        self.force: tuple[int, int] | None = None           # training: (frame, decision) taken instead there
        self.rax_by = rax_by
        # option (depots): a depot ahead of production, as the scripted builds do: started while each producer has
        # less than 2 supply left. Pros are almost never supply-blocked, so the model has hardly seen the state
        # and gives the depot under its floor once there: Gary-T was blocked in 42 of 48 games at 4-8 minutes, 437
        # of its 749 decisions while blocked "nothing it can do" (eval/why.py records, L5b)
        self.supply_ahead = False                           # (option depots; for Zerg an overlord: g.executor.supply_due)
        # option park (Gary-Z's): a decision the money doesn't cover yet, other than a larva's, is parked with its
        # money kept, and the larvae go on meanwhile (a decision at a time held every larva: 8 drones at 2:00)
        self.park = park
        self.saved: dict | None = None                      # the parked decision
        self.why_not: str | None = None                     # why plan's last None or WAIT (the records say it)
        # option pace (Gary-Z's): the next look at the model when the pros' next decision would come (the
        # model's timing head), a larva's decisions in bursts (pros spend a hatchery's larvae at once)
        self.pace = pace
        self.tech_min_p = tech_min_p                        # option: a tech path opened only on odds of this or more
        self.hatch_mass = hatch_mass                        # option: a town hall's odds summed to this (else odds_mass)
        self.wait_min_p = wait_min_p                        # option: while a decision waits, others at these odds
        # option hold (decoder dice): a roll that waits for supply or for its money (a larva's) stays the roll at the
        # next looks, and nothing else is taken meanwhile: the model's odds are of the pros' next decision, and rolling
        # again at every look (or taking the next doable one at 5%, wait_min_p) took a 7% Spawning Pool at 1:27 while
        # the drone waited for the overlord, nearly every game; the pros' natural at 1:38, gz1's at 2:06, and 13
        # drones at 4:00 to their 16 (round O6). A roll waiting for a larva still lets others by (wait_min_p): held
        # there too, the tech came late (a Lair by 7:00 in half the games, the pros' all; round G10)
        self.hold = False
        self.held: tuple[int, int] | None = None            # (the decision rolled and waited for, since)
        # units the model decided whose producer is busy, since when: up to one per producer of their kind (several
        # Marines wait for several Barracks), the macro deciding on meanwhile; trained as producers free up, and nothing
        # the model didn't decide (keep_busy, which trained idle producers' last unit again, made half of gt1's marines
        # at 10-20 minutes; removed 2026-10-23: rounds Q60a against B60a, the same results and gap without it)
        self.queued: list[tuple[tuple[int, int], int]] = []
        self.queue_depth = 1                  # the model's waiting units queued per producer (option qdepth2: 2)
        # option rethink: a unit decided whose producers' queues are full isn't held as the goal (the model wasn't
        # asked again while it waited, so it never added production with the bank: the bank escape covered for that);
        # the model is asked again at its next look
        self.rethink = False
        # option nosave: a goal the money doesn't cover yet (even with its builder walking ahead) isn't held either;
        # the model is asked again at its next look (with dice, gt1dev's drawn buildings held the macro while they
        # saved, the Command Center idle: 18.8 SCVs at 5:00, the pros' 25.3)
        self.nosave = False
        # STOP-GAP (AGENTS.md): with more than BANK_ESCAPE minerals, anything Gary can do now is taken while the
        # model's favorite waits (option nobank: off; the model reads its minerals)
        self.bank_escape = True
        self.caps_on = True                   # the pros' caps per minute (AGENTS.md register; option nocaps: off)
        self.decoder = decoder                              # "favorite" (the most likely), "dice" or "sample"
        self.rng = random.Random(seed)
        self.natural_bunker = natural_bunker
        self.next_nat_bunker = 0
        self._nat_bunker_spot = None
        self.model = model
        self.tracker = MacroTracker(model.spec, g.slot, style, getattr(g.map, "hash", None), g.matchup)
        self.goal: dict | None = None
        self.next_query = 0
        self.decisions: list[tuple[int, str, float]] = []   # (frame, decision, model probability)
        self.failed_until: dict[tuple[int, int], int] = {}  # decision -> frame it may be tried again
        self.builds = builds or {}                         # posture's build name -> scripted build
        # expand_mass: the model's odds of expanding summed over the decisions Gary took since its last
        # expansion; at this much, the next decision is the expansion (None: off). The favorite never
        # is: after the natural the model gives it 2-4% a decision, never the likeliest, though pros
        # take a third base in 86% of TvZ games of 15 min+ (median 12.9 min) and Gary in almost none
        self.expand_mass: float | None = None
        self.answers = None                                 # answers.py Answers (option answers): the zerg's plan answered
        self.expand_odds = 0.0
        # odds_mass (option mass): the same for every one-off decision (a tech building, an add-on, research,
        # an upgrade): its odds summed over the decisions taken while it was possible; at this much it is the
        # next decision. Taking the favorite each time never gets to them: held-out, the model's first
        # Factory by 8:00 came in 4% of games against the pros' 62%; in human game 3 the Refinery came at 7:26
        # and the Academy at 7:39 (pros ~4:10) with odds of 4-11% a decision for minutes
        self.odds_mass: float | None = None
        self.one_off = [k for k, (act, uid) in enumerate(model.spec.decisions) if self._one_off(act, uid)]
        self.odds: dict[int, float] = {}                    # one-off decision -> its odds summed since possible
        self.p_expand = 0.0                                 # the model's odds of expanding at its last query
        self.halls = 0                                      # Command Centers Gary has had, built or building

    def _one_off(self, act: int, uid: int) -> bool:
        """A decision made once in a while (option mass sums its odds)."""
        if self.g.race.letter == "Z":
            return act in (RESEARCH, UPGRADE, EXPAND) or (act == BMORPH and uid in Z_ONE_OFF_MORPHS) or \
                (act == BUILD and uid in Z_ONE_OFF_BUILDINGS)
        return act in (RESEARCH, UPGRADE) or (act == BUILD and uid in ONE_OFF_BUILDINGS)

    def step(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> None:
        g = self.g
        build = self.builds.get(g.posture.build) if g.posture.build else None
        if build is not None:
            if build.first(obs, mine, done, me):
                return
            if self.goal is None:
                self.goal = build.next_goal(obs, mine, done, me)
                if self.goal is None and not getattr(build, "falls_through", False):
                    return
                if self.goal is not None:           # (else the learned macro goes on, under the posture's filters)
                    g.say(f"next: {self.goal['name']} ({build.label})")
                    LOG.decision(g.hi.side, "macro", g.hi.frame, self.goal["name"], inputs=self._seen(obs, mine, me),
                                 by=f"posture's build: {build.label}")
        if self.natural_bunker and self._natural_bunker(mine, me):
            return
        if self.answers is not None:
            self.answers.observe(obs)
            if self.goal is None and g.task is None and self.answers.tech_scan(mine):
                return
            if self.goal is None and (goal := self.answers.next_goal(obs, mine, done, me)) is not None:
                self.goal = goal
                g.say(f"next: {goal['name']}")
                LOG.decision(g.hi.side, "macro", g.hi.frame, goal["name"], inputs=self._seen(obs, mine, me),
                             by="rule: an answer to the zerg's plan (answers.py)")
        self._learned(obs, mine, done, me)

    def _natural_bunker(self, mine: list[dict], me: dict) -> bool:
        """A bunker at the natural once Gary takes it. Pro TvZ Terrans that expand by 5:00 build one
        within 2:30 of the Command Center in about half the games (50-60% on every open map; 19% on
        Colosseum II, where they wall), a median 35-56 s after it; about 5 tiles from the CC, 45% of
        the way to the natural's way out (data/interim/build_orders.jsonl; replays at 5:30)."""
        g = self.g
        nat = g.natural.center
        if g.hi.frame < self.next_nat_bunker or g.task is not None:
            return False
        if not any(u["type"] in g.race.town_halls and math.dist((u["x"], u["y"]), nat) < 4 * 32 for u in mine) or \
                not any(u["type"] == T.RAX and u["completed"] for u in mine):
            return False
        if any(u["type"] == T.BUNKER and math.dist((u["x"], u["y"]), nat) < 8 * 32 for u in mine) or \
                any(r == T.BUNKER for r, *_ in g.reserved):
            return False
        if me["minerals"] - g.committed() < g.race.costs[T.BUNKER][0]:
            return False
        self.next_nat_bunker = g.hi.frame + NAT_BUNKER_RETRY
        g.task = Task("build", T.BUNKER, "natural", data={"near": self.nat_bunker_spot(), "level_at": nat},
                      started=g.hi.frame, by="natural bunker")
        g.say("builds a bunker at the natural")
        return True

    def nat_bunker_spot(self) -> tuple[int, int]:
        """Where the natural's bunker goes: NAT_BUNKER_ALONG of the way from its Command Center to its
        way out (the frontier of the main and the natural, from map knowledge), or NAT_BUNKER_TILES
        toward the nearest enemy start where the natural is open ground."""
        if self._nat_bunker_spot is None:
            from gary.walls import frontier
            g = self.g
            nat, main = g.natural.center, g.main.center
            full = g.hi.game.observe()
            fr = frontier(g.hi.game, {**full, "units": [u for u in full["units"] if u["owner"] == 11]},
                          [main, nat], list(g.enemy_starts)) \
                    if g.enemy_starts and getattr(g.hi.game, "terrain", None) is not None else None
            if fr and fr["openings"] and not fr["open"]:
                op = min(fr["openings"], key=lambda o: math.dist((o.x, o.y), nat))
                self._nat_bunker_spot = (int(nat[0] + (op.x - nat[0]) * NAT_BUNKER_ALONG),
                                         int(nat[1] + (op.y - nat[1]) * NAT_BUNKER_ALONG))
            else:
                to = min(g.enemy_starts, key=lambda s: math.dist(s, nat)) if g.enemy_starts else main
                d = math.dist(to, nat) or 1
                self._nat_bunker_spot = (int(nat[0] + (to[0] - nat[0]) / d * NAT_BUNKER_TILES * 32),
                                         int(nat[1] + (to[1] - nat[1]) / d * NAT_BUNKER_TILES * 32))
        return self._nat_bunker_spot

    def _learned(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> None:
        g = self.g
        hi = g.hi
        # a Barracks by 1:30: pro TvZ Terrans order their first one at a median of 1:28, 95% by
        # 1:39 (data/interim/build_orders.jsonl; 2% build a CC first). Without one nothing holds
        # an early pool: the bunker needs it (4 of v0.11's 9 losses to a 9-pool)
        # (a second by 3:45, as the pros of the standard style have it at 3:31, lost 12 of 144 to the
        # replayed pros, 8 without)
        # (Zerg: a Spawning Pool by 2:45; one that didn't go up at 2:50 got the model's 1-4% odds after, and a
        # Zerg without one sat on 2,000 minerals with no army to 8:25: round G1)
        first = FIRST_PRODUCER.get(g.race.letter)
        rax = sum(1 for u in mine if u["type"] == first) + sum(1 for r, *_ in g.reserved if r == first) + g.building(first) \
            if self.rax_by is not None and first is not None else 1
        if not rax and hi.frame >= self.rax_by and (self.goal is None or self.goal["id"] != first) \
                and ((g.race.letter == "T" and hi.frame < FIRST_OPENING)
                     or self.plan(BUILD, first, obs, mine, done, me) not in (None, WAIT)):
            # (and one can be built now: a Zerg with its Pool killed and no drone free set the goal and dropped it
            # 459 times in 15 s, its macro held: round G5; a Terran too after the opening: with its Barracks all
            # destroyed, in 15 of the 96 games of rounds R50a+b, all lost, it set the goal 7 to 2,955 times a game)
            fname = describe(g.race, BUILD, first)
            self.goal = {"act": BUILD, "id": first, "p": 1.0, "since": hi.frame, "name": fname}
            g.say(f"next: {fname} (none by {self.rax_by // 24 // 60}:{self.rax_by // 24 % 60:02d})")
            LOG.decision(g.hi.side, "macro", hi.frame, fname, inputs=self._seen(obs, mine, me),
                         by=f"rule: a {fname} by {self.rax_by // 24 // 60}:{self.rax_by // 24 % 60:02d}")
        if self.supply_ahead and (self.goal is None or self.goal["id"] != g.race.supply_provider) and me["supply_max"] < 200 \
                and (due := g.executor.supply_due(obs, mine, done, me)) is not None:
            act, by, uncapped = due
            name = describe(g.race, act, g.race.supply_provider)
            self.goal = {"act": act, "id": g.race.supply_provider, "p": 1.0, "since": hi.frame, "name": name}
            if uncapped:
                self.goal["uncapped"] = True
            g.say(f"next: {name} (ahead of production)")
            LOG.decision(g.hi.side, "macro", hi.frame, name, inputs=self._seen(obs, mine, me), by=by)
        if self.goal and hi.frame - self.goal["since"] > GOAL_TIMEOUT:
            g.say(f"gives up on {self.goal['name']}")
            LOG.order(g.hi.side, hi.frame, self.goal["name"], "dropped", f"not carried out in {GOAL_TIMEOUT // 24} s", by="macro")
            self.goal = None
        if self.saved is not None and hi.frame - self.saved["since"] > GOAL_TIMEOUT:
            g.say(f"gives up on {self.saved['name']}")
            LOG.order(g.hi.side, hi.frame, self.saved["name"], "dropped", f"parked: not carried out in {GOAL_TIMEOUT // 24} s",
                      by="macro")
            self.saved = None
        if self.goal is None and self.saved is not None and self._affordable(self.saved, obs, mine, done, me, own=True):
            self.goal, self.saved = self.saved, None            # (its money is there: it goes first)
            LOG.decision(g.hi.side, "macro", hi.frame, self.goal["name"], self.goal["p"], inputs=self._seen(obs, mine, me),
                         by="parked", then="unparked: its money is there")
        halls = sum(1 for u in mine if u["type"] in g.race.town_halls)
        if halls > self.halls:
            self.expand_odds = 0.0          # (only once one stands: a Command Center that didn't start is asked again)
        self.halls = halls
        if self._train_queued(obs, mine, done, me):
            return
        if self.goal is None:
            if hi.frame < self.next_query:
                return
            self.next_query = hi.frame + QUERY_EVERY
            # the most likely decision Gary can carry out now; if none can, wait for the most
            # likely one that only needs a busy building to free up
            x = self.tracker.features(obs, self.in_production(mine))
            probs, secs = self.model.probs(x)
            if self.pace:
                self.next_query = hi.frame + int(min(PACE_MAX, max(QUERY_EVERY, float(secs[0]) * 24)))
            pick = waiting = waiting_why = None
            # (the record: the model's top odds, what it saw, the options ahead of the choice and why they were
            # passed over; gary/decision_log.py)
            name = lambda k: describe(g.race, *self.model.spec.decisions[k])
            over: list[list] = []

            def record(k, by, then=None):
                LOG.decision(g.hi.side, "macro", hi.frame, None if k is None else name(k),
                             None if k is None else float(probs[0, k]), top(name, probs[0]),
                             self._seen(obs, mine, me), by, over[:8], then,
                             why=explain_macro(self.model, x, k) if k is not None else None)
            if self.decoder == "sample":
                # training (docs/CONTROL.md step 7): a draw at the temperature among the decisions Gary
                # can carry out now or soon (a busy building), logged with what it was drawn from
                cands = [k for k in np.argsort(-probs[0]) if probs[0, k] >= DICE_FLOOR
                         and self.plan(*self.model.spec.decisions[k], obs, mine, done, me) is not None]
                if not cands:
                    return
                q = np.power(probs[0, cands].astype(np.float64), 1.0 / self.temperature)
                k = int(cands[int(self.rng.choices(range(len(cands)), weights=(q / q.sum()).tolist())[0])])
                self.draws.append({"frame": hi.frame, "x": x[0], "cands": [int(c) for c in cands], "k": k})
                act, uid = self.model.spec.decisions[k]
                self.goal = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame, "name": describe(self.g.race, act, uid)}
                g.say(f"next: {self.goal['name']} ({self.goal['p']:.0%}, drawn)")
                record(k, f"sample (temperature {self.temperature:g})")
                return
            if self.decoder == "dice":
                # roll the model's odds rather than always take the favorite: a weighted shuffle,
                # then the first decision Gary can carry out (Factories, Starports, Engineering Bays
                # are a few percent each ask: the favorite, almost always a unit, never let them
                # come on time; on the pros' own states, a Factory by 8:00 in 4% of games against
                # their 62%, rolled 63%)
                w = probs[0]
                if self.dice_t != 1.0:
                    w = w ** (1.0 / self.dice_t)
                    w = w / w.sum()
                order = [k for _, k in sorted(((self.rng.random() ** (1 / w[k]), k) for k, p in enumerate(probs[0])
                                               if p >= DICE_FLOOR and w[k] > 0), reverse=True)]
                floor = DICE_FLOOR
                if self.hold and self.held is not None:
                    if hi.frame - self.held[1] < HOLD_FOR and probs[0, self.held[0]] >= DICE_FLOOR:
                        order = [self.held[0]] + [k for k in order if k != self.held[0]]
                    else:
                        self.held = None
            else:
                order, floor = np.argsort(-probs[0]), MIN_P
            k_exp = self.model.spec.decisions.index(next(d for d in self.model.spec.decisions if d[0] == EXPAND))
            self.p_expand = float(probs[0, k_exp])
            # (from the third base on: forced at 4:00-6:00 as Gary held one base under pressure, the natural
            # cost 7 games of 432; the model takes the natural itself)
            if (self.expand_mass is not None and self.expand_odds >= self.expand_mass and self.halls >= 2
                    and self.plan(*self.model.spec.decisions[k_exp], obs, mine, done, me) is not None):
                act, uid = self.model.spec.decisions[k_exp]
                self.goal = {"act": act, "id": uid, "p": float(probs[0, k_exp]), "since": hi.frame, "name": describe(self.g.race, act, uid)}
                g.say(f"next: expand ({self.expand_odds:.2f} expansions' odds since the last)")
                record(k_exp, f"rule: expansion odds summed to {self.expand_odds:.2f} since the last")
                return
            if self.odds_mass is not None:
                due = [(self.odds[k], k) for k in self.one_off if self.odds.get(k, 0.0) >= self._mass_of(k)]
                for mass, k in sorted(due, reverse=True):
                    plan = self.plan(*self.model.spec.decisions[k], obs, mine, done, me)
                    if plan is not None and plan is not WAIT:
                        act, uid = self.model.spec.decisions[k]
                        self.goal = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame, "name": describe(self.g.race, act, uid)}
                        self.odds[k] = 0.0
                        g.say(f"next: {self.goal['name']} ({mass:.2f} odds summed since it was possible)")
                        record(k, f"rule: its odds summed to {mass:.2f} since it was possible")
                        return
            if self.force is not None and hi.frame >= self.force[0]:
                # a branch (train/macro_rl.py): another decision at this moment of the same game
                k = self.force[1]
                self.force = None
                if self.plan(*self.model.spec.decisions[k], obs, mine, done, me) is not None:
                    act, uid = self.model.spec.decisions[k]
                    self.goal = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame, "name": describe(self.g.race, act, uid)}
                    g.say(f"next: {self.goal['name']} ({self.goal['p']:.0%}, forced)")
                    record(k, "forced (a branch)")
                    return
            if self.log:
                cands = [int(c) for c in order if probs[0, c] >= floor
                         and self.plan(*self.model.spec.decisions[c], obs, mine, done, me) is not None]
                logged = {"frame": hi.frame, "x": x[0], "cands": cands}
                self.draws.append(logged)
            for k in order:
                if probs[0, k] < floor:
                    break
                floor_one_off = getattr(g.posture, "one_off_min_p", None)
                if floor_one_off is not None and self._one_off(*self.model.spec.decisions[k]) and probs[0, k] < floor_one_off:
                    over.append([name(k), f"rule: the posture: a one-off under {floor_one_off:.0%}"])
                    continue
                if self.tech_min_p is not None and self.model.spec.decisions[k][0] == BUILD \
                        and self.model.spec.decisions[k][1] in TECH_PATH.get(g.race.letter, ()) and probs[0, k] < self.tech_min_p:
                    over.append([name(k), f"rule: a tech path under {self.tech_min_p:.0%}"])
                    continue
                plan = self.plan(*self.model.spec.decisions[k], obs, mine, done, me)
                if self.park and isinstance(plan, dict) and plan is not WAIT:
                    act, uid = self.model.spec.decisions[k]
                    affordable = self._affordable({"act": act, "id": uid}, obs, mine, done, me, plan=plan)
                    if g.executor.burst(act, uid) and not affordable:
                        # (a larva's the money doesn't cover yet: no goal, the dice again at the next look, as
                        # Gary-Z's; held as the goal until the money came, nothing else was rolled meanwhile: 32
                        # decisions a game in the first 4 minutes to Gary-Z's 83, the Extractor at 3:05 to 1:22)
                        if self.hold and self.held is None:
                            self.held = (k, hi.frame)
                        record(None, "dice" if self.decoder == "dice" else "favorite", f"waits: {name(k)} needs money")
                        return
                    if not g.executor.burst(act, uid) and not affordable:
                        if self.saved is None:
                            self.saved = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame,
                                          "name": describe(g.race, act, uid)}
                            over.append([name(k), "parked for its money"])
                            record(k, "dice" if self.decoder == "dice" else "favorite", "parked for its money")
                        else:
                            over.append([name(k), "one already parked: meanwhile only larvae"])
                        continue
                if plan is WAIT:
                    over.append([name(k), f"waits: {self.why_not or 'its building is busy'}" if waiting is None
                                 else f"also waits: {self.why_not or 'its building is busy'}"])
                    if waiting is None:
                        waiting, waiting_why = k, self.why_not or "its building is busy"
                        if self.hold and waiting_why.startswith("supply"):
                            if self.held is None:
                                self.held = (k, hi.frame)
                            break
                elif plan is not None:
                    # something doable now, unless the likelier decision is worth waiting for
                    # (a pro saves for the expansion rather than spend on whatever is possible)
                    if self.wait_min_p is not None and waiting is not None:
                        # (option wait_min_p, Gary-Z's: meanwhile anything at these odds; half the waited one's kept
                        # every Extractor behind the drones a larva was awaited for, to 3:15 against Gary-Z's 1:20)
                        if probs[0, k] >= self.wait_min_p:
                            pick = k
                            break
                        over.append([name(k), f"rule: under {self.wait_min_p:.0%} while {name(waiting)} waits"])
                        continue
                    if waiting is None or probs[0, k] >= 0.5 * probs[0, waiting]                             or (self.bank_escape and me["minerals"] > BANK_ESCAPE):
                        pick = k
                    else:
                        over.append([name(k), f"under half the odds of {name(waiting)}, which it waits for"])
                    break
                else:
                    over.append([name(k), f"can't now: {self.why_not}" if self.why_not else "can't now"])
            if self.park and pick is None and waiting is not None and g.executor.burst(*self.model.spec.decisions[waiting]):
                # (a larva awaited: no goal, the dice again at the next look, as Gary-Z's; held as the goal, nothing
                # else was rolled until a larva came: 5 rolls at 1:00-1:30 to Gary-Z's 7, half its Extractors)
                record(None, "dice" if self.decoder == "dice" else "favorite", f"waits: {waiting_why} for {name(waiting)}")
                return
            k = pick if pick is not None else waiting
            if k is not None:
                self.held = None                                # (taken: the next roll is a new one)
            if self.log:
                logged["k"] = None if k is None else int(k)
            record(k, "dice" if self.decoder == "dice" else "favorite",
                   f"waits: {waiting_why}" if k is not None and pick is None else None if k is not None else
                   "nothing it can do now")
            if k is None:
                return
            act, uid = self.model.spec.decisions[k]
            self.goal = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame,
                         "name": describe(self.g.race, act, uid)}
            if self.pace and g.executor.burst(act, uid):
                self.next_query = hi.frame + QUERY_EVERY      # (a larva's: the next soon, as pros spend them at once)
            self.expand_odds += float(probs[0, k_exp])
            if self.odds_mass is not None:              # (summed per decision taken, while possible or waiting)
                for j in self.one_off:
                    if j == k:
                        self.odds[j] = 0.0
                    elif probs[0, j] >= 0.01 and self.plan(*self.model.spec.decisions[j], obs, mine, done, me) is not None:
                        self.odds[j] = self.odds.get(j, 0.0) + float(probs[0, j])
            g.say(f"next: {self.goal['name']} ({self.goal['p']:.0%})")
        goal = self.goal
        plan = self.plan(goal["act"], goal["id"], obs, mine, done, me, capped=not goal.get("uncapped"))   # still possible?
        if plan is None:
            LOG.order(g.hi.side, hi.frame, goal["name"], "dropped", "no longer possible", by="macro")
            self.goal = None
            return
        if plan is WAIT:                    # e.g. the building is still busy: wait for it
            same = sum(1 for k, _ in self.queued if k == (TRAIN, goal["id"]))
            room = sum(1 for u in mine if u["type"] == g.race.producer.get(goal["id"]) and u["completed"])
            if goal["act"] == TRAIN and same < max(1, room) * self.queue_depth:
                # a unit whose producer is busy is decided now and trained when it frees up, and
                # the next decision comes meanwhile (as pros queue the SCV and send the depot's
                # builder: one decision at a time held the 9th SCV's depot 12 s behind it)
                self.queued.append(((TRAIN, goal["id"]), hi.frame))
                self.tracker.note_decision(TRAIN, goal["id"], obs["frame"])
                self.decisions.append((hi.frame, goal["name"], goal["p"]))
                self.goal = None
            elif goal["act"] == TRAIN and self.rethink:
                LOG.order(g.hi.side, hi.frame, goal["name"], "dropped", "its buildings' queues are full: asked again",
                          by="macro")
                self.goal = None
            return
        minerals, gas = plan["cost"]
        budget = me["minerals"] - g.committed() - self._saved_cost()[0]
        # a building's worker sets out WALK_AHEAD seconds of income early: it walks while the money comes
        workers = sum(1 for u in mine if u["type"] == g.race.worker)
        walk_ahead = int(WALK_AHEAD * MINING * workers) if plan["kind"] == "build" else 0
        if self.park and plan["kind"] == "build":
            # (Gary-Z's: its drone sets out at the cost less PARK_AHEAD, the larvae going on meanwhile; held until
            # 6 s of income short, nothing else decided, the second hatchery came by 4:00 in 13 of 32 games, 21)
            walk_ahead = max(walk_ahead, PARK_AHEAD)
        if budget < minerals - walk_ahead or me["gas"] - self._saved_cost()[1] < gas:
            if self.nosave:
                self.goal = None
            return
        if plan["kind"] == "build" and g.parked is not None:
            return                          # one builder walking ahead at a time (the parked build)
        if plan["kind"] == "build":
            data = {**plan.get("data", {}), "decision": (goal["act"], goal["id"])}
            if "near" in goal:                  # a goal that names its base and spot (answers.py turrets)
                data.update(near=goal["near"], level_at=goal.get("level_at"), avoid=goal.get("avoid"))
            g.task = Task("build", goal["id"], goal.get("where", plan["where"]), data=data, started=hi.frame)
        else:
            g.task = Task(plan["kind"], data=plan, started=hi.frame)
            g.spend(minerals)
        self.tracker.note_decision(goal["act"], goal["id"], obs["frame"])
        self.decisions.append((hi.frame, goal["name"], goal["p"]))
        self.goal = None

    def _seen(self, obs: dict, mine: list[dict], me: dict) -> dict:
        """What the macro saw, in brief, for its records."""
        g = self.g
        return {"m": me["minerals"], "gas": me["gas"], "supply": f"{me['supply_used']}/{me['supply_max']}",
                "workers": sum(1 for u in mine if u["type"] == g.race.worker), "bases": sum(1 for u in mine if u["type"] in g.race.town_halls),
                "army": sum(1 for u in mine if u["type"] in g.army_types), "posture": getattr(g.posture, "build", None),
                "parked": g.race.names.get(g.parked.unit_type) if g.parked is not None else None}

    def _mass_of(self, k: int) -> float:
        """The summed odds at which this one-off is due: a town hall's hatch_mass, if set."""
        act, uid = self.model.spec.decisions[k]
        if self.hatch_mass is not None and (act == EXPAND or uid == self.g.race.town_hall):
            return self.hatch_mass
        return self.odds_mass

    def _saved_cost(self) -> tuple[int, int]:
        """The parked decision's money, kept for it (option park)."""
        if self.saved is None:
            return 0, 0
        return self.g.executor.cost_of(self.saved["act"], self.saved["id"])

    def _affordable(self, goal: dict, obs: dict, mine: list[dict], done: list[dict], me: dict, own: bool = False,
                    plan: dict | None = None) -> bool:
        """Its money is there, less what is spoken for and the parked decision's (unless it is that one). A
        building's always is: its worker walks while the money comes, and the executor waits for it."""
        g = self.g
        if goal["act"] in (BUILD, EXPAND):
            return True
        plan = plan or self.plan(goal["act"], goal["id"], obs, mine, done, me)
        if not isinstance(plan, dict) or plan is WAIT:
            return False
        minerals, gas = plan["cost"]
        sm, sg = (0, 0) if own else self._saved_cost()
        return me["minerals"] - g.committed() - sm >= minerals and me["gas"] - sg >= gas

    def _train_queued(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> bool:
        """A decided unit whose producer is free now: trained (first, as pros keep production going).
        True if this turn is spent."""
        g = self.g
        for entry in list(self.queued):
            key, since = entry
            plan = self.plan(*key, obs, mine, done, me)
            if plan is None or g.hi.frame - since > GOAL_TIMEOUT:
                self.queued.remove(entry)
                continue
            if plan is WAIT or g.task is not None:
                continue
            minerals, gas = plan["cost"]
            if me["minerals"] - g.committed() < minerals or me["gas"] < gas:
                continue
            g.task = Task("command", data=plan, started=g.hi.frame)
            g.spend(minerals)
            self.queued.remove(entry)
            return True
        return False

    def plan(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict,
             capped: bool = True) -> dict | None:
        """How to carry out this decision now; WAIT if Gary will be able to soon (the building that
        makes it is busy); None if Gary can't (a requirement is missing, nothing can make it, Gary
        doesn't know how, or the posture forbids it). Gary waits for a busy building rather than
        doing something less likely: the model's next decision is usually right, it just isn't
        time yet."""
        g = self.g
        posture = g.posture
        self.why_not = None                   # (why None or WAIT, for the records: the gates' or the executor's)
        if posture.allowed is not None and (act == EXPAND or act not in (TRAIN, BUILD) or uid not in posture.allowed):
            return self._no("the posture allows it not")
        if posture.no_expand and act == EXPAND:
            return self._no("the posture: no expansion")
        if self.answers is not None and not self.answers.allows(act, uid, mine):
            return self._no("an answer to the enemy's plan says no")
        have = {u["type"] for u in done}
        if not g.race.meets(have, uid):
            return self._no("what it needs isn't there")
        if getattr(posture, "no_workers", False) and uid == g.race.worker:
            return self._no("the posture: no workers now")
        if self.failed_until.get((act, uid), 0) > g.hi.frame:
            return self._no("it just failed")      # (e.g. no room): not again right away
        # (a larva's unit and a morph too: without, gz1 made drones past the pros' cap at 1:00, and its
        # Extractor came at 3:05 where Gary-Z's dice, the drones capped, came to it by 1:25)
        # (capped: not a rule's goal that ignores the caps, as Gary-Z's overlord rule did)
        cap = self.model.cap(uid, obs["frame"]) if capped and self.caps_on and act in (TRAIN, MORPH, BMORPH, BUILD, EXPAND) \
            else None
        if cap is not None and self.count(mine, uid) >= cap:
            return self._no(f"the pros' cap for this minute ({cap})")   # as many as nine pros in ten ever had by now
        g.executor.why = None
        plan = g.executor.carry_out(act, uid, obs, mine, done, me)
        self.why_not = g.executor.why if plan is None or plan is WAIT else None
        return plan

    def _no(self, why: str) -> None:
        self.why_not = why
        return None

    def build_failed(self, t: Task) -> None:
        """A building Gary decided on couldn't be placed: it didn't happen, and it isn't retried
        for 10 s."""
        if "decision" in t.data:
            act, uid = t.data["decision"]
            self.tracker.unnote_decision(act, uid)
            self.failed_until[(act, uid)] = self.g.hi.frame + 24 * 10
            self.g.say(f"couldn't build {describe(self.g.race, act, uid)}")
            LOG.order(self.g.hi.side, self.g.hi.frame, describe(self.g.race, act, uid), "dropped",
                      f"couldn't be built ({t.data.get('failed', 'see the executor')}): not decided again for 10 s",
                      by="executor")

    # --- what Gary has -------------------------------------------------------------------------

    def in_production(self, mine: list[dict]) -> Counter:
        """Units being made right now (the game counts them, observations don't list them): the executor's."""
        return self.g.executor.in_production(mine)

    def count(self, mine: list[dict], unit_type: int) -> int:
        """Own units of a type, as the pros' caps count them: the executor's."""
        return self.g.executor.count(mine, unit_type)


# --- scripted builds a posture switches to --------------------------------------------------

RUSH_SCVS = 20                       # SCVs to keep making while defending (one base)
RUSH_RAX = 3
RAX_MONEY = 200                      # minerals left over: another Barracks
BUNKER_RETRY = 20 * 24               # a bunker order that didn't take: try again this much later


class OneBaseDefense:
    """Against a one-base all-in (v0.9): a Barracks if there's none, the bunker first (where the
    posture says; 100 minerals kept for it until it's coming), then depots ahead of production,
    marines and SCVs nonstop, more Barracks on spare money."""
    label = "rush defense"

    def __init__(self, g: Gary):
        self.g = g
        self.next_bunker = 0

    def first(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> bool:
        """The bunker before anything else (saving for it). True if this turn is spent."""
        g = self.g
        if self._bunker_coming(mine) or not any(u["type"] == T.RAX and u["completed"] for u in mine):
            return False
        budget = me["minerals"] - g.committed()
        if g.build_failed.get(T.BUNKER, -1) > self.next_bunker - BUNKER_RETRY:   # one just failed: again now
            self.next_bunker = 0
        if budget >= g.race.costs[T.BUNKER][0] - 40 and g.hi.frame >= self.next_bunker:
            self.next_bunker = g.hi.frame + BUNKER_RETRY
            g.task = Task("build", T.BUNKER, "main", data={"near": g.posture.bunker_spot, "level_at": g.main.center},
                          started=g.hi.frame)
            g.say("builds a bunker at the ramp")
        return True                      # save for the bunker

    def next_goal(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        """The next build against a one-base all-in, in the macro's goal format."""
        g = self.g
        goal = lambda act, uid, name: {"act": act, "id": uid, "p": 1.0, "since": g.hi.frame, "name": name}
        coming = lambda ut: any(u["type"] == ut and not u["completed"] for u in mine) or \
            any(r == ut for r, *_ in g.reserved)
        doable = lambda act, uid: g.macro.plan(act, uid, obs, mine, done, me) not in (None, WAIT)
        # until the bunker is up or coming, 100 minerals stay for it (it can start only once a
        # Barracks is done, and it's what holds the rush): no extra Barracks, marines and SCVs
        # only out of what's left over
        # no Barracks at all yet (the learned macro opened otherwise): that first, it's what the
        # bunker needs
        if not any(u["type"] == T.RAX for u in mine) and not coming(T.RAX):
            return goal(BUILD, T.RAX, "Barracks") if doable(BUILD, T.RAX) else None
        # (an ordered bunker's money is held by its reservation until it starts)
        keep = g.race.costs[T.BUNKER][0] if not any(u["type"] == T.BUNKER for u in mine) and \
            not any(r == T.BUNKER for r, *_ in g.reserved) else 0
        spare = lambda ut: me["minerals"] - g.committed() - keep >= g.race.costs[ut][0]
        # a depot takes longer than a marine or an SCV: start it while each producer has 2 left
        producers = sum(1 for u in done if u["type"] in (T.CC, T.RAX))
        if me["supply_max"] - me["supply_used"] < 2 * producers and me["supply_max"] < 200 and \
                not coming(T.DEPOT) and doable(BUILD, T.DEPOT):
            return goal(BUILD, T.DEPOT, "Supply Depot")
        if doable(TRAIN, T.MARINE) and spare(T.MARINE):
            return goal(TRAIN, T.MARINE, "Marine")
        if sum(1 for u in mine if u["type"] == g.race.worker) < RUSH_SCVS and doable(TRAIN, g.race.worker) and spare(g.race.worker):
            return goal(TRAIN, g.race.worker, "SCV")
        budget = me["minerals"] - g.committed()
        if not keep and budget >= RAX_MONEY and sum(1 for u in mine if u["type"] == T.RAX) < RUSH_RAX and not coming(T.RAX) and \
                doable(BUILD, T.RAX):
            return goal(BUILD, T.RAX, "Barracks")
        return None

    def _bunker_coming(self, mine: list[dict]) -> bool:
        g = self.g
        return any(u["type"] == T.BUNKER for u in mine) or any(ut == T.BUNKER for ut, *_ in g.reserved) or \
            g.building(T.BUNKER)


FLOOD_RAX = 3
FLOOD_PER_MEDIC = 4                  # a medic per this many marines
FLOOD_PER_FIREBAT = 4                # a firebat per this many marines
STIM = 0                             # research id: Stim Packs


class LingFloodDefense:
    """Against a zergling mass on its way (posture v6): depots ahead of production, Barracks to
    FLOOD_RAX, a Refinery and an Academy, Stim; medics and firebats among the marines; SCVs only
    from what's left. In the learned macro's goal format."""
    label = "zergling flood"

    def __init__(self, g: Gary):
        self.g = g

    def first(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> bool:
        return False

    def next_goal(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        g = self.g
        goal = lambda act, uid, name: {"act": act, "id": uid, "p": 1.0, "since": g.hi.frame, "name": name}
        coming = lambda ut: any(u["type"] == ut and not u["completed"] for u in mine) or \
            any(r == ut for r, *_ in g.reserved)
        doable = lambda act, uid: g.macro.plan(act, uid, obs, mine, done, me) not in (None, WAIT)
        count = lambda ut: sum(1 for u in mine if u["type"] == ut)
        budget = me["minerals"] - g.committed()
        producers = sum(1 for u in done if u["type"] in (T.CC, T.RAX))
        if me["supply_max"] - me["supply_used"] < 2 * producers and me["supply_max"] < 200 and \
                not coming(T.DEPOT) and doable(BUILD, T.DEPOT):
            return goal(BUILD, T.DEPOT, "Supply Depot")
        if count(T.RAX) < 2 and not coming(T.RAX) and doable(BUILD, T.RAX):
            return goal(BUILD, T.RAX, "Barracks")
        if not count(g.race.gas_building) and doable(BUILD, g.race.gas_building):
            return goal(BUILD, g.race.gas_building, "Refinery")
        if not count(T.ACADEMY) and doable(BUILD, T.ACADEMY):
            return goal(BUILD, T.ACADEMY, "Academy")
        if STIM not in g.researched and doable(RESEARCH, STIM):
            return goal(RESEARCH, STIM, "Stim Packs")
        marines = count(T.MARINE)
        if count(T.MEDIC) * FLOOD_PER_MEDIC < marines and doable(TRAIN, T.MEDIC):
            return goal(TRAIN, T.MEDIC, "Medic")
        if count(T.FIREBAT) * FLOOD_PER_FIREBAT < marines and doable(TRAIN, T.FIREBAT):
            return goal(TRAIN, T.FIREBAT, "Firebat")
        if doable(TRAIN, T.MARINE):
            return goal(TRAIN, T.MARINE, "Marine")
        if count(T.RAX) < FLOOD_RAX and not coming(T.RAX) and budget >= RAX_MONEY and doable(BUILD, T.RAX):
            return goal(BUILD, T.RAX, "Barracks")
        if doable(TRAIN, g.race.worker) and budget >= g.race.costs[g.race.worker][0] + 50:
            return goal(TRAIN, g.race.worker, "SCV")
        return None
