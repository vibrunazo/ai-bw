"""Macro decisions (layer L2, docs/CONTROL.md): what to build, train, research next.

  learned           the macro model (gary/policy/macro.py, ~14,000 pro TvZ games; Gary v0.2+):
                    whenever Gary is free to decide, the likeliest decision it can carry out now
                    (requirements built, a building free, a base free), then wait for the money.
                    Steerable by build style. What the posture forbids (L1) it never plans.
  one_base_defense  a scripted build a posture can switch to (v0.9's answer to an early pool): a
                    bunker at the ramp first, then depots ahead of production, marines and SCVs
                    nonstop, a second Barracks on spare money. It gives goals in the learned
                    macro's format, carried out the same way.

The executor (L2b) carries the decisions out: a build task or a command task.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from gary import terran as T
from gary.agent.executor import COST
from gary.agent.gary import Gary, Task
from gary.policy.macro import ACT_NAMES, MacroModel, MacroTracker

TRAIN, BUILD, RESEARCH, UPGRADE, EXPAND = 0, 3, 4, 5, 6
TOWN_HALLS = {106, 131, 132, 133, 154}
MAX_SCVS = 60
WALK_AHEAD = 6                  # seconds: a building's worker sets out this much income before the money is there
MINING = 1.0                    # minerals a second an SCV mines (about)
GOAL_TIMEOUT = 24 * 30          # a decision Gary can't afford or do within 30 s is dropped
MIN_P = 0.05                    # never take a decision the model gives less than this: unlikely
                                # choices lead into states pros rarely reach, where it knows less
BUSY = {T.ORDER_RESEARCH, T.ORDER_UPGRADE, T.ORDER_BUILD_ADDON}
WAIT = {"kind": "wait"}         # plan(): possible, but not right now
KEEP_BUSY = {T.RAX, T.FACTORY, T.STARPORT}   # producers that keep making their last unit (keep_busy)
NAT_BUNKER_ALONG = 0.45         # the natural's bunker: this far from its Command Center toward its way out
NAT_BUNKER_TILES = 5            # ...or this far toward the enemy where the natural has no choke
NAT_BUNKER_RETRY = 20 * 24


def describe(act: int, uid: int) -> str:
    if act == TRAIN:
        return T.NAMES.get(uid, f"unit {uid}")
    if act == RESEARCH:
        return T.TECH_NAMES.get(uid, f"tech {uid}")
    if act == UPGRADE:
        return T.UPGRADE_NAMES.get(uid, f"upgrade {uid}")
    if act == EXPAND:
        return "expand"
    return T.NAMES.get(uid, f"{ACT_NAMES[act]} {uid}")


class LearnedMacro:
    def __init__(self, g: Gary, model: MacroModel, style: int | None = None, builds: dict | None = None,
                 natural_bunker: bool = False, rax_by: int = 90 * 24, keep_busy: bool = False):
        self.g = g
        self.rax_by = rax_by
        self.queued: dict[tuple[int, int], int] = {}        # units decided whose producer is busy -> since
        self.keep_busy = keep_busy
        self.last_unit: dict[int, int] = {}                 # producer type -> the unit last decided for it
        self.natural_bunker = natural_bunker
        self.next_nat_bunker = 0
        self._nat_bunker_spot = None
        self.model = model
        self.tracker = MacroTracker(model.spec, g.slot, style)
        self.goal: dict | None = None
        self.next_query = 0
        self.decisions: list[tuple[int, str, float]] = []   # (frame, decision, model probability)
        self.failed_until: dict[tuple[int, int], int] = {}  # decision -> frame it may be tried again
        self.builds = builds or {}                         # posture's build name -> scripted build

    def step(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> None:
        g = self.g
        build = self.builds.get(g.posture.build) if g.posture.build else None
        if build is not None:
            if build.first(obs, mine, done, me):
                return
            if self.goal is None:
                self.goal = build.next_goal(obs, mine, done, me)
                if self.goal is None:
                    return
                g.say(f"next: {self.goal['name']} ({build.label})")
        if self.natural_bunker and self._natural_bunker(mine, me):
            return
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
        if not any(u["type"] == T.CC and math.dist((u["x"], u["y"]), nat) < 4 * 32 for u in mine) or                 not any(u["type"] == T.RAX and u["completed"] for u in mine):
            return False
        if any(u["type"] == T.BUNKER and math.dist((u["x"], u["y"]), nat) < 8 * 32 for u in mine) or                 any(r == T.BUNKER for r, *_ in g.reserved):
            return False
        if me["minerals"] - g.committed() < T.COST[T.BUNKER][0]:
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
                          [main, nat], list(g.enemy_starts))                 if g.enemy_starts and getattr(g.hi.game, "terrain", None) is not None else None
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
        building_rax = g.building(T.RAX)
        if (hi.frame >= self.rax_by and (self.goal is None or self.goal["id"] != T.RAX) and not building_rax
                and not any(u["type"] == T.RAX for u in mine) and not any(r == T.RAX for r, *_ in g.reserved)):
            self.goal = {"act": BUILD, "id": T.RAX, "p": 1.0, "since": hi.frame, "name": "Barracks"}
            g.say(f"next: Barracks (none by {self.rax_by // 24 // 60}:{self.rax_by // 24 % 60:02d})")
        if self.goal and hi.frame - self.goal["since"] > GOAL_TIMEOUT:
            g.say(f"gives up on {self.goal['name']}")
            self.goal = None
        if self._train_queued(obs, mine, done, me) or self._keep_busy(obs, mine, done, me):
            return
        if self.goal is None:
            if hi.frame < self.next_query:
                return
            self.next_query = hi.frame + 12
            # the most likely decision Gary can carry out now; if none can, wait for the most
            # likely one that only needs a busy building to free up
            probs, _ = self.model.probs(self.tracker.features(obs, self.in_production(mine)))
            pick = waiting = None
            for k in np.argsort(-probs[0]):
                if probs[0, k] < MIN_P:
                    break
                plan = self.plan(*self.model.spec.decisions[k], obs, mine, done, me)
                if plan is WAIT:
                    waiting = waiting if waiting is not None else k
                elif plan is not None:
                    # something doable now, unless the likelier decision is worth waiting for
                    # (a pro saves for the expansion rather than spend on whatever is possible)
                    if waiting is None or probs[0, k] >= 0.5 * probs[0, waiting] or me["minerals"] > 400:
                        pick = k
                    break
            k = pick if pick is not None else waiting
            if k is None:
                return
            act, uid = self.model.spec.decisions[k]
            self.goal = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame,
                         "name": describe(act, uid)}
            g.say(f"next: {self.goal['name']} ({self.goal['p']:.0%})")
        goal = self.goal
        plan = self.plan(goal["act"], goal["id"], obs, mine, done, me)   # still possible?
        if plan is None:
            self.goal = None
            return
        if plan is WAIT:                    # e.g. the building is still busy: wait for it
            if goal["act"] == TRAIN and (TRAIN, goal["id"]) not in self.queued:
                # a unit whose producer is busy is decided now and trained when it frees up, and
                # the next decision comes meanwhile (as pros queue the SCV and send the depot's
                # builder: one decision at a time held the 9th SCV's depot 12 s behind it)
                self.queued[(TRAIN, goal["id"])] = hi.frame
                self.tracker.note_decision(TRAIN, goal["id"], obs["frame"])
                self.decisions.append((hi.frame, goal["name"], goal["p"]))
                self.goal = None
            return
        minerals, gas = plan["cost"]
        budget = me["minerals"] - g.committed()
        # a building's worker sets out WALK_AHEAD seconds of income early: it walks while the money comes
        workers = sum(1 for u in mine if u["type"] == T.SCV)
        walk_ahead = int(WALK_AHEAD * MINING * workers) if plan["kind"] == "build" else 0
        if budget < minerals - walk_ahead or me["gas"] < gas:
            return
        if plan["kind"] == "build" and g.parked is not None:
            return                          # one builder walking ahead at a time (the parked build)
        if plan["kind"] == "build":
            data = {**plan.get("data", {}), "decision": (goal["act"], goal["id"])}
            g.task = Task("build", goal["id"], plan["where"], data=data, started=hi.frame)
        else:
            g.task = Task("command", data=plan, started=hi.frame)
            self._note_unit(goal["act"], goal["id"])
        self.tracker.note_decision(goal["act"], goal["id"], obs["frame"])
        self.decisions.append((hi.frame, goal["name"], goal["p"]))
        self.goal = None

    def _train_queued(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> bool:
        """A decided unit whose producer is free now: trained (first, as pros keep production going).
        True if this turn is spent."""
        g = self.g
        for key, since in list(self.queued.items()):
            plan = self.plan(*key, obs, mine, done, me)
            if plan is None or g.hi.frame - since > GOAL_TIMEOUT:
                del self.queued[key]
                continue
            if plan is WAIT or g.task is not None:
                continue
            minerals, gas = plan["cost"]
            if me["minerals"] - g.committed() < minerals or me["gas"] < gas:
                continue
            g.task = Task("command", data=plan, started=g.hi.frame)
            del self.queued[key]
            self._note_unit(*key)
            return True
        return False

    def _note_unit(self, act: int, uid: int) -> None:
        if act == TRAIN and T.PRODUCER.get(uid) in KEEP_BUSY:
            self.last_unit[T.PRODUCER[uid]] = uid

    def _keep_busy(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> bool:
        """An idle Barracks, Factory or Starport trains the unit the model last chose for it again,
        with the money left after what the pending decision needs (pros keep production going by
        habit; one decision at a time left 2 of Gary's 3 Barracks idle with 300 minerals banked, and
        half the pros' army at 8:00). True if this turn is spent."""
        g = self.g
        if not self.keep_busy or g.task is not None:
            return False
        reserve = 0
        if self.goal is not None and self.goal["act"] not in (BUILD, EXPAND):
            # (a building saved up for doesn't stop production: pros keep their Barracks going
            # while the Command Center's 400 come in; held back for it, Gary's first Barracks idled
            # from 2:30 to 5:00, 0-3 army units at 5:00 in the games it lost. Its money is spoken
            # for once its builder walks there)
            p = self.plan(self.goal["act"], self.goal["id"], obs, mine, done, me)
            reserve = p["cost"][0] if isinstance(p, dict) and p is not WAIT else 0
        for uid in self.last_unit.values():
            plan = self.plan(TRAIN, uid, obs, mine, done, me)
            if not isinstance(plan, dict) or plan.get("kind") != "command":
                continue
            minerals, gas = plan["cost"]
            if me["minerals"] - g.committed() - reserve < minerals or me["gas"] < gas:
                continue
            g.task = Task("command", data=plan, started=g.hi.frame)
            self.tracker.note_decision(TRAIN, uid, obs["frame"])
            return True
        return False

    def plan(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        """How to carry out this decision now; WAIT if Gary will be able to soon (the building that
        makes it is busy); None if Gary can't (a requirement is missing, nothing can make it, Gary
        doesn't know how, or the posture forbids it). Gary waits for a busy building rather than
        doing something less likely: the model's next decision is usually right, it just isn't
        time yet."""
        g = self.g
        posture = g.posture
        if posture.allowed is not None and (act == EXPAND or act not in (TRAIN, BUILD) or uid not in posture.allowed):
            return None
        if posture.no_expand and act == EXPAND:
            return None
        have = {u["type"] for u in done}
        if any(r not in have for r in T.REQUIRES.get(uid, [])):
            return None
        if self.failed_until.get((act, uid), 0) > g.hi.frame:
            return None                       # just failed (e.g. no room): not again right away
        cap = self.model.cap(uid, obs["frame"]) if act in (TRAIN, BUILD, EXPAND) else None
        if cap is not None and self.count(mine, uid) >= cap:
            return None                       # already as many as nine pros in ten ever had by now
        if act == TRAIN and uid in T.PRODUCER:
            if me["supply_used"] + T.SUPPLY[uid] > me["supply_max"]:
                # supply blocked: wait if a depot is on its way, else let a depot come next
                coming = any(u["type"] == T.DEPOT and not u["completed"] for u in mine) or \
                    any(ut == T.DEPOT for ut, *_ in g.reserved) or \
                    g.building(T.DEPOT)
                return WAIT if coming else None
            if uid == T.SCV and sum(1 for u in mine if u["type"] == T.SCV) >= MAX_SCVS:
                return None
            addon = T.NEEDS_ADDON.get(uid)
            if addon is not None and not any(u["type"] == addon for u in done):
                return None                       # e.g. tanks: no Machine Shop yet
            b = self.free_building(obs, mine, done, T.PRODUCER[uid], addon)
            return self._command_or_wait(b, done, T.PRODUCER[uid], act, uid, T.COST[uid])
        if act == BUILD and uid in T.ADDON_PARENT:
            ox, oy = T.ADDON_OFFSET                      # (only where the add-on fits)
            fits = lambda b: g.hi.game.can_place(g.slot, uid, (b["x"] - T.SIZE[b["type"]][0] * 16) // 32 + ox,
                                                 (b["y"] - T.SIZE[b["type"]][1] * 16) // 32 + oy, 0)
            bare = [b for b in done if b["type"] == T.ADDON_PARENT[uid] and self.addon_of(b, mine) is None and fits(b)]
            if not bare:
                return None
            free = [b for b in bare if b.get("queue", 0) == 0 and b["order"] not in BUSY]
            return {"kind": "command", "act": act, "id": uid, "building": free[0], "cost": T.COST[uid]} \
                if free else WAIT
        if act == BUILD and uid == T.REFINERY:
            spot = self.free_geyser(obs, mine)
            return {"kind": "build", "where": spot[0], "data": {"spot": spot[1]}, "cost": T.COST[uid]} \
                if spot else None
        if act == BUILD and uid in T.SIZE and uid != T.CC:
            return {"kind": "build", "where": "main", "cost": T.COST[uid]}
        if act == EXPAND and uid == T.CC:
            base = self.free_base(obs)
            return {"kind": "build", "where": "natural" if base is g.natural else base,
                    "cost": T.COST[T.CC]} if base else None
        if act == RESEARCH and uid in T.RESEARCH and uid not in g.researched:
            where, cost = T.RESEARCH[uid]
            b = self.free_building(obs, mine, done, where)
            return self._command_or_wait(b, done, where, act, uid, cost)
        if act == UPGRADE and uid in T.UPGRADE:
            where, cost, levels = T.UPGRADE[uid]
            level = g.upgraded[uid]
            if level >= levels or (level >= 1 and T.UPGRADE_LEVEL_2_NEEDS not in have):
                return None
            b = self.free_building(obs, mine, done, where)
            cost = (cost[0] + 75 * level, cost[1] + 75 * level)
            return self._command_or_wait(b, done, where, act, uid, cost)
        return None

    def _command_or_wait(self, b: dict | None, done: list[dict], kind: int, act: int, uid: int,
                         cost: tuple[int, int]) -> dict | None:
        if b is not None:
            return {"kind": "command", "act": act, "id": uid, "building": b, "cost": cost}
        return WAIT if any(u["type"] == kind for u in done) else None

    def build_failed(self, t: Task) -> None:
        """A building Gary decided on couldn't be placed: it didn't happen, and it isn't retried
        for 10 s."""
        if "decision" in t.data:
            act, uid = t.data["decision"]
            self.tracker.unnote_decision(act, uid)
            self.failed_until[(act, uid)] = self.g.hi.frame + 24 * 10
            self.g.say(f"couldn't build {describe(act, uid)}")

    # --- what Gary has -------------------------------------------------------------------------

    def in_production(self, mine: list[dict]) -> Counter:
        """Units being made right now (the game counts them, observations don't list them)."""
        out: Counter = Counter()
        for b in mine:
            if b["completed"] and b.get("queue", 0) > 0 and b["tag"] in self.g.producing:
                out[self.g.producing[b["tag"]]] += 1
        return out

    def count(self, mine: list[dict], unit_type: int) -> int:
        """Own units of a type, counting ones in production and buildings about to be placed."""
        g = self.g
        n = sum(1 for u in mine if u["type"] == unit_type) + self.in_production(mine)[unit_type]
        n += sum(1 for ut, *_ in g.reserved if ut == unit_type)
        if g.building(unit_type):
            n += 1
        return n

    @staticmethod
    def addon_of(b: dict, mine: list[dict]) -> dict | None:
        """The add-on attached to a building (it sits just right of the building's lower half)."""
        ax, ay = b["x"] + 96, b["y"] + 16
        return next((u for u in mine if u["type"] in T.ADDON_PARENT and math.dist((u["x"], u["y"]), (ax, ay)) < 24),
                    None)

    def free_building(self, obs: dict, mine: list[dict], done: list[dict], kind: int,
                      addon: int | None = None) -> dict | None:
        g = self.g
        blind = round(g.hi.p.reaction_ms / 42) + g.hi.act_latency_frames + 4
        for b in done:
            if b["type"] != kind or b.get("queue", 0) > 0 or b["order"] in BUSY:
                continue
            if obs["now"] - g.ordered_at.get(b["tag"], -10**9) <= blind:
                continue
            if addon is not None:
                a = self.addon_of(b, mine)
                if a is None or a["type"] != addon or not a["completed"]:
                    continue
            return b
        return None

    def free_geyser(self, obs: dict, mine: list[dict]):
        g = self.g
        refineries = [u for u in mine if u["type"] == T.REFINERY] + \
                     [{"x": tx * 32 + 64, "y": ty * 32 + 32} for ut, tx, ty in g.reserved if ut == T.REFINERY]
        for base in g.executor.my_bases(mine):
            for gx, gy in base.geysers:
                if any(math.dist((r["x"], r["y"]), (gx, gy)) < 32 for r in refineries):
                    continue
                where = "main" if base is g.main else "natural" if base is g.natural else base
                return where, ((gx - 64) // 32, (gy - 32) // 32)
        return None

    def free_base(self, obs: dict):
        """The nearest base nobody has taken (the natural first), as far as Gary has seen."""
        g = self.g
        halls = [u for u in obs["units"] if u["type"] in TOWN_HALLS]
        reserved = [(tx * 32 + 64, ty * 32 + 48) for ut, tx, ty in g.reserved if ut == T.CC]
        enemy = [tuple(s) for s in g.enemy_starts]
        free = [b for b in g.map.bases if b is not g.main
                and not any(math.dist(b.center, (h["x"], h["y"])) < 10 * 32 for h in halls)
                and not any(math.dist(b.center, r) < 10 * 32 for r in reserved)
                and not any(math.dist(b.center, s) < 10 * 32 for s in enemy)]
        if g.natural in free:
            return g.natural
        return min(free, key=lambda b: math.dist(b.center, g.main.center)) if free else None


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
        if budget >= T.COST[T.BUNKER][0] - 40 and g.hi.frame >= self.next_bunker:
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
        keep = T.COST[T.BUNKER][0] if not any(u["type"] == T.BUNKER for u in mine) and \
            not any(r == T.BUNKER for r, *_ in g.reserved) else 0
        spare = lambda ut: me["minerals"] - g.committed() - keep >= T.COST[ut][0]
        # a depot takes longer than a marine or an SCV: start it while each producer has 2 left
        producers = sum(1 for u in done if u["type"] in (T.CC, T.RAX))
        if me["supply_max"] - me["supply_used"] < 2 * producers and me["supply_max"] < 200 and \
                not coming(T.DEPOT) and doable(BUILD, T.DEPOT):
            return goal(BUILD, T.DEPOT, "Supply Depot")
        if doable(TRAIN, T.MARINE) and spare(T.MARINE):
            return goal(TRAIN, T.MARINE, "Marine")
        if sum(1 for u in mine if u["type"] == T.SCV) < RUSH_SCVS and doable(TRAIN, T.SCV) and spare(T.SCV):
            return goal(TRAIN, T.SCV, "SCV")
        budget = me["minerals"] - g.committed()
        if not keep and budget >= RAX_MONEY and sum(1 for u in mine if u["type"] == T.RAX) < RUSH_RAX and not coming(T.RAX) and \
                doable(BUILD, T.RAX):
            return goal(BUILD, T.RAX, "Barracks")
        return None

    def _bunker_coming(self, mine: list[dict]) -> bool:
        g = self.g
        return any(u["type"] == T.BUNKER for u in mine) or any(ut == T.BUNKER for ut, *_ in g.reserved) or \
            g.building(T.BUNKER)
