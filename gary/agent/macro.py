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
GOAL_TIMEOUT = 24 * 30          # a decision Gary can't afford or do within 30 s is dropped
MIN_P = 0.05                    # never take a decision the model gives less than this: unlikely
                                # choices lead into states pros rarely reach, where it knows less
BUSY = {T.ORDER_RESEARCH, T.ORDER_UPGRADE, T.ORDER_BUILD_ADDON}
WAIT = {"kind": "wait"}         # plan(): possible, but not right now


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
    def __init__(self, g: Gary, model: MacroModel, style: int | None = None, builds: dict | None = None):
        self.g = g
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
        self._learned(obs, mine, done, me)

    def _learned(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> None:
        g = self.g
        hi = g.hi
        if self.goal and hi.frame - self.goal["since"] > GOAL_TIMEOUT:
            g.say(f"gives up on {self.goal['name']}")
            self.goal = None
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
            return
        minerals, gas = plan["cost"]
        budget = me["minerals"] - sum(COST[ut] for ut, *_ in g.reserved)
        walk_ahead = 40 if plan["kind"] == "build" else 0      # send the worker a little early
        if budget < minerals - walk_ahead or me["gas"] < gas:
            return
        if plan["kind"] == "build":
            data = {**plan.get("data", {}), "decision": (goal["act"], goal["id"])}
            g.task = Task("build", goal["id"], plan["where"], data=data, started=hi.frame)
        else:
            g.task = Task("command", data=plan, started=hi.frame)
        self.tracker.note_decision(goal["act"], goal["id"], obs["frame"])
        self.decisions.append((hi.frame, goal["name"], goal["p"]))
        self.goal = None

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
                    (g.task is not None and g.task.kind == "build" and g.task.unit_type == T.DEPOT)
                return WAIT if coming else None
            if uid == T.SCV and sum(1 for u in mine if u["type"] == T.SCV) >= MAX_SCVS:
                return None
            addon = T.NEEDS_ADDON.get(uid)
            if addon is not None and not any(u["type"] == addon for u in done):
                return None                       # e.g. tanks: no Machine Shop yet
            b = self.free_building(obs, mine, done, T.PRODUCER[uid], addon)
            return self._command_or_wait(b, done, T.PRODUCER[uid], act, uid, T.COST[uid])
        if act == BUILD and uid in T.ADDON_PARENT:
            bare = [b for b in done if b["type"] == T.ADDON_PARENT[uid] and self.addon_of(b, mine) is None]
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
        if g.task is not None and g.task.kind == "build" and g.task.unit_type == unit_type:
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
        budget = me["minerals"] - sum(COST[ut] for ut, *_ in g.reserved)
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
        keep = T.COST[T.BUNKER][0] if not self._bunker_coming(mine) else 0
        spare = lambda ut: me["minerals"] - sum(COST[r] for r, *_ in g.reserved) - keep >= T.COST[ut][0]
        # a depot takes longer than a marine or an SCV: start it while each producer has 2 left
        producers = sum(1 for u in done if u["type"] in (T.CC, T.RAX))
        if me["supply_max"] - me["supply_used"] < 2 * producers and me["supply_max"] < 200 and \
                not coming(T.DEPOT) and doable(BUILD, T.DEPOT):
            return goal(BUILD, T.DEPOT, "Supply Depot")
        if doable(TRAIN, T.MARINE) and spare(T.MARINE):
            return goal(TRAIN, T.MARINE, "Marine")
        if sum(1 for u in mine if u["type"] == T.SCV) < RUSH_SCVS and doable(TRAIN, T.SCV) and spare(T.SCV):
            return goal(TRAIN, T.SCV, "SCV")
        budget = me["minerals"] - sum(COST[ut] for ut, *_ in g.reserved)
        if not keep and budget >= RAX_MONEY and sum(1 for u in mine if u["type"] == T.RAX) < RUSH_RAX and not coming(T.RAX) and \
                doable(BUILD, T.RAX):
            return goal(BUILD, T.RAX, "Barracks")
        return None

    def _bunker_coming(self, mine: list[dict]) -> bool:
        g = self.g
        return any(u["type"] == T.BUNKER for u in mine) or any(ut == T.BUNKER for ut, *_ in g.reserved) or \
            (g.task is not None and g.task.kind == "build" and g.task.unit_type == T.BUNKER)
