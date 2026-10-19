"""The Terran executor (docs/REFACTOR.md Phase 1): what only Terran hands do, over the race-neutral
Executor. SCVs finish buildings nobody builds; add-ons stand to the right of their building, built by
it, and other buildings keep off their spot; depots, the refinery's workers and the Comsat are the
macro layer's and the fight layer's, through these same hands."""

from __future__ import annotations

import math

from gary import orders as O
from gary import terran as T
from gary.agent.executor import BUSY_BUILDING, BUSY_ORDERS, RESUME_AFTER, RESUME_RETRY, Executor
from gary.agent.macro import BUILD, EXPAND, RESEARCH, TRAIN, UPGRADE, WAIT
from gary.agent.gary import Gary, Task
from gary.interface import screen_of

MAX_SCVS = 60                        # SCVs at most, whatever the bases' saturation


class TerranExecutor(Executor):
    def __init__(self, g: Gary):
        super().__init__(g)
        self.stalled_since: dict[int, int] = {}       # unfinished building -> since when nobody builds it
        self.basic["resume"] = self._task_resume
        g.task_runners["resume"] = self._run_basic

    def carry_out(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        """What a macro decision means for Terran hands now (LearnedMacro.plan's gates passed): a command
        to a building (a unit, an add-on, research, an upgrade), a build task for an SCV, WAIT (the
        building that makes it is busy, or a depot is coming), or None (Gary can't)."""
        g = self.g
        have = {u["type"] for u in done}
        if act == TRAIN and uid in g.race.producer:
            if me["supply_used"] + g.race.supply[uid] > me["supply_max"]:
                # supply blocked: wait if a depot is on its way, else let a depot come next
                coming = any(u["type"] == T.DEPOT and not u["completed"] for u in mine) or \
                    any(ut == T.DEPOT for ut, *_ in g.reserved) or \
                    g.building(T.DEPOT)
                return WAIT if coming else None
            if uid == g.race.worker and sum(1 for u in mine if u["type"] == g.race.worker) >= min(MAX_SCVS, self.saturation(mine)):
                return None
            addon = T.NEEDS_ADDON.get(uid)
            if addon is not None and not any(u["type"] == addon for u in done):
                return None                       # e.g. tanks: no Machine Shop yet
            b = self.free_building(obs, mine, done, g.race.producer[uid], addon)
            return self._command_or_wait(b, done, g.race.producer[uid], act, uid, g.race.costs[uid])
        if act == BUILD and uid in T.ADDON_PARENT:
            ox, oy = T.ADDON_OFFSET                      # (only where the add-on fits)
            fits = lambda b: g.hi.game.can_place(g.slot, uid, (b["x"] - g.race.sizes[b["type"]][0] * 16) // 32 + ox,
                                                 (b["y"] - g.race.sizes[b["type"]][1] * 16) // 32 + oy, 0)
            bare = [b for b in done if b["type"] == T.ADDON_PARENT[uid] and self.addon_of(b, mine) is None and fits(b)]
            if not bare:
                return None
            free = [b for b in bare if b.get("queue", 0) == 0 and b["order"] not in BUSY_BUILDING]
            return {"kind": "command", "act": act, "id": uid, "building": free[0], "cost": g.race.costs[uid]} \
                if free else WAIT
        if act == BUILD and uid == g.race.gas_building:
            spot = self.free_geyser(obs, mine)
            return {"kind": "build", "where": spot[0], "data": {"spot": spot[1]}, "cost": g.race.costs[uid]} \
                if spot else None
        if act == BUILD and uid in g.race.sizes and uid != g.race.town_hall:
            return {"kind": "build", "where": "main", "cost": g.race.costs[uid]}
        if act == EXPAND and uid == g.race.town_hall:
            base = self.free_base(obs)
            return {"kind": "build", "where": "natural" if base is g.natural else base,
                    "cost": g.race.costs[g.race.town_hall]} if base else None
        if act == RESEARCH and uid in g.race.research and uid not in g.researched:
            where, cost = g.race.research[uid]
            b = self.free_building(obs, mine, done, where)
            return self._command_or_wait(b, done, where, act, uid, cost)
        if act == UPGRADE and uid in g.race.upgrade:
            where, cost, levels = g.race.upgrade[uid]
            level = g.upgraded[uid]
            if level >= levels or (level >= 1 and T.UPGRADE_LEVEL_2_NEEDS not in have):
                return None
            b = self.free_building(obs, mine, done, where)
            cost = (cost[0] + 75 * level, cost[1] + 75 * level)
            return self._command_or_wait(b, done, where, act, uid, cost)
        return None

    @staticmethod
    def addon_of(b: dict, mine: list[dict]) -> dict | None:
        """The add-on attached to a building (it sits just right of the building's lower half)."""
        ax, ay = b["x"] + 96, b["y"] + 16
        return next((u for u in mine if u["type"] in T.ADDON_PARENT and math.dist((u["x"], u["y"]), (ax, ay)) < 24),
                    None)

    def supply_due(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> tuple[int, str, bool] | None:
        """A depot ahead of production, as the scripted builds do: started while each producer has less
        than 2 supply left. Pros are almost never supply-blocked, so the model has hardly seen the state
        and gives the depot under its floor once there: Gary-T was blocked in 42 of 48 games at 4-8
        minutes, 437 of its 749 decisions while blocked "nothing it can do" (eval/why.py records, L5b)."""
        g = self.g
        producers = sum(1 for u in done if u["type"] in (T.CC, T.RAX, T.FACTORY, T.STARPORT))
        coming = any(u["type"] == T.DEPOT and not u["completed"] for u in mine) or \
            any(ut == T.DEPOT for ut, *_ in g.reserved) or g.building(T.DEPOT)
        if me["supply_max"] - me["supply_used"] < 2 * producers and not coming \
                and g.macro.plan(BUILD, T.DEPOT, obs, mine, done, me) not in (None, WAIT):
            return BUILD, f"rule: a depot ahead of production ({me['supply_max'] - me['supply_used']} left, {producers} producers)", \
                False
        return None

    def site_ok(self, unit_type: int, tx: int, ty: int) -> bool:
        """No creep under it, and off the spot of an add-on to come."""
        return not self.on_addon_spot(unit_type, tx, ty) and not self.creep_coming(unit_type, tx, ty)

    def command_build(self, obs: dict, b: dict, uid: int) -> bool:
        """An add-on, next to the selected building."""
        hi = self.g.hi
        tx, ty = (b["x"] - 64) // 32 + T.ADDON_OFFSET[0], (b["y"] - 48) // 32 + T.ADDON_OFFSET[1]
        p = screen_of(obs, tx * 32 + 16, ty * 32 + 16)
        if not p:
            hi.camera_minimap(b["x"], b["y"])
            return False
        hi.build(uid, *p, order=O.ORDER_PLACE_ADDON)
        return True

    def resume_construction(self, mine: list[dict]) -> bool:
        """An unfinished building nobody builds (its SCV was killed or sent away) gets another SCV
        to finish it, as players do. True if Gary acted."""
        g = self.g
        now = g.hi.frame
        building = {u.get("order_target") for u in mine if u["type"] == g.race.worker}
        stalled = None
        for b in mine:
            if not (106 <= b["type"] <= 175) or b["completed"] or b["type"] in T.ADDON_PARENT or b["tag"] in building:
                self.stalled_since.pop(b["tag"], None)
                continue
            since = self.stalled_since.setdefault(b["tag"], now)
            if stalled is None and now - since >= RESUME_AFTER:
                stalled = b
        if stalled is None:
            return False
        self.stalled_since[stalled["tag"]] = now + RESUME_RETRY - RESUME_AFTER
        g.task = Task("resume", stalled["type"], data={"tag": stalled["tag"], "x": stalled["x"], "y": stalled["y"]},
                      started=now)
        return True

    def on_addon_spot(self, unit_type: int, tx: int, ty: int) -> bool:
        """The building would stand where one of Gary's buildings' add-on goes (a Comsat Station by
        the natural's Command Center was asked for every second for minutes: its bunker stood there)."""
        g = self.g
        w, h = g.race.sizes[unit_type]
        parents = set(T.ADDON_PARENT.values())
        spots = [((u["x"] - g.race.sizes[u["type"]][0] * 16) // 32, (u["y"] - g.race.sizes[u["type"]][1] * 16) // 32)
                 for u in g.hi.observe()["units"] if u["owner"] == g.slot and u["type"] in parents]
        spots += [(rx, ry) for rt, rx, ry in g.reserved if rt in parents]
        ox, oy = T.ADDON_OFFSET
        return any(tx < px + ox + 2 and px + ox < tx + w and ty < py + oy + 2 and py + oy < ty + h for px, py in spots)

    def _task_resume(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Camera on the building, select the nearest free SCV, right-click the building."""
        g, hi, d = self.g, self.g.hi, t.data
        b = next((u for u in mine if u["tag"] == d["tag"]), None)
        if b is None or b["completed"] or hi.frame - t.started > 24 * 8:
            g.task = None
            return
        if t.stage == "start":
            if not screen_of(obs, b["x"], b["y"]):
                hi.camera_minimap(b["x"], b["y"])
                return
            scvs = [u for u in mine if u["type"] == g.race.worker and u["completed"] and u["tag"] not in g.pulled | g.block_units
                    and u["order"] not in BUSY_ORDERS and screen_of(obs, u["x"], u["y"])]
            if not scvs:
                g.task = None
                return
            s = min(scvs, key=lambda u: math.dist((u["x"], u["y"]), (b["x"], b["y"])))
            x, y = screen_of(obs, s["x"], s["y"])
            hi.box(max(0, x - 4), max(0, y - 4), x + 4, y + 4)
            d["scv"] = s["tag"]
            t.stage = "order"
            return
        p = screen_of(obs, b["x"], b["y"])
        if d.get("scv") in obs["selection"] and p:
            hi.right_click(*p)
            g.say(f"sends an SCV to finish the {g.race.names.get(b['type'], 'building')}")
        g.task = None
