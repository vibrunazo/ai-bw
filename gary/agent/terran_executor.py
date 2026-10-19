"""The Terran executor (docs/REFACTOR.md Phase 1): what only Terran hands do, over the race-neutral
Executor. SCVs finish buildings nobody builds; add-ons stand to the right of their building, built by
it, and other buildings keep off their spot; depots, the refinery's workers and the Comsat are the
macro layer's and the fight layer's, through these same hands."""

from __future__ import annotations

import math

from gary import orders as O
from gary import terran as T
from gary.agent.executor import BUSY_ORDERS, RESUME_AFTER, RESUME_RETRY, Executor
from gary.agent.gary import Gary, Task
from gary.interface import screen_of


class TerranExecutor(Executor):
    def __init__(self, g: Gary):
        super().__init__(g)
        self.stalled_since: dict[int, int] = {}       # unfinished building -> since when nobody builds it
        self.basic["resume"] = self._task_resume
        g.task_runners["resume"] = self._run_basic

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
