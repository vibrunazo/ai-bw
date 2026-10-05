"""The macro executor (layer L2b, docs/CONTROL.md): how macro decisions get done with Gary's hands.

Building placement, workers (mining, gas, building), hotkeys and rallies for production buildings,
finishing buildings nobody builds, and selecting a building to train or research in. Scripted
(provider `scripted/v1`, from Gary v0.1-v0.2). It works on what the decision layers ask for
(build tasks, commands) and on chores it notices itself (idle workers, a refinery short of
workers, a new building without a hotkey).
"""

from __future__ import annotations

import math

from gary import terran as T
from gary.agent.gary import LOC_NATURAL, Gary, Task
from gary.interface import screen_of
from gary.mapinfo import Base

CC, SCV = T.CC, T.SCV
COST = {unit: minerals for unit, (minerals, _gas) in T.COST.items()}
SIZE = T.SIZE                                              # tiles
HK_CC = [4, 5]                        # one hotkey per Command Center (buildings can't share one)
HK_RAX = [6, 7, 8, 9]
BUSY_ORDERS = {30, 31, 33}            # placing / constructing a building
RESUME_AFTER = 24                     # an unfinished building nobody has built for a second gets an SCV
RESUME_RETRY = 24 * 4                 # ...and if that one doesn't arrive either, another after 4 s
IDLE = 3                              # order "PlayerGuard": standing around
RESET_HARVEST = 151                   # the order of a worker stepping out of a refinery (or off a mineral)
TASK_TIMEOUT = 24 * 40                # build, hotkey, mine and resume tasks give up after 40 s
COMMAND_TIMEOUT = 24 * 20             # commands and gas after 20 s


class ScriptedExecutor:
    def __init__(self, g: Gary):
        self.g = g
        self.next_sweep = 0                           # next look for idle workers off screen
        self.stalled_since: dict[int, int] = {}       # unfinished building -> since when nobody builds it
        self._terrain_cache = None                    # see terrain()
        for kind in ("build", "hotkey_building", "mine", "resume"):
            g.task_runners[kind] = self._run_basic
        g.task_runners["command"] = self._run_command
        g.task_runners["gas"] = self._run_command

    # --- chores, in the arbiter's order ------------------------------------------------------

    def release_reservations(self, mine: list[dict]) -> None:
        """A reserved spot is free again once its building exists, or after 40 s if it never
        appeared (the worker was interrupted, or the spot got blocked)."""
        g = self.g
        keep = []
        for r in g.reserved:
            ut, tx, ty = r
            cx, cy = tx * 32 + SIZE[ut][0] * 16, ty * 32 + SIZE[ut][1] * 16
            built = any(u["type"] == ut and math.dist((u["x"], u["y"]), (cx, cy)) < 48 for u in mine)
            if not built and g.hi.frame - g.reserved_at.get(r, 0) < 24 * 40:
                keep.append(r)
                continue
            g.reserved_at.pop(r, None)
        g.reserved = keep

    def resume_construction(self, mine: list[dict]) -> bool:
        """An unfinished building nobody builds (its SCV was killed or sent away) gets another SCV
        to finish it, as players do. True if Gary acted."""
        g = self.g
        now = g.hi.frame
        building = {u.get("order_target") for u in mine if u["type"] == SCV}
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

    def idle_mining(self, obs: dict, done: list[dict]) -> bool:
        """Idle workers on screen go back to mining (game start, after building something)."""
        g = self.g
        idle = [u for u in done if u["type"] == SCV and u["order"] == IDLE and screen_of(obs, u["x"], u["y"])
                and g.ignore_idle_until.get(u["tag"], 0) <= g.hi.frame]
        if idle:
            g.task = Task("mine", data={"tags": [u["tag"] for u in idle]}, started=g.hi.frame)
            return True
        return False

    def idle_sweep(self, obs: dict, done: list[dict]) -> bool:
        """Every 10 s, look at an idle worker the camera isn't showing, so the mining routine
        (which works on what's on screen) can send it back. True if Gary moved the camera."""
        g, hi = self.g, self.g.hi
        if hi.frame < self.next_sweep:
            return False
        busy = {g.scout} | set(g.pulled)
        away = [u for u in done if u["type"] == SCV and u["order"] == IDLE and u["tag"] not in busy
                and not screen_of(obs, u["x"], u["y"]) and g.ignore_idle_until.get(u["tag"], 0) <= hi.frame]
        self.next_sweep = hi.frame + 24 * 10
        if not away:
            return False
        hi.camera_minimap(away[0]["x"], away[0]["y"])
        return True

    def hotkey_new_buildings(self, done: list[dict]) -> bool:
        """New production buildings get a hotkey and a rally point."""
        g = self.g
        for b in done:
            if b["type"] in g.production_types and b["tag"] not in g.hotkeyed:
                g.task = Task("hotkey_building", b["type"], data={"tag": b["tag"], "x": b["x"], "y": b["y"]},
                              started=g.hi.frame)
                return True
        return False

    def refill_gas(self, obs: dict, done: list[dict]) -> bool:
        """A finished refinery with fewer than three workers gets more."""
        g = self.g
        for r in done:
            if r["type"] == T.REFINERY and self.on_gas(obs, r["tag"]) < 3 and \
                    g.ignore_idle_until.get(r["tag"], 0) <= g.hi.frame:
                g.task = Task("gas", data={"tag": r["tag"], "x": r["x"], "y": r["y"]}, started=g.hi.frame)
                return True
        return False

    def adopt_hotkeys(self) -> None:
        """For a game Gary takes over midway (eval/scenarios.py): its production buildings get
        hotkeys as if it had set them itself, earlier in its own game."""
        g = self.g
        obs = g.hi.observe()
        for b in obs["units"]:
            if b["owner"] != g.slot or not b["completed"] or b["tag"] in g.hotkeyed:
                continue
            if b["type"] not in g.production_types:
                continue
            used = set(g.hotkeyed.values())
            hk = next((k for k in (HK_CC if b["type"] == CC else HK_RAX) if k not in used), -1)
            g.hotkeyed[b["tag"]] = hk
            if hk >= 0:
                g.hi.hotkeys[hk] = [b["tag"]]

    # --- workers and bases --------------------------------------------------------------------

    def my_bases(self, mine: list[dict]) -> list[Base]:
        """Bases with one of Gary's finished Command Centers."""
        halls = [u for u in mine if u["type"] == CC and u["completed"]]
        return [b for b in self.g.map.bases if any(math.dist(b.center, (h["x"], h["y"])) < 6 * 32 for h in halls)]

    def free_worker(self, obs: dict, u: dict) -> bool:
        """A finished SCV on screen that isn't busy placing or constructing a building."""
        return u["type"] == SCV and u["completed"] and u["order"] not in BUSY_ORDERS \
            and bool(screen_of(obs, u["x"], u["y"]))

    def pick_worker(self, obs: dict, mine: list[dict]) -> dict | None:
        cands = [u for u in mine if self.free_worker(obs, u)]
        return min(cands, key=lambda u: math.dist((u["x"], u["y"]), self.g.main.center)) if cands else None

    def on_gas(self, obs: dict, refinery: int) -> int:
        """Workers on this refinery: the ones Gary sent that are still on gas (on a gas order, just
        out of the refinery, or out of sight: a worker inside a refinery isn't visible), and any
        worker seen heading into it (the SCV that built it starts on gas by itself)."""
        g = self.g
        seen = {u["tag"]: u for u in obs["units"] if u["owner"] == g.slot}
        alive = {w for w in g.gas_workers.get(refinery, set())
                 if w not in seen or seen[w]["order"] in T.GAS_ORDERS or seen[w]["order"] == RESET_HARVEST}
        alive |= {t for t, u in seen.items() if u["type"] == T.SCV and u["order"] in T.GAS_ORDERS
                  and u.get("order_target") == refinery}
        g.gas_workers[refinery] = alive
        return len(alive)

    # --- placement ------------------------------------------------------------------------------

    def spot(self, unit_type: int, base: Base, builder: int) -> tuple[int, int] | None:
        """A spot at the base, or at another of Gary's bases when that one is full."""
        spot = self._base_spot(unit_type, base, builder)
        if spot is None:
            for other in self.my_bases([u for u in self.g.hi.observe()["units"] if u["owner"] == self.g.slot]):
                if other is not base and (spot := self._base_spot(unit_type, other, builder)):
                    break
        return spot

    def _base_spot(self, unit_type: int, base: Base, builder: int) -> tuple[int, int] | None:
        """A free spot near the base, on the side away from the minerals."""
        mx = sum(x for x, _ in base.minerals) / len(base.minerals)
        my = sum(y for _, y in base.minerals) / len(base.minerals)
        ax = base.center[0] + (base.center[0] - mx) * 0.9
        ay = base.center[1] + (base.center[1] - my) * 0.9
        w, h = SIZE[unit_type]
        cands = []
        for ty in range(int(ay // 32) - 16, int(ay // 32) + 16):
            for tx in range(int(ax // 32) - 16, int(ax // 32) + 16):
                cands.append((math.dist((tx * 32 + w * 16, ty * 32 + h * 16), (ax, ay)), tx, ty))
        for _, tx, ty in sorted(cands):
            if any(abs(tx - rx) < SIZE[rt][0] + 1 and abs(ty - ry) < SIZE[rt][1] + 1 for rt, rx, ry in self.g.reserved):
                continue
            # keep a walkable gap around buildings: the bigger footprint must fit too
            if self.fits(unit_type, tx, ty, builder) and self.fits(unit_type, tx - 1, ty - 1, builder) and \
                    self.fits(unit_type, tx + 1, ty + 1, builder):
                return tx, ty
        return None

    def spot_near(self, unit_type: int, near: tuple[int, int], builder: int,
                  level_at: tuple[int, int] | None = None) -> tuple[int, int] | None:
        """The free spot closest to a point (e.g. a bunker at the top of the main ramp), on the ground
        level of level_at (e.g. the main: the closest free tile to the top of a ramp can be at its
        bottom). On a flat map every tile is on that level; if no spot on it fits, any level does."""
        w, h = SIZE[unit_type]
        cx, cy = near[0] // 32, near[1] // 32
        terrain = self.terrain()
        level = int(terrain[1][level_at[1] // 8, level_at[0] // 8]) if terrain and level_at else None
        cands = sorted((math.dist((tx * 32 + w * 16, ty * 32 + h * 16), near), tx, ty)
                       for ty in range(cy - 8, cy + 8) for tx in range(cx - 8, cx + 8))
        for same_level in ((True, False) if level is not None else (False,)):
            for _, tx, ty in cands:
                if any(abs(tx - rx) < SIZE[rt][0] + 1 and abs(ty - ry) < SIZE[rt][1] + 1 for rt, rx, ry in self.g.reserved):
                    continue
                if same_level and (terrain[1][ty * 4:(ty + h) * 4, tx * 4:(tx + w) * 4] != level).any():
                    continue
                if self.fits(unit_type, tx, ty, builder):
                    return tx, ty
        return None

    def fits(self, unit_type: int, tx: int, ty: int, builder: int) -> bool:
        """The game accepts the building there, and the terrain under it is buildable. The game's
        check skips the terrain of fogged tiles (you can order a build into the fog): a spot at the
        fogged edge of a ramp passes, then fails once a unit sees it and the SCV keeps mining. The
        map's terrain is map knowledge, so Gary checks it too (where the env can tell it)."""
        if not self.g.hi.game.can_place(self.g.slot, unit_type, tx, ty, builder):
            return False
        terrain = self.terrain()
        if terrain is None:
            return True
        w, h = SIZE[unit_type]
        return bool(terrain[0][ty:ty + h, tx:tx + w].all())

    def terrain(self):
        """The map's buildable tiles (no 8x8 part unwalkable) and ground height per 8x8 minitile,
        or None where the env can't tell (the live Remastered client)."""
        if self._terrain_cache is None:
            terrain = getattr(self.g.hi.game, "terrain", None)
            if terrain is None:
                self._terrain_cache = False
            else:
                walk, height, build = terrain()
                th, tw = build.shape
                self._terrain_cache = (build & walk.reshape(th, 4, tw, 4).all(axis=(1, 3)), height)
        return self._terrain_cache or None

    def town_hall_spot(self, base: Base, builder: int) -> tuple[int, int]:
        """The base's town-hall tile, or the nearest tile around it the game accepts (map knowledge
        can be a tile off)."""
        tx, ty = base.tile
        near = sorted(((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)), key=lambda d: abs(d[0]) + abs(d[1]))
        for dx, dy in near:
            if self.fits(CC, tx + dx, ty + dy, builder):
                return tx + dx, ty + dy
        return base.tile

    # --- tasks ----------------------------------------------------------------------------------

    def _run_basic(self, obs: dict, mine: list[dict], t: Task) -> None:
        g = self.g
        if g.hi.frame - t.started > TASK_TIMEOUT:     # give up on anything stuck for 40 s
            if t.kind == "hotkey_building":
                g.hotkeyed[t.data["tag"]] = -1        # live without a hotkey for this one
            if t.kind == "build":
                return self.fail_build(t)
            g.task = None
            return
        {"build": self._task_build, "hotkey_building": self._task_hotkey, "mine": self._task_mine,
         "resume": self._task_resume}[t.kind](obs, mine, t)

    def _run_command(self, obs: dict, mine: list[dict], t: Task) -> None:
        if self.g.hi.frame - t.started > COMMAND_TIMEOUT:
            self.g.task = None
            return
        (self._task_command if t.kind == "command" else self._task_gas)(obs, mine, t)

    def fail_build(self, t: Task) -> None:
        """A building Gary decided on couldn't be placed: the macro layer hears of it."""
        if self.g.macro is not None:
            self.g.macro.build_failed(t)
        self.g.task = None

    def _task_build(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi = self.g, self.g.hi
        if t.stage == "start":                      # get a worker from the main
            if not g.look_at(obs, "main"):
                return
            w = t.data.get("worker")
            if w and obs["selection"] == [w]:
                t.stage = "place" if t.where == "main" else "travel"
                return
            # the click landed on a neighbor in a stack (the one drawn on top wins): a human just
            # uses whichever free SCV got selected instead of clicking again
            if w and len(obs["selection"]) == 1:
                got = next((u for u in mine if u["tag"] == obs["selection"][0]), None)
                if got and self.free_worker(obs, got):
                    t.data["worker"] = got["tag"]
                    t.stage = "place" if t.where == "main" else "travel"
                    return
            w = self.pick_worker(obs, mine)
            if not w:
                self.fail_build(t)
                return
            t.data["worker"] = w["tag"]
            p = screen_of(obs, w["x"], w["y"])
            hi.click(*p)
            return
        if t.stage == "travel":                      # walk the worker to the base
            hi.minimap_right_click(*g.base(t.where).center)
            t.stage = "wait_arrival"
            return
        if t.stage == "wait_arrival":
            w = next((u for u in mine if u["tag"] == t.data["worker"]), None)
            if not w:
                self.fail_build(t)
                return
            if math.dist((w["x"], w["y"]), g.base(t.where).center) > 6 * 32:
                return
            if not g.look_at(obs, t.where):
                return
            if t.where == "natural" and LOC_NATURAL not in hi.camera_locations:
                hi.camera_location_set(LOC_NATURAL)
                return
            t.stage = "place"
            return
        if t.stage == "place":
            budget = obs["me"]["minerals"] - sum(COST[ut] for ut, *_ in g.reserved)
            if budget < COST[t.unit_type]:
                return                                  # wait for money with the worker selected
            base = g.base(t.where)
            if "spot" not in t.data:
                if not g.look_at(obs, t.where):
                    return
                spot = self.town_hall_spot(base, t.data["worker"]) if t.unit_type == CC else \
                    self.spot_near(t.unit_type, t.data["near"], t.data["worker"], t.data.get("level_at")) if "near" in t.data else \
                    self.spot(t.unit_type, base, t.data["worker"])
                if not spot:
                    self.fail_build(t)
                    return
                t.data["spot"] = spot
            spot = t.data["spot"]
            w, h = SIZE[t.unit_type]
            p = screen_of(obs, spot[0] * 32 + 16, spot[1] * 32 + 16)
            if not p:                                   # the spot is off screen: look there first
                if t.data.get("looked"):
                    self.fail_build(t)
                    return
                t.data["looked"] = True
                hi.camera_minimap(spot[0] * 32 + w * 16, spot[1] * 32 + h * 16)
                return
            if hi.build(t.unit_type, *p).accepted:
                g.reserved.append((t.unit_type, *spot))
                g.reserved_at[(t.unit_type, *spot)] = hi.frame
                g.task = None

    def _task_hotkey(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi = self.g, self.g.hi
        tag = t.data["tag"]
        where = "natural" if math.dist((t.data["x"], t.data["y"]), g.natural.center) < 10 * 32 else "main"
        if obs["selection"] == [tag]:
            if t.stage == "start":
                pool = HK_CC if t.unit_type == CC else HK_RAX
                used = set(g.hotkeyed.values())
                hk = next((k for k in pool if k not in used), None)
                if hk is None:
                    g.hotkeyed[tag] = -1
                    g.task = None
                    return
                hi.hotkey_set(hk)
                g.hotkeyed[tag] = hk
                t.stage = "rally"
                return
            if t.stage == "rally":                      # CCs rally to minerals (auto-mine), barracks forward
                if t.unit_type == CC:
                    base = g.main if where == "main" else g.natural
                    m = base.minerals[len(base.minerals) // 2]
                    p = screen_of(obs, *m)
                    if p:
                        hi.right_click(*p)
                else:
                    hi.minimap_right_click(*g.rally)
                g.task = None
                return
        if not screen_of(obs, t.data["x"], t.data["y"]):
            hi.camera_minimap(t.data["x"], t.data["y"])
            return
        hi.click(*screen_of(obs, t.data["x"], t.data["y"]))

    def _task_mine(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Select idle workers (drag box around them, or a click) and right-click a mineral field."""
        g, hi = self.g, self.g.hi
        tags = set(t.data["tags"])
        workers = [u for u in mine if u["tag"] in tags and screen_of(obs, u["x"], u["y"])]
        if not workers:
            g.task = None
            return
        if t.stage == "start":
            pts = [screen_of(obs, u["x"], u["y"]) for u in workers]
            # a drag box only picks up own units: a plain click on a worker standing on a
            # mineral field can select the field instead
            hi.box(max(0, min(p[0] for p in pts) - 10), max(0, min(p[1] for p in pts) - 10),
                   min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + 10),
                   min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + 10))
            t.stage = "send"
            return
        # the box may have caught busy workers too (e.g. one walking off to build): never send
        # those back to mining; select an idle one with a click instead
        if not obs["selection"] or not set(obs["selection"]) <= tags:
            if t.stage == "send":
                t.stage = "click"
                hi.click(*screen_of(obs, workers[0]["x"], workers[0]["y"]))
                return
            for tag in tags:
                g.ignore_idle_until[tag] = hi.frame + 24 * 20
            g.task = None
            return
        # the closest mineral field to the Command Center of the nearest base Gary owns (a field
        # at a base without a Command Center means a long walk with every load)
        w = workers[0]
        bases = self.my_bases(mine) or [g.main]
        base = min(bases, key=lambda b: math.dist(b.center, (w["x"], w["y"])))
        fields = [m for m in base.minerals if screen_of(obs, *m)] or \
                 [m for b in bases for m in b.minerals if screen_of(obs, *m)]
        if fields:
            m = min(fields, key=lambda m: math.dist(m, base.center))
            hi.right_click(*screen_of(obs, *m))
        # whether or not it worked, don't fixate on these workers: look again in 20 s
        for tag in tags:
            g.ignore_idle_until[tag] = hi.frame + 24 * 20
        g.task = None

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
            scvs = [u for u in mine if u["type"] == SCV and u["completed"] and u["tag"] not in g.pulled
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
            g.say(f"sends an SCV to finish the {T.NAMES.get(b['type'], 'building')}")
        g.task = None

    def select(self, obs: dict, b: dict) -> bool:
        """Get this building selected (hotkey, or look at it and click). True once it is."""
        g, hi = self.g, self.g.hi
        if obs["selection"] == [b["tag"]]:
            return True
        hk = g.hotkeyed.get(b["tag"], -1)
        if hk >= 0:
            hi.hotkey_recall(hk)
            return False
        p = screen_of(obs, b["x"], b["y"])
        if p:
            hi.click(*p)
        else:
            hi.camera_minimap(b["x"], b["y"])
        return False

    def _task_command(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Train, research, upgrade or build an add-on: select the building, press the hotkey."""
        from gary.agent.macro import BUILD, RESEARCH, TRAIN, UPGRADE
        g, hi, d = self.g, self.g.hi, t.data
        b = next((u for u in mine if u["tag"] == d["building"]["tag"]), None)
        if b is None:
            g.task = None
            return
        if not self.select(obs, b):
            return
        act, uid = d["act"], d["id"]
        if act == TRAIN:
            if hi.train(uid).accepted:
                g.producing[b["tag"]] = uid
                g.ordered_at[b["tag"]] = obs["now"]
        elif act == RESEARCH:
            if hi.research(uid).accepted:
                g.researched.add(uid)
        elif act == UPGRADE:
            if hi.upgrade(uid).accepted:
                g.upgraded[uid] += 1
        elif act == BUILD:                      # an add-on, next to the selected building
            tx, ty = (b["x"] - 64) // 32 + T.ADDON_OFFSET[0], (b["y"] - 48) // 32 + T.ADDON_OFFSET[1]
            p = screen_of(obs, tx * 32 + 16, ty * 32 + 16)
            if not p:
                hi.camera_minimap(b["x"], b["y"])
                return
            hi.build(uid, *p, order=T.ORDER_PLACE_ADDON)
        g.task = None

    def _task_gas(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Send workers to a finished refinery, a few at a time, until three are on it."""
        g, hi, d = self.g, self.g.hi, t.data
        ref = next((u for u in mine if u["tag"] == d["tag"]), None)
        if ref is None or self.on_gas(obs, d["tag"]) >= 3:
            g.task = None
            return
        rp = screen_of(obs, ref["x"], ref["y"])
        if not rp:
            hi.camera_minimap(ref["x"], ref["y"])
            return
        if t.stage == "send":
            picked = [u["tag"] for u in mine if u["tag"] in obs["selection"] and u["type"] == T.SCV]
            if picked:
                hi.right_click(*rp)
                g.gas_workers.setdefault(d["tag"], set()).update(picked)
            t.stage = "start"
            d["tries"] = d.get("tries", 0) + 1
            if d["tries"] >= 5:                 # look again in 20 s rather than keep trying now
                g.ignore_idle_until[d["tag"]] = hi.frame + 24 * 20
                g.task = None
            return
        workers = [u for u in mine if self.free_worker(obs, u) and u["order"] not in T.GAS_ORDERS]
        if not workers:
            g.task = None
            return
        w = min(workers, key=lambda u: math.dist((u["x"], u["y"]), (ref["x"], ref["y"])))
        x, y = screen_of(obs, w["x"], w["y"])
        # a small drag box: a click on a mining worker can select its mineral field instead
        hi.box(max(0, x - 6), max(0, y - 6), min(hi.p.viewport[0] - 1, x + 6), min(hi.p.viewport[1] - 1, y + 6))
        t.stage = "send"
