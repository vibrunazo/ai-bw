"""Gary v0.1, Terran: a scripted one-barracks expand into marines, played with human hands.

Everything goes through the human interface (gary/interface.py): Gary sees with fog of war and a
reaction delay, moves the camera with location hotkeys and the minimap, selects by clicking and
drag boxes (no multi-building selection, like the real game), and spends APM.

Plan (the taxonomy's most common TvZ Terran opening, simplified):
  supply 9 depot, 11 barracks, 15 Command Center at the natural, 16 depot, 18 and 20 barracks;
  then constant SCVs (two bases), marines from every barracks, depots before supply runs out,
  and an attack-move to the enemy start with every 16 marines.
Not yet: gas, academy, medics, upgrades, scouting, defense, retreating, micro.

    python -m gary.bots.terran_v01 --map path/to/map.scx --minutes 10
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

from gary import commands as C
from gary import terran as T
from gary.env import LIVE_COMMAND_DELAY, Game, GameError
from gary.interface import PROFILES, HumanInterface, screen_of
from gary.mapinfo import Base, MapInfo
from gary.version import announcement

DEPOT, RAX, CC, SCV, MARINE = C.SUPPLY_DEPOT, C.BARRACKS, C.COMMAND_CENTER, C.SCV, C.MARINE
COST = {unit: minerals for unit, (minerals, _gas) in T.COST.items()}
SIZE = T.SIZE                                              # tiles
BUILD_ORDER = [(9, DEPOT, "main"), (11, RAX, "main"), (15, CC, "natural"),
               (16, DEPOT, "main"), (18, RAX, "main"), (20, RAX, "main")]
MAX_SCVS = 36
WAVE = 16
LOC_MAIN, LOC_NATURAL = 2, 3          # F2, F3
HK_CC = [4, 5]                        # one hotkey per Command Center (buildings can't share one)
HK_RAX = [6, 7, 8, 9]
HK_ARMY = [1, 2]
BUSY_ORDERS = {30, 31, 33}            # placing / constructing a building
RESUME_AFTER = 24                     # an unfinished building nobody has built for a second gets an SCV
RESUME_RETRY = 24 * 4                 # ...and if that one doesn't arrive either, another after 4 s
IDLE = 3                              # order "PlayerGuard": standing around


@dataclass
class Task:
    kind: str                          # "build", "hotkey_building", "attack"
    unit_type: int = -1
    where: str | Base = "main"            # "main", "natural" or any Base
    stage: str = "start"
    data: dict = field(default_factory=dict)
    started: int = 0


class TerranGary:
    army_types = {MARINE}                 # what joins the attack waves
    version = "v0.1"
    production_types = (CC, RAX)             # buildings that get a hotkey

    def __init__(self, hi: HumanInterface, mapinfo: MapInfo):
        self.hi = hi
        self.map = mapinfo
        self.slot = hi.slot
        self._terrain_cache = None          # see _terrain
        obs = hi.observe()
        cc = next(u for u in obs["units"] if u["owner"] == self.slot and u["type"] == CC)
        self.main = mapinfo.base_near((cc["x"], cc["y"]))
        self.natural = mapinfo.natural_of(self.main)
        self.enemy_starts = mapinfo.enemy_starts(self.main) or mapinfo.starts
        self.rally = self._toward_center(self.natural.center, 6 * 32)
        self.order_step = 0
        self.order_retry: list[int] = []          # build-order steps whose building never got placed
        self.retry_after = 0                      # ...redone from this frame (a failure waits 5 s)
        self.task: Task | None = None
        self.hotkeyed: dict[int, int] = {}        # building tag -> hotkey
        self.ordered_at: dict[int, int] = {}      # building tag -> frame of its last train order
        self.rallied: set[int] = set()
        self.reserved: list[tuple[int, int, int]] = []   # (unit type, tile x, tile y) being built
        self.reserved_at: dict[tuple[int, int, int], int] = {}
        self.reserved_step: dict[tuple[int, int, int], int] = {}   # reservation -> its build-order step
        self.attacks_sent = 0
        self.army_sent: set[int] = set()
        self.setup_done = False
        self.announce = announcement(self.version)   # said in the chat at the start of the game
        self.ignore_idle_until: dict[int, int] = {}   # workers we failed to send: retry later
        self.next_sweep = 0                           # next look for idle workers off screen
        self.stalled_since: dict[int, int] = {}       # unfinished building -> since when nobody builds it

    # --- geometry ------------------------------------------------------------------------

    def _toward_center(self, p: tuple[int, int], d: int) -> tuple[int, int]:
        cx, cy = self.map.size[0] / 2, self.map.size[1] / 2
        n = math.dist(p, (cx, cy)) or 1
        return (round(p[0] + (cx - p[0]) * d / n), round(p[1] + (cy - p[1]) * d / n))

    def _spot(self, unit_type: int, base: Base, builder: int) -> tuple[int, int] | None:
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
            if any(abs(tx - rx) < SIZE[rt][0] + 1 and abs(ty - ry) < SIZE[rt][1] + 1 for rt, rx, ry in self.reserved):
                continue
            # keep a walkable gap around buildings: the bigger footprint must fit too
            if self._fits(unit_type, tx, ty, builder) and self._fits(unit_type, tx - 1, ty - 1, builder) and \
                    self._fits(unit_type, tx + 1, ty + 1, builder):
                return tx, ty
        return None

    def _spot_near(self, unit_type: int, near: tuple[int, int], builder: int) -> tuple[int, int] | None:
        """The free spot closest to a point (e.g. a bunker at the top of the main ramp), on the same
        ground level as the point: the closest free tile to the top of a ramp can be at its bottom."""
        w, h = SIZE[unit_type]
        cx, cy = near[0] // 32, near[1] // 32
        terrain = self._terrain()
        level = int(terrain[1][near[1] // 8, near[0] // 8]) if terrain else None
        cands = sorted((math.dist((tx * 32 + w * 16, ty * 32 + h * 16), near), tx, ty)
                       for ty in range(cy - 8, cy + 8) for tx in range(cx - 8, cx + 8))
        for _, tx, ty in cands:
            if any(abs(tx - rx) < SIZE[rt][0] + 1 and abs(ty - ry) < SIZE[rt][1] + 1 for rt, rx, ry in self.reserved):
                continue
            if level is not None and (terrain[1][ty * 4:(ty + h) * 4, tx * 4:(tx + w) * 4] != level).any():
                continue
            if self._fits(unit_type, tx, ty, builder):
                return tx, ty
        return None

    def _fits(self, unit_type: int, tx: int, ty: int, builder: int) -> bool:
        """The game accepts the building there, and the terrain under it is buildable. The game's
        check skips the terrain of fogged tiles (you can order a build into the fog): a spot at the
        fogged edge of a ramp passes, then fails once a unit sees it and the SCV keeps mining. The
        map's terrain is map knowledge, so Gary checks it too (where the env can tell it)."""
        if not self.hi.game.can_place(self.slot, unit_type, tx, ty, builder):
            return False
        terrain = self._terrain()
        if terrain is None:
            return True
        w, h = SIZE[unit_type]
        return bool(terrain[0][ty:ty + h, tx:tx + w].all())

    def _terrain(self):
        """The map's buildable tiles (no 8x8 part unwalkable) and ground height per 8x8 minitile,
        or None where the env can't tell (the live Remastered client)."""
        if self._terrain_cache is None:
            terrain = getattr(self.hi.game, "terrain", None)
            if terrain is None:
                self._terrain_cache = False
            else:
                walk, height, build = terrain()
                th, tw = build.shape
                self._terrain_cache = (build & walk.reshape(th, 4, tw, 4).all(axis=(1, 3)), height)
        return self._terrain_cache or None

    # --- camera ---------------------------------------------------------------------------

    def _base(self, where: str | Base) -> Base:
        return self.main if where == "main" else self.natural if where == "natural" else where

    def _look_at(self, obs: dict, where: str | Base) -> bool:
        """Make sure the base ("main", "natural" or any Base) is on screen. True if it already is
        (no action taken)."""
        base = self._base(where)
        if screen_of(obs, *base.center):
            return True
        loc = {"main": LOC_MAIN, "natural": LOC_NATURAL}.get(where) if isinstance(where, str) else None
        if loc in self.hi.camera_locations:
            self.hi.camera_location(loc)
        else:
            self.hi.camera_minimap(*base.center)
        return False

    # --- main loop ------------------------------------------------------------------------

    def act(self) -> None:
        hi = self.hi
        if hi.pending:
            return
        obs = hi.observe()
        if not self.setup_done:
            return self._setup(obs)
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == self.slot]
        done = [u for u in mine if u["completed"]]
        self._release_reservations(mine)

        if self.task:
            return self._run_task(obs, mine)
        if self._resume_construction(mine):
            return

        # idle workers go back to mining (game start, after building something)
        idle = [u for u in done if u["type"] == SCV and u["order"] == IDLE and screen_of(obs, u["x"], u["y"])
                and self.ignore_idle_until.get(u["tag"], 0) <= hi.frame]
        if idle:
            self.task = Task("mine", data={"tags": [u["tag"] for u in idle]}, started=hi.frame)
            return
        if self._idle_sweep(obs, done):
            return
        # new buildings get a hotkey and a rally point
        for b in done:
            if b["type"] in self.production_types and b["tag"] not in self.hotkeyed:
                self.task = Task("hotkey_building", b["type"], data={"tag": b["tag"], "x": b["x"], "y": b["y"]},
                                 started=hi.frame)
                return
        # build order, then supply
        supply = me["supply_used"]
        # minerals already promised to buildings ordered but not started yet
        budget = me["minerals"] - sum(COST[ut] for ut, *_ in self.reserved)
        # (a failed step is redone before moving on, like a human who sees the depot never went down)
        step = self.order_retry[0] if self.order_retry else self.order_step
        due = step < len(BUILD_ORDER) and supply >= BUILD_ORDER[step][0] and             (not self.order_retry or hi.frame >= self.retry_after)
        if due:
            _, ut, where = BUILD_ORDER[step]
            if budget >= COST[ut] - 30:
                if self.order_retry:
                    self.order_retry.pop(0)
                else:
                    self.order_step += 1
                self.task = Task("build", ut, where, data={"step": step}, started=hi.frame)
                return
        # supply coming: unfinished depots (8) and Command Centers (10) count already
        coming = sum(8 for u in mine if u["type"] == DEPOT and not u["completed"])
        coming += sum(10 for u in mine if u["type"] == CC and not u["completed"])
        coming += sum(8 for ut, *_ in self.reserved if ut == DEPOT)
        n_prod = sum(1 for u in mine if u["type"] in (CC, RAX) and u["completed"])
        if (me["supply_max"] + coming < 200 and self.order_step >= 2
                and me["supply_max"] + coming - supply <= 2 + 2 * n_prod and budget >= 100):
            self.task = Task("build", DEPOT, "main", started=hi.frame)
            return
        # floating money: more barracks (up to 8)
        n_rax = sum(1 for u in mine if u["type"] == RAX) + sum(1 for ut, *_ in self.reserved if ut == RAX)
        if self.order_step >= len(BUILD_ORDER) and budget >= 400 and n_rax < 8:
            self.task = Task("build", RAX, "main", started=hi.frame)
            return
        # attack with every full wave
        marines = [u for u in done if u["type"] in self.army_types and u["tag"] not in self.army_sent]
        if len(marines) >= WAVE:
            self.task = Task("attack", started=hi.frame)
            return
        if due:
            return              # save up for the next building instead of spending on units
        self._produce(obs, done, me, budget)

    def _setup(self, obs: dict) -> None:
        """Say which Gary this is in the chat, then remember the main as F2 (the main CC gets its
        hotkey through the normal path)."""
        if self.announce:
            self.hi.chat(self.announce.pop(0))
            return
        if not self._look_at(obs, "main"):
            return
        self.hi.camera_location_set(LOC_MAIN)
        self.setup_done = True

    def _produce(self, obs: dict, done: list[dict], me: dict, budget: int) -> None:
        """Cycle building hotkeys: SCVs from Command Centers, marines from barracks."""
        scvs = sum(1 for u in obs["units"] if u["owner"] == self.slot and u["type"] == SCV)
        if me["supply_used"] >= me["supply_max"]:
            return
        # keep one unit queued in each building (a human keeps queues short to save money). Like a
        # human, remember the order just given: it shows only after the reaction delay plus the
        # network delay, and reordering before then queues extra units.
        blind = round(self.hi.p.reaction_ms / 42) + self.hi.act_latency_frames + 4
        idle = [u for u in done if u["type"] in (CC, RAX) and self.hotkeyed.get(u["tag"], -1) >= 0
                and u.get("queue", 0) == 0 and obs["now"] - self.ordered_at.get(u["tag"], -10**9) > blind]
        for b in idle:
            want = SCV if b["type"] == CC else MARINE
            if want == SCV and scvs >= MAX_SCVS:
                continue
            if budget < COST[want]:
                return
            hk = self.hotkeyed[b["tag"]]
            if obs["selection"] != [b["tag"]]:
                self.hi.hotkey_recall(hk)
            elif self.hi.train(want).accepted:
                self.ordered_at[b["tag"]] = obs["now"]
            return

    def _resume_construction(self, mine: list[dict]) -> bool:
        """An unfinished building nobody builds (its SCV was killed or sent away) gets another SCV
        to finish it, as players do. True if Gary acted."""
        now = self.hi.frame
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
        self.task = Task("resume", stalled["type"], data={"tag": stalled["tag"], "x": stalled["x"], "y": stalled["y"]},
                         started=now)
        return True

    def _task_resume(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Camera on the building, select the nearest free SCV, right-click the building."""
        hi, d = self.hi, t.data
        b = next((u for u in mine if u["tag"] == d["tag"]), None)
        if b is None or b["completed"] or hi.frame - t.started > 24 * 8:
            self.task = None
            return
        if t.stage == "start":
            if not screen_of(obs, b["x"], b["y"]):
                hi.camera_minimap(b["x"], b["y"])
                return
            busy = getattr(self, "pulled", set())
            scvs = [u for u in mine if u["type"] == SCV and u["completed"] and u["tag"] not in busy
                    and u["order"] not in BUSY_ORDERS and screen_of(obs, u["x"], u["y"])]
            if not scvs:
                self.task = None
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
            self._say(f"sends an SCV to finish the {T.NAMES.get(b['type'], 'building')}")
        self.task = None

    def adopt_hotkeys(self) -> None:
        """For a game Gary takes over midway (eval/scenarios.py): its production buildings get
        hotkeys as if it had set them itself, earlier in its own game."""
        obs = self.hi.observe()
        for b in obs["units"]:
            if b["owner"] != self.slot or not b["completed"] or b["tag"] in self.hotkeyed:
                continue
            if b["type"] not in self.production_types:
                continue
            used = set(self.hotkeyed.values())
            hk = next((k for k in (HK_CC if b["type"] == CC else HK_RAX) if k not in used), -1)
            self.hotkeyed[b["tag"]] = hk
            if hk >= 0:
                self.hi.hotkeys[hk] = [b["tag"]]

    # --- tasks --------------------------------------------------------------------------------

    def _run_task(self, obs: dict, mine: list[dict]) -> None:
        t = self.task
        if self.hi.frame - t.started > 24 * 40:     # give up on anything stuck for 40 s
            if t.kind == "hotkey_building":
                self.hotkeyed[t.data["tag"]] = -1    # live without a hotkey for this one
            if t.kind == "build":
                return self._fail_build(t)
            self.task = None
            return
        if t.kind == "build":
            return self._task_build(obs, mine, t)
        if t.kind == "hotkey_building":
            return self._task_hotkey(obs, t)
        if t.kind == "attack":
            return self._task_attack(obs, mine, t)
        if t.kind == "mine":
            return self._task_mine(obs, mine, t)
        if t.kind == "resume":
            return self._task_resume(obs, mine, t)

    def _task_mine(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Select idle workers (drag box around them, or a click) and right-click a mineral field."""
        hi = self.hi
        tags = set(t.data["tags"])
        workers = [u for u in mine if u["tag"] in tags and screen_of(obs, u["x"], u["y"])]
        if not workers:
            self.task = None
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
                self.ignore_idle_until[tag] = hi.frame + 24 * 20
            self.task = None
            return
        # the closest mineral field to the Command Center of the nearest base Gary owns (a field
        # at a base without a Command Center means a long walk with every load)
        w = workers[0]
        bases = self._my_bases(mine) or [self.main]
        base = min(bases, key=lambda b: math.dist(b.center, (w["x"], w["y"])))
        fields = [m for m in base.minerals if screen_of(obs, *m)] or \
                 [m for b in bases for m in b.minerals if screen_of(obs, *m)]
        if fields:
            m = min(fields, key=lambda m: math.dist(m, base.center))
            hi.right_click(*screen_of(obs, *m))
        # whether or not it worked, don't fixate on these workers: look again in 20 s
        for tag in tags:
            self.ignore_idle_until[tag] = hi.frame + 24 * 20
        self.task = None

    def _my_bases(self, mine: list[dict]) -> list[Base]:
        """Bases with one of Gary's finished Command Centers."""
        halls = [u for u in mine if u["type"] == CC and u["completed"]]
        return [b for b in self.map.bases if any(math.dist(b.center, (h["x"], h["y"])) < 6 * 32 for h in halls)]

    def _town_hall_spot(self, base: Base, builder: int) -> tuple[int, int]:
        """The base's town-hall tile, or the nearest tile around it the game accepts (map knowledge
        can be a tile off)."""
        tx, ty = base.tile
        near = sorted(((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)), key=lambda d: abs(d[0]) + abs(d[1]))
        for dx, dy in near:
            if self._fits(CC, tx + dx, ty + dy, builder):
                return tx + dx, ty + dy
        return base.tile

    def _idle_sweep(self, obs: dict, done: list[dict]) -> bool:
        """Every 10 s, look at an idle worker the camera isn't showing, so the mining routine
        (which works on what's on screen) can send it back. True if Gary moved the camera."""
        hi = self.hi
        if hi.frame < self.next_sweep:
            return False
        busy = {getattr(self, "scout", None)} | set(getattr(self, "pulled", set()))
        away = [u for u in done if u["type"] == SCV and u["order"] == IDLE and u["tag"] not in busy
                and not screen_of(obs, u["x"], u["y"]) and self.ignore_idle_until.get(u["tag"], 0) <= hi.frame]
        self.next_sweep = hi.frame + 24 * 10
        if not away:
            return False
        hi.camera_minimap(away[0]["x"], away[0]["y"])
        return True

    def _free_worker(self, obs: dict, u: dict) -> bool:
        """A finished SCV on screen that isn't busy placing or constructing a building."""
        return u["type"] == SCV and u["completed"] and u["order"] not in BUSY_ORDERS             and bool(screen_of(obs, u["x"], u["y"]))

    def _pick_worker(self, obs: dict, mine: list[dict]) -> dict | None:
        cands = [u for u in mine if self._free_worker(obs, u)]
        return min(cands, key=lambda u: math.dist((u["x"], u["y"]), self.main.center)) if cands else None

    def _fail_build(self, t: Task) -> None:
        """Give up on this building. A build-order step is queued to be redone (otherwise a lost
        first depot leaves the bot supply-blocked: the supply logic only starts after step 2);
        supply depots and extra barracks are simply re-decided by the main loop."""
        if "step" in t.data:
            self._retry_step(t.data["step"])
        self.task = None

    def _retry_step(self, step: int) -> None:
        self.order_retry.append(step)
        self.retry_after = self.hi.frame + 24 * 5

    def _task_build(self, obs: dict, mine: list[dict], t: Task) -> None:
        hi = self.hi
        if t.stage == "start":                      # get a worker from the main
            if not self._look_at(obs, "main"):
                return
            w = t.data.get("worker")
            if w and obs["selection"] == [w]:
                t.stage = "place" if t.where == "main" else "travel"
                return
            # the click landed on a neighbor in a stack (the one drawn on top wins): a human just
            # uses whichever free SCV got selected instead of clicking again
            if w and len(obs["selection"]) == 1:
                got = next((u for u in mine if u["tag"] == obs["selection"][0]), None)
                if got and self._free_worker(obs, got):
                    t.data["worker"] = got["tag"]
                    t.stage = "place" if t.where == "main" else "travel"
                    return
            w = self._pick_worker(obs, mine)
            if not w:
                self._fail_build(t)
                return
            t.data["worker"] = w["tag"]
            p = screen_of(obs, w["x"], w["y"])
            hi.click(*p)
            return
        if t.stage == "travel":                      # walk the worker to the base
            hi.minimap_right_click(*self._base(t.where).center)
            t.stage = "wait_arrival"
            return
        if t.stage == "wait_arrival":
            w = next((u for u in mine if u["tag"] == t.data["worker"]), None)
            if not w:
                self._fail_build(t)
                return
            if math.dist((w["x"], w["y"]), self._base(t.where).center) > 6 * 32:
                return
            if not self._look_at(obs, t.where):
                return
            if t.where == "natural" and LOC_NATURAL not in hi.camera_locations:
                hi.camera_location_set(LOC_NATURAL)
                return
            t.stage = "place"
            return
        if t.stage == "place":
            budget = obs["me"]["minerals"] - sum(COST[ut] for ut, *_ in self.reserved)
            if budget < COST[t.unit_type]:
                return                                  # wait for money with the worker selected
            base = self._base(t.where)
            if "spot" not in t.data:
                if not self._look_at(obs, t.where):
                    return
                spot = self._town_hall_spot(base, t.data["worker"]) if t.unit_type == CC else \
                    self._spot_near(t.unit_type, t.data["near"], t.data["worker"]) if "near" in t.data else \
                    self._spot(t.unit_type, base, t.data["worker"])
                if not spot:
                    self._fail_build(t)
                    return
                t.data["spot"] = spot
            spot = t.data["spot"]
            w, h = SIZE[t.unit_type]
            p = screen_of(obs, spot[0] * 32 + 16, spot[1] * 32 + 16)
            if not p:                                   # the spot is off screen: look there first
                if t.data.get("looked"):
                    self._fail_build(t)
                    return
                t.data["looked"] = True
                hi.camera_minimap(spot[0] * 32 + w * 16, spot[1] * 32 + h * 16)
                return
            if hi.build(t.unit_type, *p).accepted:
                self.reserved.append((t.unit_type, *spot))
                self.reserved_at[(t.unit_type, *spot)] = hi.frame
                if "step" in t.data:
                    self.reserved_step[(t.unit_type, *spot)] = t.data["step"]
                self.task = None

    def _task_hotkey(self, obs: dict, t: Task) -> None:
        hi = self.hi
        tag = t.data["tag"]
        where = "natural" if math.dist((t.data["x"], t.data["y"]), self.natural.center) < 10 * 32 else "main"
        if obs["selection"] == [tag]:
            if t.stage == "start":
                pool = HK_CC if t.unit_type == CC else HK_RAX
                used = set(self.hotkeyed.values())
                hk = next((k for k in pool if k not in used), None)
                if hk is None:
                    self.hotkeyed[tag] = -1
                    self.task = None
                    return
                hi.hotkey_set(hk)
                self.hotkeyed[tag] = hk
                t.stage = "rally"
                return
            if t.stage == "rally":                      # CCs rally to minerals (auto-mine), barracks forward
                if t.unit_type == CC:
                    base = self.main if where == "main" else self.natural
                    m = base.minerals[len(base.minerals) // 2]
                    p = screen_of(obs, *m)
                    if p:
                        hi.right_click(*p)
                else:
                    hi.minimap_right_click(*self.rally)
                self.task = None
                return
        if not screen_of(obs, t.data["x"], t.data["y"]):
            hi.camera_minimap(t.data["x"], t.data["y"])
            return
        hi.click(*screen_of(obs, t.data["x"], t.data["y"]))

    def _task_attack(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Gather marines at the rally point with drag boxes, hotkey them, attack-move."""
        hi = self.hi
        waiting = [u for u in mine if u["type"] in self.army_types and u["completed"]
                   and u["tag"] not in self.army_sent]
        if not waiting:
            self.attacks_sent += 1
            self.task = None
            return
        # look where the waiting marines are (normally the rally point)
        cx = sorted(u["x"] for u in waiting)[len(waiting) // 2]
        cy = sorted(u["y"] for u in waiting)[len(waiting) // 2]
        if not screen_of(obs, cx, cy):
            hi.camera_minimap(cx, cy)
            return
        target = self.enemy_starts[self.attacks_sent % len(self.enemy_starts)]
        group = t.data.get("group", 0)
        if t.stage in ("start", "box"):
            here = [u for u in waiting if screen_of(obs, u["x"], u["y"])]
            if not here or group >= len(HK_ARMY) * 3:
                self.attacks_sent += 1
                self.task = None
                return
            xs = [screen_of(obs, u["x"], u["y"]) for u in here]
            x0 = max(0, min(p[0] for p in xs) - 12)
            y0 = max(0, min(p[1] for p in xs) - 12)
            x1 = min(hi.p.viewport[0] - 1, max(p[0] for p in xs) + 12)
            y1 = min(hi.p.viewport[1] - 1, max(p[1] for p in xs) + 12)
            hi.box(x0, y0, x1, y1)
            t.stage = "attack"
            return
        if t.stage == "attack":
            sent = [tag for tag in obs["selection"]]
            hi.minimap_command(C.ORDER_ATTACK_MOVE, *target)
            self.army_sent.update(sent)
            t.data["group"] = group + 1
            t.stage = "box"

    def _release_reservations(self, mine: list[dict]) -> None:
        """A reserved spot is free again once its building exists, or after 40 s if it never
        appeared (the worker was interrupted, or the spot got blocked)."""
        keep = []
        for r in self.reserved:
            ut, tx, ty = r
            cx, cy = tx * 32 + SIZE[ut][0] * 16, ty * 32 + SIZE[ut][1] * 16
            built = any(u["type"] == ut and math.dist((u["x"], u["y"]), (cx, cy)) < 48 for u in mine)
            if not built and self.hi.frame - self.reserved_at.get(r, 0) < 24 * 40:
                keep.append(r)
                continue
            # the placement never happened (rejected: a unit walked onto the spot): redo its step
            step = self.reserved_step.pop(r, None)
            if not built and step is not None:
                self._retry_step(step)
            self.reserved_at.pop(r, None)
        self.reserved = keep


def _terran_slot(game) -> int:
    """The slot that owns a Command Center: TerranGary plays Terran. (Slot order is the engine's,
    not the order of the races list, so bind by what's actually on the map.)"""
    obs = game.observe()
    slots = {p["slot"] for p in obs["players"]}
    for u in obs["units"]:
        if u["type"] == CC and u["owner"] in slots:
            return u["owner"]
    raise SystemExit("no Terran (Command Center) player in this game")


def _report(hi: HumanInterface, game) -> None:
    obs = game.observe()
    me = next(p for p in obs["players"] if p["slot"] == hi.slot)
    count = lambda t: sum(1 for u in obs["units"] if u["owner"] == hi.slot and u["type"] == t)
    print(f"{hi.frame // 1440}:00  supply {me['supply_used']:.0f}/{me['supply_max']:.0f}  "
          f"minerals {me['minerals']}  SCVs {count(SCV)}  marines {count(MARINE)}  "
          f"CC {count(CC)}  rax {count(RAX)}  depots {count(DEPOT)}  "
          f"APM {round(hi.stats['actions'] / max(1, hi.frame / 1440))}", flush=True)


def play(map_path: str, minutes: float, seed: int | None, save: str, opponent: str = "idle",
         command_delay: int = LIVE_COMMAND_DELAY, make_bot=None, name: str = "Gary v0.1 (T)") -> None:
    """Play on a headless OpenBW game (gary/env.py): fast, deterministic, saves a replay.
    command_delay: frames from sending a command to it running, as in a networked game.
    make_bot(hi, mapinfo) builds the bot (default: this one)."""
    races = ["T", "Z"]
    make_bot = make_bot or TerranGary
    with Game.new(map_path, races, [name, "Idle (Z)"], seed=seed,
                  command_delay=command_delay) as game:
        mapinfo = MapInfo.from_game(game)
        hi = HumanInterface(game, _terran_slot(game), PROFILES["pro"], seed=1)
        gary = make_bot(hi, mapinfo)
        end = int(minutes * 60 * 24)
        while hi.frame < end:
            gary.act()
            hi.step(2)
            if hi.frame % (24 * 60) < 2:
                _report(hi, game)
        game.save_replay(save)
        pov = save[:-4] + ".pov.jsonl" if save.endswith(".rep") else save + ".pov.jsonl"
        hi.save_pov(pov)
        print(f"saved {save} and {pov}")


def play_live(pipe: str, minutes: float, profile: str, pov: str | None, make_bot=None) -> None:
    """Play in a live StarCraft: Remastered client through the bridge (adapters/scr_bridge).

    Gary plays the client's local player (the bridge only accepts that slot), so it must be
    Terran. act() latency is calibrated from the adapter's reported latency_frames so the
    human interface lands actions when they really take effect. The client writes its own
    replay on match end.
    """
    from gary.scr_env import ScrGame
    game = ScrGame.connect(pipe)
    hi: HumanInterface | None = None
    try:
        # wait for a game to start and for Gary's Command Center to exist
        while True:
            st = game.status()
            slot = st.get("local_player", -1)
            if st.get("in_game") and slot >= 0:
                obs = game.observe()
                mains = [u["type"] for u in obs["units"] if u["owner"] == slot
                         and u["type"] in (CC, C.NEXUS, C.HATCHERY)]
                if CC in mains:
                    break
                if mains:   # a game is running, but Gary isn't Terran in it
                    raise SystemExit(
                        f"the local player is {'Protoss' if mains[0] == C.NEXUS else 'Zerg'}: "
                        f"Gary v0.1 plays Terran. Start a Terran game (melee assigns a random race: "
                        f"end this one and try again, or use auto_game.py --preset lan-create --race T)")
            print("waiting for a live game with Gary's Command Center (start one)...")
            game.step(24)
        st = game.status()
        mapinfo = MapInfo.from_game(game)
        hi = HumanInterface(game, slot, PROFILES[profile], seed=1)
        print(f"playing slot {slot} | act() latency calibrated to {hi.act_latency_frames} frames "
              f"(adapter reported latency_frames={st.get('latency_frames')})", flush=True)
        gary = (make_bot or TerranGary)(hi, mapinfo)
        end = hi.frame + int(minutes * 60 * 24)
        while hi.frame < end:
            gary.act()
            hi.step(2)                       # waits for the live game to advance
            if hi.frame % (24 * 60) < 2:
                _report(hi, game)
    except GameError as e:
        print(f"game over / bridge lost: {e}")
    finally:
        if pov and hi is not None:
            try:
                hi.save_pov(pov)
                print(f"saved {pov}")
            except Exception as e:
                print(f"could not save pov: {e}")
        game.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", help="a .scm/.scx map (or a pre-1.18 replay) for headless play")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", default="gary_v01.rep")
    ap.add_argument("--command-delay", type=int, default=LIVE_COMMAND_DELAY,
                    help=f"headless: frames before a command runs (default {LIVE_COMMAND_DELAY}, as measured live)")
    ap.add_argument("--live", action="store_true",
                    help="play in a live SC:R client via the bridge instead of headless OpenBW")
    ap.add_argument("--pipe", default=r"\\.\pipe\gary_scr", help="bridge named pipe (with --live)")
    ap.add_argument("--profile", default="pro", choices=sorted(PROFILES))
    ap.add_argument("--pov", help="where to save the point-of-view log (default: alongside --save)")
    args = ap.parse_args()
    if args.live:
        play_live(args.pipe, args.minutes, args.profile,
                  args.pov or "gary_v01_live.pov.jsonl")
    else:
        if not args.map:
            ap.error("--map is required unless --live")
        play(args.map, args.minutes, args.seed, args.save, command_delay=args.command_delay)


if __name__ == "__main__":
    main()
