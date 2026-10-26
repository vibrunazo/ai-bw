"""The macro executor (layer L2b, docs/CONTROL.md): how macro decisions get done with Gary's hands.

Building placement, workers (mining, gas, building), hotkeys and rallies for production buildings,
and selecting a building to train or research in. Scripted (provider `scripted/v1`, from Gary
v0.1-v0.2). It works on what the decision layers ask for (build tasks, commands) and on chores it
notices itself (idle workers, a refinery short of workers, a new building without a hotkey).

`Executor` here is the race-neutral part; a race plugs in its own over it (docs/REFACTOR.md):
gary/agent/terran_executor.py, gary/agent/zerg_executor.py.
"""

from __future__ import annotations

import math
from collections import Counter

from gary import commands as C
from gary import orders as O
from gary.agent.gary import LOC_NATURAL, Gary, Task
from gary.agent.macro import TOWN_HALLS, WAIT
from gary.decision_log import LOG
from gary.interface import screen_of
from gary.mapinfo import Base

SELECT_TRIES = 6                      # clicks on a building that select nothing before it's left alone...
UNSELECTABLE_FOR = 60 * 24            # ...for this long (live, one took 1,335 clicks and froze production)
HK_CC = [4, 5]                        # one hotkey per Command Center (buildings can't share one)
HK_RAX = [6, 7, 8, 9]                 # (another layout with control groups: gary/agent/grouping.py LAYOUTS)


def key_pool(g, hall: bool) -> list[int]:
    """The keys a new town hall (hall) or production building can take."""
    halls, production = g.key_pools or (HK_CC, HK_RAX)
    return halls if hall else production
STRAY = 12 * 32                              # home_scvs: a worker this far from every Command Center is astray
MINE_CLICKS = 4                               # home_scvs: clicks to get an idle worker alone (a box takes marines too)
FIGHT_CALM = 10 * 24                          # home_scvs: pulled or attack-moving workers are left alone this long
                                              # after zerg were last at Gary's bases (a fight's defenders)
SATURATED, UNDER = 1.8, 1.0                   # balance: workers a mineral patch at a base that gives some away,
                                              # and at one that takes them
TRANSFER_EVERY = 10 * 24                      # balance: bases' saturation looked at this often
DANGER_FOR = 60 * 24                          # balance: a base with zerg fighters seen near this recently isn't sent to
                                              # (20 s let mutalisks that hit, hid in the fog and came back draw the main's
                                              # workers to the natural under them: human game 4)
GATHERING = set(range(79, 92))                # orders: to, at, mining and returning minerals or gas
ZERG_FIGHTERS = {37, 38, 39, 43, 103, 46, 47, 44, 45, 62}
ATTACK_MOVE_ORDER, PATROL_ORDER = 14, 152
RALLY_DOWN, RALLY_OUT = 285 * 24, 450 * 24   # rally_ramp: inside the main's ramp until 4:45, at its
                                             # bottom until 7:30, then in front of the natural (pros' marines
                                             # at 3:00-4:00: the ramp's top 33%, the main 30%; at 5:00-6:00
                                             # its bottom 30-34%, the natural 17-19%; Gary's rally in front
                                             # of the natural from the start put a quarter of its marines in
                                             # the open)
BUSY_ORDERS = {25, 26, 30, 31, 33, 70}   # placing / constructing a building (a drone: walking to its spot, landing
                                         # there, morphing: 70 is DroneLand, and a drone re-sent then never started)
BAD_FOR = 60 * 24                     # a spot where a build failed (not for money) isn't tried again for this long
WIDE_SPOT = 24                  # option widespot: the main's search box this many tiles each way
HALL_CLEAR = 3                  # option widespot: tiles kept free around the natural's town hall site
LOST_ORDER = 6 * 24                   # a reserved spot with no SCV on its way after this: the order was lost
RESUME_AFTER = 24                     # an unfinished building nobody has built for a second gets an SCV
RESUME_RETRY = 24 * 4                 # ...and if that one doesn't arrive either, another after 4 s
IDLE = 3                              # order "PlayerGuard": standing around
MINING_STILL = 87                     # order "MiningMinerals": at its mineral field (lets a build order go)
SPLIT_UNTIL = 10 * 24                 # the game's start: idle workers sent one at a time, each to its own field
SPLIT_CLICKS = 3                      # clicks on one worker before the split gives way to a box (all to one field)
FIELD_NEAR = 40                       # px: a worker's order target this close to a field is on it (its near edge: a field is 64 x 32)
FIELD_BESIDE = 64                     # px: ...or the worker itself this close to it, mining (30-51 seen)
STANDING = {MINING_STILL, 86, IDLE}   # ...and waiting at the field, idle: where a box finds them
MOVE = 6
STANDING_DETOUR = 4 * 32              # a standing builder no farther than this beyond the nearest one
ARRIVING = 4 * 32                     # a parked builder this close to its spot is waited for until it stops
PICK_TRIES = 3                        # a builder not selected after this many tries: another one
PATH_CLEAR = 24                       # px: a building kept this far off a mineral path (spot_near avoid)
MINE_ADD = 5                          # builder_back: idle workers added to a mining task's click
NEIGHBOR = 2 * 32                     # px: an SCV this close to the parked one may be selected instead
RESELECT_TRIES = 6                    # a parked builder not selected again after this many: the build fails
RESET_HARVEST = 151                   # the order of a worker stepping out of a refinery (or off a mineral)
TRAVEL_FOR = 24 * 40                  # ...and one walking to another base, after this at most
RESEND_TRIES = 3                      # a build's SCV stopped on its way to another base: sent again this often
PARK_FOR = 24 * 20                    # a build's SCV walks to the spot ahead of the money: back to it after this at most
BUILDERS = 3                          # saturation: workers past the bases' minerals and gas, for building
BUSY_BUILDING = {O.ORDER_RESEARCH, O.ORDER_UPGRADE, O.ORDER_BUILD_ADDON}   # a building busy with these makes nothing
TASK_TIMEOUT = 24 * 40                # build, hotkey, mine and resume tasks give up after 40 s
COMMAND_TIMEOUT = 24 * 20             # commands and gas after 20 s
CREEP_BUILDINGS = {131, 132, 133, 143, 144, 146}   # Hatchery, Lair, Hive, Creep, Spore and Sunken Colony
CREEP_RX, CREEP_RY = 320, 200         # creep spreads up to this far from one (pixels, an ellipse; OpenBW's
                                      # spread_creep), a tile every few seconds



def field_click(obs: dict, field: tuple[int, int], hall: tuple[int, int]) -> tuple[int, int]:
    """Where to right-click a mineral field to gather from it: its far side from the town hall, where no miner
    stands (a click on its middle hit an SCV standing there and became a follow order: a quarter of the time
    workers were off their work in a test game)."""
    dx, dy = field[0] - hall[0], field[1] - hall[1]
    d = math.hypot(dx, dy) or 1
    return screen_of(obs, field[0] + dx / d * 12, field[1] + dy / d * 6) or screen_of(obs, *field)


def _to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """The distance from p to the segment a-b."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / ((dx * dx + dy * dy) or 1)))
    return math.dist(p, (ax + t * dx, ay + t * dy))

class Executor:
    build_order = C.ORDER_PLACE_BUILDING          # the order a worker places a building with
    builder_retries = 0                           # a build whose worker is lost (died, pulled): another this often
    box_extras: frozenset = frozenset()           # units a box around workers may take that a mining order leaves be
    box_orders: frozenset = frozenset()           # ...and orders of other free workers in it, sent mining with them
    army_select = "box"                           # how the motor selects an army's units: drag boxes, or "portrait"
                                                  # (a box, then a ctrl+click on a portrait: no drones or overlords along)
    """The race-neutral hands (docs/REFACTOR.md Phase 1): reservations, workers to mining and gas,
    transfers, picking and selecting a worker, placement, hotkeys and rallies, commands to a building,
    and a worker's build (walk, place). A race's executor adds what only it does: its task kinds
    (`basic`), its placement rule (`site_ok`), its chores."""

    def __init__(self, g: Gary):
        self.g = g
        self.next_sweep = 0                           # next look for idle workers off screen
        self._terrain_cache = None                    # see terrain()
        self._ramp = None                             # rerally: (spot inside the main's ramp, its bottom), or False
        self.danger: dict[tuple, int] = {}            # balance: base center -> frame zerg fighters were last seen near it
        self.next_transfer = 0
        self._bases_by_walk: list | None = None       # free_base: the bases by a worker's walk from the natural
        self.why: str | None = None                   # why carry_out's last None or WAIT (no, wait)
        self.wide_spot = False                        # option widespot: spot() widens the main, then the natural
        # the map's own creep (map knowledge, read now: at the start of a game, before any zerg's)
        creep = getattr(g.hi.game, "creep", None)
        self._map_creep = creep() if creep else None
        self.basic = {"build": self._task_build, "hotkey_building": self._task_hotkey, "mine": self._task_mine,
                      "rerally": self._task_rerally, "transfer": self._task_transfer}
        for kind in self.basic:
            g.task_runners[kind] = self._run_basic
        self.commands = {"command": self._task_command, "gas": self._task_gas}   # (the runs with COMMAND_TIMEOUT)
        for kind in self.commands:
            g.task_runners[kind] = self._run_command

    # --- chores, in the arbiter's order ------------------------------------------------------

    def resume_construction(self, mine: list[dict]) -> bool:
        """An unfinished building nobody builds gets a worker to finish it: a race whose buildings need
        one (Terran). True if Gary acted."""
        return False

    def release_reservations(self, mine: list[dict]) -> None:
        """A reserved spot is free again once its building exists, or after 40 s if it never
        appeared (the worker was interrupted, or the spot got blocked), or after 6 s if no SCV is
        on its way to place it (the order was lost: an SCV busy constructing keeps at it), so the
        macro orders it again rather than wait 40 s for it."""
        g = self.g
        keep = []
        me = g.hi.observe()["me"]
        for r in g.reserved:
            ut, tx, ty = r
            cx, cy = tx * 32 + g.race.sizes[ut][0] * 16, ty * 32 + g.race.sizes[ut][1] * 16
            built = any(u["type"] == ut and math.dist((u["x"], u["y"]), (cx, cy)) < 48 for u in mine)
            age = g.hi.frame - g.reserved_at.get(r, 0)
            heading = any(u["type"] == g.race.worker and u["order"] in BUSY_ORDERS and "tx" in u
                          and math.dist((u["tx"], u["ty"]), (cx, cy)) < 96 for u in mine)
            # its builder stopped placing it and nothing was started: the build failed (the money
            # was spent meanwhile, or the game wouldn't place it there: its pathfinder left the
            # SCV short of the spot). Again now, and somewhere else if money wasn't short
            builder = next((u for u in mine if u["tag"] == g.builder_of.get(r)), None)
            failed = not built and builder is not None and age > 24 and builder["order"] not in BUSY_ORDERS \
                and math.dist((builder["x"], builder["y"]), (cx, cy)) < 4 * 32
            if not built and not failed and age < 24 * 40 and (heading or age < LOST_ORDER):
                keep.append(r)
                continue
            if failed:
                g.build_failed[ut] = g.hi.frame
                if me is not None and me["minerals"] < g.race.costs[ut][0]:
                    g.say(f"its {g.race.names.get(ut, ut)} failed for money: again")
                    LOG.order(g.hi.side, g.hi.frame, f"{g.race.names.get(ut, ut)} at {(tx, ty)}", "retried", "failed for money",
                              by="executor")
                elif g.hi.frame - g.failed_at.get(r, -10 ** 9) < BAD_FOR:
                    g.bad_spots[r] = g.hi.frame              # (twice: somewhere else)
                    g.say(f"couldn't place its {g.race.names.get(ut, ut)} there twice: elsewhere")
                    LOG.order(g.hi.side, g.hi.frame, f"{g.race.names.get(ut, ut)} at {(tx, ty)}", "retried",
                              "the game wouldn't place it there twice: elsewhere", by="executor")
                else:
                    # the game's pathfinder can leave an SCV stuck short of the spot (the order then
                    # fails): the same SCV again from where it stands, a new order clears that
                    g.failed_at[r] = g.hi.frame
                    g.retry_with[ut] = builder["tag"]
                    g.say(f"its {g.race.names.get(ut, ut)} didn't start: again, with the same SCV")
                    LOG.order(g.hi.side, g.hi.frame, f"{g.race.names.get(ut, ut)} at {(tx, ty)}", "retried",
                              "its SCV stopped and nothing started: again, the same SCV", by="executor")
            elif not built and age < 24 * 40:
                g.say(f"its {g.race.names.get(ut, ut)} order was lost: again")
                LOG.order(g.hi.side, g.hi.frame, f"{g.race.names.get(ut, ut)} at {(tx, ty)}", "retried",
                          "no SCV on its way to it: the order was lost", by="executor")
            g.reserved_at.pop(r, None)
            g.builder_of.pop(r, None)
        g.reserved = keep

    def _idle(self, u: dict, mine: list[dict]) -> bool:
        """A worker the mining routine should send back: standing idle, or (home_scvs, an option) away
        from every Command Center (STRAY) idle or still attack-moving, nobody holding it (a pull that
        ended left them where they stood: 9-21 at once in the middle of the map, against Zerg v2/v3)."""
        if u["order"] == IDLE:
            return True
        g = self.g
        if not g.home_scvs or u["order"] not in (ATTACK_MOVE_ORDER, PATROL_ORDER) or g.leases.holder(u["tag"]) \
                or g.hi.frame - g.calm_since < FIGHT_CALM:
            return False                              # (a fight at home lately: they may be its defenders)
        ccs = [(c["x"], c["y"]) for c in mine if c["type"] in g.race.town_halls]
        return bool(ccs) and min(math.dist((u["x"], u["y"]), c) for c in ccs) > STRAY

    def idle_mining(self, obs: dict, done: list[dict]) -> bool:
        """Idle workers on screen go back to mining (game start, after building something)."""
        g = self.g
        idle = [u for u in done if u["type"] == g.race.worker and self._idle(u, done) and screen_of(obs, u["x"], u["y"])
                and g.ignore_idle_until.get(u["tag"], 0) <= g.hi.frame and u["tag"] != g.scout
                and u["tag"] not in g.block_units]
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
        busy = {g.scout} | (g.pulled if not g.home_scvs or g.hi.frame - g.calm_since < FIGHT_CALM else set()) | g.block_units
        away = [u for u in done if u["type"] == g.race.worker and self._idle(u, done) and u["tag"] not in busy
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

    def rerally(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        """rally_ramp (an option): where new fighters gather moves as the pros' does (the top of the
        main's ramp, then its bottom, then in front of the natural: RALLY_PLAN), and the production
        buildings rallied elsewhere get the new point (their hotkey, a minimap right-click)."""
        g = self.g
        if g.rally_plan != "ramp" or g.posture.rush or g.posture.hold:
            return False                              # (the early-pool posture has its own rally)
        spot = self._rally_spot()
        if spot is not None and spot != g.rally:
            g.rally = spot
        for b in done:
            hk = g.hotkeyed.get(b["tag"])
            if b["type"] in g.production_types and b["type"] not in g.race.town_halls and hk is not None and hk >= 0 \
                    and g.rallied.get(b["tag"]) != g.rally:
                g.task = Task("rerally", b["type"], data={"tag": b["tag"], "hk": hk}, started=g.hi.frame)
                return True
        return False

    def _rally_spot(self):
        """The rally of the moment (RALLY_PLAN), or None (no reliable ramp: as before)."""
        g = self.g
        if self._ramp is None:
            self._ramp = False
            try:
                from gary.agent.block import _analysis
                a = _analysis(g)
                if a["reliable"]:
                    self._ramp = (tuple(a["ramp_inside"]), tuple(a["ramp_bottom"]))
            except Exception:                         # (no terrain in this game)
                pass
        if not self._ramp:
            return None
        f = g.hi.frame
        return self._ramp[0] if f < RALLY_DOWN else self._ramp[1] if f < RALLY_OUT else g.toward_center(g.natural.center, 6 * 32)

    def _task_rerally(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi = self.g, self.g.hi
        tag = t.data["tag"]
        if not any(u["tag"] == tag for u in mine):
            g.task = None
            return
        if obs["selection"] != [tag]:
            hi.hotkey_recall(t.data["hk"])
            if t.data.setdefault("tries", 0) >= 3:
                g.rallied[tag] = g.rally              # (its hotkey brings up something else: leave it)
                g.task = None
            t.data["tries"] += 1
            return
        hi.minimap_right_click(*g.rally)
        g.rallied[tag] = g.rally
        g.task = None

    def refill_gas(self, obs: dict, done: list[dict]) -> bool:
        """A finished refinery with fewer than three workers gets more."""
        g = self.g
        for r in done:
            if r["type"] == g.race.gas_building and self.on_gas(obs, r["tag"]) < 3 and \
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
            hk = next((k for k in key_pool(g, b["type"] in g.race.town_halls) if k not in used), -1)
            g.hotkeyed[b["tag"]] = hk
            if hk >= 0:
                g.hi.hotkeys[hk] = [b["tag"]]

    # --- workers and bases --------------------------------------------------------------------

    def my_bases(self, mine: list[dict]) -> list[Base]:
        """Bases with one of Gary's finished Command Centers."""
        halls = [u for u in mine if u["type"] in self.g.race.town_halls and u["completed"]]
        return [b for b in self.g.map.bases if any(math.dist(b.center, (h["x"], h["y"])) < 6 * 32 for h in halls)]

    def mine_base(self, mine: list[dict], at, obs: dict | None = None):
        """The base Gary's workers at `at` should mine at: the nearest it owns, or (balance, an
        option) the least saturated (workers gathering there per mineral patch) with no zerg
        fighters seen near it for DANGER_FOR, the nearest on a tie: pros send workers where the
        patches are free, not to the nearest base."""
        g = self.g
        bases = self.my_bases(mine) or [g.main]
        if not g.balance:
            return min(bases, key=lambda b: math.dist(b.center, at))
        if obs is not None:
            self.note_danger(obs)
        safe = [b for b in bases if g.hi.frame - self.danger.get(b.center, -10 ** 9) > DANGER_FOR] or bases
        return min(safe, key=lambda b: (round(self.saturation_at(b, mine) * 2) / 2, math.dist(b.center, at)))

    def saturation_at(self, base, mine: list[dict]) -> float:
        """Workers gathering at this base per mineral patch."""
        n = sum(1 for u in mine if u["type"] == self.g.race.worker and u["order"] in GATHERING and math.dist((u["x"], u["y"]), base.center) < 10 * 32)
        return n / max(1, len(base.minerals))

    def note_danger(self, obs: dict) -> None:
        """balance: Gary's bases with zerg fighters near them now (what Gary sees)."""
        g = self.g
        fighters = [u for u in obs["units"] if u["owner"] not in (g.slot, 11) and u["type"] in ZERG_FIGHTERS]
        for b in g.map.bases:
            if any(math.dist(b.center, (u["x"], u["y"])) < 14 * 32 for u in fighters):
                self.danger[b.center] = g.hi.frame

    def transfer(self, obs: dict, mine: list[dict]) -> bool:
        """balance: workers from a base over SATURATED a patch to a safe one under UNDER (a new
        Command Center, mostly), a group at once: pros move a bunch the moment a base is up."""
        g, hi = self.g, self.g.hi
        if not g.balance or hi.frame < self.next_transfer:
            return False
        self.next_transfer = hi.frame + TRANSFER_EVERY
        self.note_danger(obs)
        bases = self.my_bases(mine)
        if len(bases) < 2:
            return False
        sat = {b.center: self.saturation_at(b, mine) for b in bases}
        src = max(bases, key=lambda b: sat[b.center])
        safe = [b for b in bases if b is not src and hi.frame - self.danger.get(b.center, -10 ** 9) > DANGER_FOR]
        if sat[src.center] <= SATURATED or not safe:
            return False
        dst = min(safe, key=lambda b: sat[b.center])
        if sat[dst.center] >= UNDER:
            return False
        k = min(int((sat[src.center] - 1.5) * len(src.minerals)), int((1.5 - sat[dst.center]) * len(dst.minerals)), 8)
        if k < 3:
            return False
        line = (sum(x for x, _ in src.minerals) / len(src.minerals), sum(y for _, y in src.minerals) / len(src.minerals))
        miners = sorted((u for u in mine if u["type"] == g.race.worker and u["order"] in GATHERING and u["tag"] not in g.pulled | g.block_units
                         and math.dist((u["x"], u["y"]), src.center) < 10 * 32),
                        key=lambda u: math.dist((u["x"], u["y"]), line))[:k]
        g.task = Task("transfer", data={"tags": [u["tag"] for u in miners], "to": dst.center}, started=hi.frame)
        g.say(f"transfers {len(miners)} workers to the base at {dst.center[0] // 32},{dst.center[1] // 32}")
        return True

    def _task_transfer(self, obs: dict, mine: list[dict], t: Task) -> None:
        """A box around the chosen workers (at their mineral line), then the camera to the other base and a
        right-click on one of its mineral fields: a gather order (a minimap right-click is a move)."""
        g, hi = self.g, self.g.hi
        tags = set(t.data["tags"])
        workers = [u for u in mine if u["tag"] in tags]
        if not workers or t.data.get("tries", 0) >= 3:
            g.task = None
            return
        if t.stage == "start":
            x = sorted(u["x"] for u in workers)[len(workers) // 2]
            y = sorted(u["y"] for u in workers)[len(workers) // 2]
            on = [u for u in workers if screen_of(obs, u["x"], u["y"])]
            if len(on) < len(workers) // 2 + 1:
                hi.camera_minimap(x, y)
                t.data["tries"] = t.data.get("tries", 0) + 1
                return
            pts = [hi.aim(obs, u) for u in on]
            hi.box(max(0, min(p[0] for p in pts) - 4), max(0, min(p[1] for p in pts) - 4),
                   min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + 4), min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + 4))
            t.stage = "send"
            return
        sel = [u for u in mine if u["tag"] in obs["selection"]]
        base = next(b for b in g.map.bases if b.center == tuple(t.data["to"]))
        if t.stage == "gather":                       # (the camera at the base: its field, a gather order)
            fields = self._clear_fields(mine, [m for m in base.minerals if screen_of(obs, *m)])
            if fields and sel:
                hi.right_click(*field_click(obs, min(fields, key=lambda m: math.dist(m, base.center)), base.center))
            elif sel:
                hi.minimap_right_click(*min(base.minerals, key=lambda m: math.dist(m, base.center)))
            g.task = None
            return
        if sel and all(u["type"] == g.race.worker for u in sel) and len(sel) >= 2:
            # the camera to the other base, the field clicked next turn: a minimap right-click on its minerals
            # was a move, and the workers stood idle there (human game 4: "SCVs mining fine on one base go stand
            # idle in the main for no reason")
            hi.camera_minimap(*base.center)
            t.stage = "gather"
            return
        t.stage = "start"
        t.data["tries"] = t.data.get("tries", 0) + 1

    def free_worker(self, obs: dict, u: dict) -> bool:
        """A finished SCV on screen that isn't busy placing or constructing a building (or walking
        to build one)."""
        g = self.g
        # (nor one just given a build: the order takes a few frames to show in its state, and the
        # next build's task took it as free and sent it walking elsewhere: the first build lost)
        builders = {g.builder_of.get(r) for r in g.reserved}
        return u["type"] == g.race.worker and u["completed"] and u["order"] not in BUSY_ORDERS \
            and bool(screen_of(obs, u["x"], u["y"])) and u["tag"] != g.scout and u["tag"] not in g.pulled | g.block_units \
            and (g.parked is None or g.parked.data["worker"] != u["tag"]) and u["tag"] not in builders

    def resume_parked(self, obs: dict, mine: list[dict]) -> bool:
        """The parked build (its SCV walking there) goes on once its money is there, or after
        PARK_FOR anyway (it then waits for the money at the spot). True if it took the hands."""
        g = self.g
        t = g.parked
        if t is None or g.task is not None:
            return False
        if not any(u["tag"] == t.data["worker"] for u in mine):
            g.parked = None
            if self.lost_builder(t, "its worker is gone"):
                g.task = t
                return True
            return False
        if t.data.get("resume") == "wait_arrival":           # walking to another base: until it's there
            w = next(u for u in mine if u["tag"] == t.data["worker"])
            center = g.base(t.where).center
            walking = w["order"] == MOVE and math.dist((w.get("tx", 0), w.get("ty", 0)), center) < 6 * 32
            far = math.dist((w["x"], w["y"]), center) > 6 * 32
            if far and g.hi.frame - t.data["walking"] < TRAVEL_FOR and (walking or g.hi.frame - t.data["walking"] < 2 * 24):
                return False                                  # (stopped short: back to it now, to send it again)
            if far and g.hi.frame - t.data["walking"] >= TRAVEL_FOR:
                # not there after TRAVEL_FOR: the build fails and the macro layer decides again (the hands
                # taken back for it waited 40 s more on an SCV pacing behind marines posted on the ramp,
                # and the Refinery asked for meanwhile was given up twice)
                g.parked = None
                self.fail_build(t, "its SCV didn't get to the base")
                return False
        else:
            budget = obs["me"]["minerals"] - sum(g.race.minerals[ut] for ut, *_ in g.reserved)
            if budget < g.race.minerals[t.unit_type] and g.hi.frame - t.data["walking"] < PARK_FOR:
                return False
            # just about there: until it stops (a box finds a standing SCV where Gary saw it; one
            # slowing down at its spot was missed 7 times a game)
            w = next(u for u in mine if u["tag"] == t.data["worker"])
            if w["order"] == MOVE and (w.get("tx") or w.get("ty")) and \
                    math.dist((w["x"], w["y"]), (w["tx"], w["ty"])) < ARRIVING and g.hi.frame - t.data["walking"] < PARK_FOR:
                return False
        g.task, g.parked = t, None
        t.started = g.hi.frame          # (the walk had its own limit: a third base's walk outlasted TASK_TIMEOUT)
        g.run_task(obs, mine)
        return True

    def pick_worker(self, obs: dict, mine: list[dict], near=None, skip=()) -> dict | None:
        """A free SCV on screen: the one nearest where it will build, if known (a long walk across
        the mineral line can leave it stuck short of the spot), else nearest the main; one standing
        if any (at its mineral field, or idle): a box finds it where Gary saw it, while one walking
        to or from the minerals was missed by a box in 1 try in 4 (it speeds up, slows down and
        turns faster than Gary can lead it)."""
        cands = [u for u in mine if self.free_worker(obs, u) and u["tag"] not in skip]
        if not cands:
            return None
        to = near or self.g.main.center
        far = lambda u: math.dist((u["x"], u["y"]), to)
        nearest = min(cands, key=far)
        # (but not a longer walk: farther than this and the buildings came late, army built -5 by 12:00)
        standing = [u for u in cands if u["order"] in STANDING and far(u) <= far(nearest) + STANDING_DETOUR]
        return min(standing, key=far) if standing else nearest

    def select_worker(self, obs: dict, w: dict) -> None:
        """Select this SCV: a tiny drag box where it will be (interface.aim), which takes units before
        buildings; a click there if the last box took several SCVs (two share a mineral field, and a
        build order to several is refused), which takes one: it or the neighbor on top. (Clicked
        where it was seen, one at its mineral field or by a Command Center gave nothing or the
        Command Center, a walking one had moved on, and Gary clicked again every 2 frames: 150-235
        clicks a game.)"""
        hi = self.g.hi
        x, y = hi.aim(obs, w)
        sel = [u for u in obs["units"] if u["tag"] in set(obs["selection"])]
        if len(sel) > 1 and all(u["type"] == self.g.race.worker for u in sel):
            hi.click(x, y)
            return
        hi.box(max(0, x - 4), max(0, y - 4), min(hi.p.viewport[0] - 1, x + 4), min(hi.p.viewport[1] - 1, y + 4))

    def on_gas(self, obs: dict, refinery: int) -> int:
        """Workers on this refinery: the ones Gary sent that are still on gas (on a gas order, just
        out of the refinery, or out of sight: a worker inside a refinery isn't visible), and any
        worker seen heading into it (the SCV that built it starts on gas by itself)."""
        g = self.g
        seen = {u["tag"]: u for u in obs["units"] if u["owner"] == g.slot}
        alive = {w for w in g.gas_workers.get(refinery, set())
                 if w not in seen or seen[w]["order"] in O.GAS_ORDERS or seen[w]["order"] == RESET_HARVEST}
        alive |= {t for t, u in seen.items() if u["type"] == g.race.worker and u["order"] in O.GAS_ORDERS
                  and u.get("order_target") == refinery}
        g.gas_workers[refinery] = alive
        return len(alive)

    # --- what Gary has to carry out decisions with ------------------------------------------------

    def in_production(self, mine: list[dict], first: bool = False) -> Counter:
        """Units being made right now (the game counts them, observations don't list them): by its
        buildings' queues and what Gary last started in each. first: only each building's unit in training, as
        the game (and so the replays) count it, not the ones queued behind it (option unitcount)."""
        out: Counter = Counter()
        for b in mine:
            q = b.get("queue", 0)
            if b["completed"] and q > 0 and b["tag"] in self.g.producing:
                for uid in self.g.producing[b["tag"]][-q:][:1 if first else q]:     # (the queue holds the last q started)
                    out[uid] += 1
        return out

    def no(self, why: str) -> None:
        """carry_out can't: why, in words (the macro's records say it of each option passed over)."""
        self.why = why
        return None

    def wait(self, why: str) -> dict:
        """carry_out will be able to soon: what it waits for, in words."""
        self.why = why
        return WAIT

    def count(self, mine: list[dict], unit_type: int) -> int:
        """Own units of a type, counting ones in production and buildings about to be placed."""
        g = self.g
        n = sum(1 for u in mine if u["type"] == unit_type) + self.in_production(mine)[unit_type]
        n += sum(1 for ut, *_ in g.reserved if ut == unit_type)
        if g.building(unit_type):
            n += 1
        return n

    def cost_of(self, act: int, uid: int) -> tuple[int, int]:
        """A decision's money (minerals, gas)."""
        from gary.agent.macro import EXPAND, RESEARCH, UPGRADE
        race = self.g.race
        if act == EXPAND:
            return race.costs[race.town_hall]
        if act == RESEARCH:
            return race.research[uid][1]
        if act == UPGRADE:
            return race.upgrade[uid][1]
        return race.costs.get(uid, (0, 0))

    def burst(self, act: int, uid: int) -> bool:
        """A decision a player makes several of at once (Zerg: a larva's): the macro's next look comes soon."""
        return False

    def supply_due(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> tuple[int, str, bool] | None:
        """Supply ahead of production (option depots): (the act that makes the race's supply provider, why, whether
        it goes past the pros' caps), when one is due; None."""
        return None

    def saturation(self, mine: list[dict]) -> int:
        """SCVs enough for Gary's bases (with a Command Center, finished or not): 2 per mineral field,
        3 per geyser, and builders. Pros have 39 at 12:00 on two bases; Gary, making SCVs to 60, had
        42-51 there, the minerals of the army it lacked."""
        halls = [u for u in mine if u["type"] in self.g.race.town_halls]
        bases = [b for b in self.g.map.bases if any(math.dist(b.center, (h["x"], h["y"])) < 6 * 32 for h in halls)]
        return sum(2 * len(b.minerals) + 3 * len(b.geysers) for b in bases) + BUILDERS

    def free_building(self, obs: dict, mine: list[dict], done: list[dict], kind: int,
                      addon: int | None = None, depth: int = 1) -> dict | None:
        """A building of this kind that can take an order now: an empty queue, or (depth) fewer than depth in it,
        the shortest first."""
        g = self.g
        blind = round(g.hi.p.reaction_ms / 42) + g.hi.act_latency_frames + 4
        for b in sorted(done, key=lambda u: u.get("queue", 0)):
            if b["type"] != kind or b.get("queue", 0) >= depth or b["order"] in BUSY_BUILDING:
                continue
            if obs["now"] - g.ordered_at.get(b["tag"], -10**9) <= blind:
                continue
            if g.unselectable.get(b["tag"], -1) > g.hi.frame:
                continue                        # (its clicks select nothing: executor.py select)
            if addon is not None:
                a = self.addon_of(b, mine)
                if a is None or a["type"] != addon or not a["completed"]:
                    continue
            return b
        return None

    def _command_or_wait(self, b: dict | None, done: list[dict], kind: int, act: int, uid: int,
                         cost: tuple[int, int]) -> dict | None:
        if b is not None:
            return {"kind": "command", "act": act, "id": uid, "building": b, "cost": cost}
        name = self.g.race.names.get(kind, str(kind))
        return self.wait(f"its {name} is busy") if any(u["type"] == kind for u in done) else self.no(f"no {name}")

    def free_geyser(self, obs: dict, mine: list[dict]):
        g = self.g
        refineries = [u for u in mine if u["type"] == g.race.gas_building] + \
                     [{"x": tx * 32 + 64, "y": ty * 32 + 32} for ut, tx, ty in g.reserved if ut == g.race.gas_building]
        for base in self.my_bases(mine):
            for gx, gy in base.geysers:
                if any(math.dist((r["x"], r["y"]), (gx, gy)) < 32 for r in refineries):
                    continue
                where = "main" if base is g.main else "natural" if base is g.natural else base
                return where, ((gx - 64) // 32, (gy - 32) // 32)
        return None

    def free_base(self, obs: dict):
        """The nearest base nobody has taken (the natural first), as far as Gary has seen: nearest by a
        worker's walk from the natural (terrain.py), not as the crow flies."""
        g = self.g
        halls = [u for u in obs["units"] if u["type"] in TOWN_HALLS]
        reserved = [(tx * 32 + 64, ty * 32 + 48) for ut, tx, ty in g.reserved if ut == g.race.town_hall]
        enemy = [tuple(s) for s in g.enemy_starts]
        free = [b for b in g.map.bases if b is not g.main
                and not any(math.dist(b.center, (h["x"], h["y"])) < 10 * 32 for h in halls)
                and not any(math.dist(b.center, r) < 10 * 32 for r in reserved)
                and not any(math.dist(b.center, s) < 10 * 32 for s in enemy)]
        if g.natural in free:
            return g.natural
        if not free:
            return None
        if self._bases_by_walk is None:
            from gary.agent.block import _analysis
            try:
                self._bases_by_walk = [tuple(c) for c in _analysis(g).get("bases_by_walk", [])]
            except Exception:                    # (no terrain in this game)
                self._bases_by_walk = []
        walk = self._bases_by_walk
        return min(free, key=lambda b: (walk.index(tuple(b.center)) if tuple(b.center) in walk else len(walk),
                                        math.dist(b.center, g.main.center)))

    # --- placement ------------------------------------------------------------------------------

    def spot(self, unit_type: int, base: Base, builder: int) -> tuple[int, int] | None:
        """A spot at the base, or at another of Gary's bases when that one is full. With a sim city
        (gary/agent/simcity.py), a spot in the wall first, where one narrows the frontier."""
        if self.g.simcity is not None and (wall := self.g.simcity.spot(unit_type, builder)):
            return wall
        spot = self._base_spot(unit_type, base, builder)
        if spot is None:
            for other in self.my_bases([u for u in self.g.hi.observe()["units"] if u["owner"] == self.g.slot]):
                if other is not base and (spot := self._base_spot(unit_type, other, builder)):
                    break
        if spot is None and self.wide_spot:
            # (option widespot: the main's box widened, its own ground level first, then the natural's area with no
            # town hall yet, its town hall's ground kept clear; a full main on one base dropped 16.8 builds a game in
            # gt1dev, eval/macro_pipe.py, while the wider box had 26-74 spots and the natural 60-147 at the same
            # moments. Without the clearance its buildings stood on the natural's own site: expansions dropped 2.5
            # times as often, 1.5 bases at 12:00 against 1.7, round WS120a)
            g = self.g
            spot = self._base_spot(unit_type, g.main, builder, radius=WIDE_SPOT, level_of=g.main.center)
            if spot is None and g.natural is not None:
                spot = self._base_spot(unit_type, g.natural, builder, clear=g.natural.tile)
        return spot

    def _base_spot(self, unit_type: int, base: Base, builder: int, radius: int = 16,
                   level_of: tuple[int, int] | None = None,
                   clear: tuple[int, int] | None = None) -> tuple[int, int] | None:
        """A free spot near the base, on the side away from the minerals, within radius tiles of that side's
        anchor; with level_of, on that point's ground level if any spot there fits; with clear (a town hall's
        tile), off that town hall's ground and its add-on's, HALL_CLEAR tiles around them."""
        mx = sum(x for x, _ in base.minerals) / len(base.minerals)
        my = sum(y for _, y in base.minerals) / len(base.minerals)
        ax = base.center[0] + (base.center[0] - mx) * 0.9
        ay = base.center[1] + (base.center[1] - my) * 0.9
        w, h = self.g.race.sizes[unit_type]
        cands = []
        for ty in range(int(ay // 32) - radius, int(ay // 32) + radius):
            for tx in range(int(ax // 32) - radius, int(ax // 32) + radius):
                cands.append((math.dist((tx * 32 + w * 16, ty * 32 + h * 16), (ax, ay)), tx, ty))
        terrain = self.terrain() if level_of is not None else None
        if terrain is not None:                  # (its level's tiles first, the rest after)
            level = int(terrain[1][int(level_of[1]) // 8, int(level_of[0]) // 8])
            off = lambda tx, ty: bool((terrain[1][max(0, ty) * 4:(ty + h) * 4, max(0, tx) * 4:(tx + w) * 4] != level).any())
            cands = [(off(tx, ty), d, tx, ty) for d, tx, ty in cands]
            cands = [(d, tx, ty) for _, d, tx, ty in sorted(cands)]
        else:
            cands = sorted(cands)
        if clear is not None:
            hw, hh = self.g.race.sizes[self.g.race.town_hall]
            bx, by = clear
            cands = [(d, tx, ty) for d, tx, ty in cands
                     if not (tx < bx + hw + 2 + HALL_CLEAR and tx + w > bx - HALL_CLEAR
                             and ty < by + hh + HALL_CLEAR and ty + h > by - HALL_CLEAR)]
        for _, tx, ty in cands:
            if any(abs(tx - rx) < self.g.race.sizes[rt][0] + 1 and abs(ty - ry) < self.g.race.sizes[rt][1] + 1 for rt, rx, ry in self.g.reserved) \
                    or self._bad(unit_type, tx, ty):
                continue
            # keep a walkable gap around buildings: the bigger footprint must fit too
            if self.fits(unit_type, tx, ty, builder) and self.fits(unit_type, tx - 1, ty - 1, builder) and \
                    self.fits(unit_type, tx + 1, ty + 1, builder):
                return tx, ty
        return None

    def spot_near(self, unit_type: int, near: tuple[int, int], builder: int,
                  level_at: tuple[int, int] | None = None, avoid=None) -> tuple[int, int] | None:
        """The free spot closest to a point (e.g. a bunker at the top of the main ramp), on the ground
        level of level_at (e.g. the main: the closest free tile to the top of a ramp can be at its
        bottom). On a flat map every tile is on that level; if no spot on it fits, any level does.
        avoid: segments the building stands clear of (a Command Center to each of its mineral fields:
        the SCVs' paths)."""
        w, h = self.g.race.sizes[unit_type]
        cx, cy = near[0] // 32, near[1] // 32
        terrain = self.terrain()
        level = int(terrain[1][level_at[1] // 8, level_at[0] // 8]) if terrain and level_at else None
        cands = sorted((math.dist((tx * 32 + w * 16, ty * 32 + h * 16), near), tx, ty)
                       for ty in range(cy - 8, cy + 8) for tx in range(cx - 8, cx + 8))
        # (a tile in from the map's edge: a turret at tile 0 passed the game's check live and its builder
        # never got there, human game 3)
        mw, mh = (self.g.map.size[0] // 32, self.g.map.size[1] // 32) if self.g.map is not None else (256, 256)
        cands = [c for c in cands if 1 <= c[1] and 1 <= c[2] and c[1] + w <= mw - 1 and c[2] + h <= mh - 1]
        for same_level in ((True, False) if level is not None else (False,)):
            for _, tx, ty in cands:
                if any(abs(tx - rx) < self.g.race.sizes[rt][0] + 1 and abs(ty - ry) < self.g.race.sizes[rt][1] + 1 for rt, rx, ry in self.g.reserved) \
                        or self._bad(unit_type, tx, ty):
                    continue
                if same_level and (terrain[1][ty * 4:(ty + h) * 4, tx * 4:(tx + w) * 4] != level).any():
                    continue
                if avoid and any(_to_segment((tx * 32 + w * 16, ty * 32 + h * 16), a, b) < w * 16 + PATH_CLEAR
                                 for a, b in avoid):
                    continue
                if self.fits(unit_type, tx, ty, builder):
                    return tx, ty
        return None

    def _bad(self, unit_type: int, tx: int, ty: int) -> bool:
        """A spot where a build of this type failed less than BAD_FOR ago (the game wouldn't place it)."""
        f = self.g.bad_spots.get((unit_type, tx, ty))
        return f is not None and self.g.hi.frame - f < BAD_FOR

    def fits(self, unit_type: int, tx: int, ty: int, builder: int) -> bool:
        """The game accepts the building there, and the terrain under it is buildable. The game's
        check skips the terrain of fogged tiles (you can order a build into the fog): a spot at the
        fogged edge of a ramp passes, then fails once a unit sees it and the SCV keeps mining. The
        map's terrain is map knowledge, so Gary checks it too (where the env can tell it)."""
        if not self.g.hi.game.can_place(self.g.slot, unit_type, tx, ty, builder):
            return False
        # (the race's rule whether the terrain is known or not: live, without it, it was skipped, and
        # turrets stood where both Command Centers' Comsats go: no scan against lurkers, human game 2)
        if not self.site_ok(unit_type, tx, ty):
            return False
        terrain = self.terrain()
        if terrain is None:
            return True
        w, h = self.g.race.sizes[unit_type]
        return bool(terrain[0][ty:ty + h, tx:tx + w].all())

    def site_ok(self, unit_type: int, tx: int, ty: int) -> bool:
        """This race's placement rule beyond the game's check: no creep under it, now or soon."""
        return not self.creep_coming(unit_type, tx, ty)

    def creep_coming(self, unit_type: int, tx: int, ty: int) -> bool:
        """Creep under the building, now or soon: the game accepts a build on fogged creep it hasn't
        seen, and it fails once the SCV gets there. The map's own creep (Colosseum II has a strip of
        it across each natural's choke) is map knowledge, read at the start; the zerg's spreads
        around its creep buildings, as seen (the ellipse it can reach)."""
        w, h = self.g.race.sizes[unit_type]
        if self._map_creep is not None and self._map_creep[ty:ty + h, tx:tx + w].any():
            return True
        seen = [(u["x"], u["y"]) for u in self.g.hi.observe()["units"]
                if u["owner"] not in (self.g.slot, 11) and u["type"] in CREEP_BUILDINGS]
        corners = [(x * 32 + 16, y * 32 + 16) for x in (tx, tx + w - 1) for y in (ty, ty + h - 1)]
        return any(((x - cx) / CREEP_RX) ** 2 + ((y - cy) / CREEP_RY) ** 2 <= 1
                   for cx, cy in seen for x, y in corners)

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
            if self.fits(self.g.race.town_hall, tx + dx, ty + dy, builder):
                return tx + dx, ty + dy
        return base.tile

    # --- tasks ----------------------------------------------------------------------------------

    def _run_basic(self, obs: dict, mine: list[dict], t: Task) -> None:
        g = self.g
        if g.hi.frame - t.started > TASK_TIMEOUT:     # give up on anything stuck for 40 s
            if t.kind == "hotkey_building":
                g.hotkeyed[t.data["tag"]] = -1        # live without a hotkey for this one
            if t.kind == "build":
                return self.fail_build(t, "stuck for 40 s")
            g.task = None
            return
        self.basic[t.kind](obs, mine, t)

    def _run_command(self, obs: dict, mine: list[dict], t: Task) -> None:
        if self.g.hi.frame - t.started > COMMAND_TIMEOUT:
            LOG.order(self.g.hi.side, self.g.hi.frame, t.kind, "dropped", f"not done in {COMMAND_TIMEOUT // 24} s", by="executor")
            self.g.task = None
            return
        self.commands[t.kind](obs, mine, t)

    def lost_builder(self, t: Task, why: str) -> bool:
        """The build's worker is lost: another one (builder_retries times; True: the task starts over with it,
        as the task on the hands), else the build fails."""
        g = self.g
        if t.data.get("lost", 0) >= self.builder_retries:
            if why == "its worker is gone":
                why = "its SCV is gone"              # (the records' words, as before)
            elif why == "its worker was pulled or is gone":
                why = "its SCV was pulled or is gone"
            self.fail_build(t, why)
            return False
        t.data["lost"] = t.data.get("lost", 0) + 1
        for k in ("worker", "walking", "resume", "looked", "clicked", "money_wait", "unhooked", "reselects", "again_tries",
                  "resent", "resent_at"):
            t.data.pop(k, None)
        t.stage, t.started = "start", g.hi.frame
        g.task = t
        LOG.order(g.hi.side, g.hi.frame, g.race.names.get(t.unit_type, str(t.unit_type)), "retried",
                  f"{why}: another ({t.data['lost']} of {self.builder_retries})", by="executor")
        return True

    def fail_build(self, t: Task, why: str = "") -> None:
        """A building Gary decided on couldn't be placed: the macro layer hears of it (and why: the game's
        records, gary/decision_log.py)."""
        t.data["failed"] = why
        if self.g.macro is not None:
            self.g.macro.build_failed(t)
        self.g.task = None

    def _task_build(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi = self.g, self.g.hi
        if t.stage == "back":
            # builder_back: as pros, the builder's next order queued (shift) on a mineral field, so it
            # goes back to mining once the building stands (human game 2: 11-17 of 30-39 SCVs stood idle
            # from 8:00 to 12:00, builders and repairers done, sent back one by one)
            base = g.base(t.where) if t.where in ("main", "natural") or hasattr(t.where, "minerals") else g.main
            fields = [m for m in base.minerals if screen_of(obs, *m)]
            if fields and obs["selection"] == [t.data["worker"]]:
                hi.right_click(*screen_of(obs, *min(fields, key=lambda m: math.dist(m, base.center))), queued=True)
            g.task = None
            return
        if t.stage == "start":                      # get a worker from the main
            again = next((u for u in mine if u["tag"] == g.retry_with.get(t.unit_type)), None)
            if again is not None and again["order"] not in BUSY_ORDERS and \
                    t.data.get("again_tries", 0) < PICK_TRIES:   # (a failed build's SCV, where it stands)
                if obs["selection"] == [again["tag"]]:
                    g.retry_with.pop(t.unit_type, None)
                    t.data["worker"] = again["tag"]
                    t.stage = "place" if t.where == "main" else "travel"
                    return
                p = screen_of(obs, again["x"], again["y"])
                if p is None:
                    hi.camera_minimap(again["x"], again["y"])
                else:
                    t.data["again_tries"] = t.data.get("again_tries", 0) + 1
                    self.select_worker(obs, again)
                return
            g.retry_with.pop(t.unit_type, None)
            if not g.look_at(obs, t.data.get("from", "main")):
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
            tries = t.data.setdefault("tries", {})
            if w and tries.get(w, 0) < PICK_TRIES and any(u["tag"] == w and self.free_worker(obs, u) for u in mine):
                w = next(u for u in mine if u["tag"] == w)                 # the same one again
            else:
                skip = {k for k, n in tries.items() if n >= PICK_TRIES}
                w = self.pick_worker(obs, mine, t.data.get("near"), skip)
                if w is None and skip:                                      # tried them all: again
                    tries.clear()
                    w = self.pick_worker(obs, mine, t.data.get("near"))
            if not w and t.data.get("from", "main") == "main" and \
                    any(u["type"] in g.race.town_halls and math.dist((u["x"], u["y"]), g.natural.center) < 4 * 32 for u in mine):
                # no free SCV in the main (they mine at the natural, or died there): one from the
                # natural (13 Barracks in a row failed for it in one game)
                t.data["from"] = "natural"
                return
            others = [b for b in self.my_bases(mine) if b is not g.main and b is not g.natural
                      and b.center not in t.data.get("looked_at", [])]
            if not w and t.data.get("from", "main") != "main" and others:
                # nor there: the next base of Gary's with a town hall (gz1's drones mined at its third: 9 builds a
                # game dropped "no free SCV", round G1)
                t.data.setdefault("looked_at", []).append(others[0].center)
                t.data["from"] = others[0]
                return
            if not w:
                self.fail_build(t, "no free SCV")
                return
            t.data["worker"] = w["tag"]
            tries[w["tag"]] = tries.get(w["tag"], 0) + 1
            self.select_worker(obs, w)
            return
        if t.stage == "reselect":                    # a fight put the build aside: the same worker again
            w = next((u for u in mine if u["tag"] == t.data["worker"]), None)
            if not w or w["tag"] in g.pulled | g.block_units:
                self.lost_builder(t, "its worker was pulled or is gone")
                return
            if obs["selection"] == [w["tag"]]:
                t.stage = t.data.pop("resume")
                t.data.pop("looked", None)               # (the spot may be off screen again: look again)
                t.data.pop("clicked", None)
                t.data["reselects"] = 0
                return
            # a neighbor in a stack, that this stage's click selected: use it. Only if the click changed
            # the selection and the SCV stands by Gary's: a drag box that caught nothing leaves the old
            # selection, and live a parked build took the SCV just sent to mine that way, while its
            # own SCV stood on the spot and blocked it (LAN game 1)
            if len(obs["selection"]) == 1 and "clicked" in t.data and obs["selection"] != t.data["clicked"]:
                got = next((u for u in mine if u["tag"] == obs["selection"][0]), None)
                if got and self.free_worker(obs, got) and math.dist((got["x"], got["y"]), (w["x"], w["y"])) < NEIGHBOR:
                    t.data["worker"] = got["tag"]
                    t.stage = t.data.pop("resume")
                    t.data.pop("looked", None)
                    t.data.pop("clicked", None)
                    t.data["reselects"] = 0
                    return
            p = screen_of(obs, w["x"], w["y"])
            if not p:
                hi.camera_minimap(w["x"], w["y"])
                return
            t.data["clicked"] = list(obs["selection"])   # (what was selected before this click)
            t.data["reselects"] = t.data.get("reselects", 0) + 1
            if t.data["reselects"] > RESELECT_TRIES:
                self.fail_build(t, "its SCV wouldn't be selected again")                   # (the macro layer decides again)
                return
            self.select_worker(obs, w)
            return
        if t.stage == "travel":                      # walk the worker to the base
            hi.minimap_right_click(*g.base(t.where).center)
            t.stage = "wait_arrival"
            if g.parked is None:
                # the hands go back to production while it walks (holding them for the walk to the
                # natural, Gary's Barracks stood idle with 464 minerals at 3:15); back to it there
                t.data["walking"] = hi.frame
                g.ignore_idle_until[t.data["worker"]] = hi.frame + TRAVEL_FOR
                t.data["resume"], t.stage = "wait_arrival", "reselect"
                g.parked, g.task = t, None
            return
        if t.stage == "wait_arrival":
            w = next((u for u in mine if u["tag"] == t.data["worker"]), None)
            if not w:
                self.lost_builder(t, "its worker is gone")
                return
            center = g.base(t.where).center
            if math.dist((w["x"], w["y"]), center) > 6 * 32:
                # stopped or taken on the way (a pull, a fight) and left standing: sent again (a third
                # base's builder stood idle in the main for 80 s while the build waited for it)
                going = w["order"] == MOVE and math.dist((w.get("tx", 0), w.get("ty", 0)), center) < 6 * 32
                going = going or hi.frame - t.data.get("resent_at", -10 ** 9) < 2 * 24      # (the last click landing)
                if not going and obs["selection"] == [w["tag"]]:
                    t.data["resent_at"] = hi.frame
                    t.data["resent"] = t.data.get("resent", 0) + 1
                    if t.data["resent"] > RESEND_TRIES:
                        self.fail_build(t, "its SCV wouldn't walk to the base")
                        return
                    hi.minimap_right_click(*center)
                elif not going:
                    t.data["resume"], t.stage = "wait_arrival", "reselect"
                return
            if not g.look_at(obs, t.where):
                return
            if t.where == "natural" and LOC_NATURAL not in hi.camera_locations:
                hi.camera_location_set(LOC_NATURAL)
                return
            t.stage = "place"
            return
        if t.stage == "place":
            base = g.base(t.where)
            if "spot" not in t.data:
                if not g.look_at(obs, t.where):
                    return
                spot = self.town_hall_spot(base, t.data["worker"]) if t.unit_type == g.race.town_hall else \
                    self.spot_near(t.unit_type, t.data["near"], t.data["worker"], t.data.get("level_at"),
                                   t.data.get("avoid")) if "near" in t.data else \
                    self.spot(t.unit_type, base, t.data["worker"])
                if not spot:
                    self.fail_build(t, "no spot for it")
                    return
                t.data["spot"] = spot
            spot = t.data["spot"]
            w, h = g.race.sizes[t.unit_type]
            p = screen_of(obs, spot[0] * 32 + 16, spot[1] * 32 + 16)
            if not p:                                   # the spot is off screen: look there first
                if t.data.get("looked"):
                    self.fail_build(t, "its spot is off screen after a look")
                    return
                t.data["looked"] = True
                hi.camera_minimap(spot[0] * 32 + w * 16, spot[1] * 32 + h * 16)
                return
            budget = obs["me"]["minerals"] - sum(g.race.minerals[ut] for ut, *_ in g.reserved)
            if budget < g.race.minerals[t.unit_type]:
                # the worker walks there while the money comes (as pros send it: it arrives with
                # the money, not after it)
                if not t.data.get("money_wait"):
                    t.data["money_wait"] = True
                    t.data["walking"] = hi.frame
                    if t.unit_type == g.race.gas_building:
                        # beside the geyser, not on it: a right-click on a geyser is a gas order, and
                        # the SCV then refused the refinery (it came a minute late, 3 games in 4)
                        cx, cy = spot[0] * 32 + w * 16, spot[1] * 32 + h * 16
                        mx, my = base.center
                        d = math.dist((cx, cy), (mx, my)) or 1
                        p = screen_of(obs, cx + (mx - cx) / d * 80, cy + (my - cy) / d * 80) or p
                    hi.right_click(*p)
                    # the hands are free meanwhile (production goes on); back to this SCV when the
                    # money is there (resume_parked)
                    g.ignore_idle_until[t.data["worker"]] = hi.frame + PARK_FOR
                    t.data["resume"], t.stage = "place", "reselect"
                    g.parked, g.task = t, None
                return
            builder = next((u for u in mine if u["tag"] == t.data["worker"]), None)
            if builder and builder["order"] == MINING_STILL and not t.data.get("unhooked"):
                # mid-mining, an SCV lets a build order go (3 tries in 7 in OpenBW): off the field first
                t.data["unhooked"] = True
                hi.right_click(*p)
                return
            if hi.build(t.unit_type, *p, order=self.build_order).accepted:
                g.reserved.append((t.unit_type, *spot))
                g.reserved_at[(t.unit_type, *spot)] = hi.frame
                g.builder_of[(t.unit_type, *spot)] = t.data["worker"]
                if getattr(g, "builder_back", False) and t.unit_type != g.race.gas_building:
                    t.stage = "back"            # (a Refinery's builder goes on to gas by itself)
                else:
                    g.task = None

    def _task_hotkey(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi = self.g, self.g.hi
        tag = t.data["tag"]
        where = "natural" if math.dist((t.data["x"], t.data["y"]), g.natural.center) < 10 * 32 else "main"
        if obs["selection"] == [tag]:
            if t.stage == "start":
                pool = key_pool(g, t.unit_type == g.race.town_hall)
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
                if t.unit_type == g.race.town_hall:
                    base = g.main if where == "main" else g.natural
                    m = base.minerals[len(base.minerals) // 2]
                    p = screen_of(obs, *m)
                    if p:
                        hi.right_click(*p)
                else:
                    hi.minimap_right_click(*g.rally)
                    g.rallied[tag] = g.rally
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
        if t.stage == "gather":                           # (the camera moved to the base: the workers may be off
            return self._mine_gather(obs, mine, t, tags)  # screen now, still selected)
        workers = [u for u in mine if u["tag"] in tags and screen_of(obs, u["x"], u["y"])]
        if not workers:
            g.task = None
            return
        ccs = [(c["x"], c["y"]) for c in mine if c["type"] in g.race.town_halls]
        if t.stage == "start" and g.balance and ccs and \
                all(min(math.dist((u["x"], u["y"]), c) for c in ccs) > STRAY for u in workers):
            hi.ctrl_click(*hi.aim(obs, workers[0]))   # (out in the field: every worker on the screen, at once)
            t.stage = "send"
            return
        if t.stage in ("start", "split") and hi.frame < SPLIT_UNTIL and not t.data.get("nosplit") and                 (len(workers) > 1 or t.stage == "split"):
            return self._mine_split(obs, mine, t, workers)
        if t.stage == "start":
            pts = [hi.aim(obs, u) for u in workers]
            # a drag box only picks up own units: a plain click on a worker standing on a
            # mineral field can select the field instead
            hi.box(max(0, min(p[0] for p in pts) - 10), max(0, min(p[1] for p in pts) - 10),
                   min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + 10),
                   min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + 10))
            t.stage = "send"
            return
        # the box may have caught busy workers too (e.g. one walking off to build): never send
        # those back to mining; select an idle one with a click instead. Miners in it are fine: a gather
        # order keeps them gathering (on the current ladder maps miners stand stacked, every box caught
        # some, and 39 of 188 mining tasks in human game 4 ended without one order)
        if not obs["selection"] or not self._mine_selection_ok(obs, mine, tags):
            if t.stage == "send":
                t.stage = "click"
                hi.click(*hi.aim(obs, workers[0]))
                return
            # home_scvs: the box caught marines standing with them (out in the field): each worker
            # clicked in turn, up to MINE_CLICKS, before they're left for 20 s
            clicks = t.data.get("clicks", 1)
            if g.home_scvs and clicks < MINE_CLICKS:
                t.data["clicks"] = clicks + 1
                hi.click(*hi.aim(obs, workers[clicks % len(workers)]))
                return
            for tag in tags:
                g.ignore_idle_until[tag] = hi.frame + 24 * 20
            g.task = None
            return
        # builder_back too: the other idle workers on the screen added (shift-click) before sending, up to
        # MINE_ADD (a box among mining SCVs catches busy ones: then one idle worker a task, never enough)
        if getattr(g, "builder_back", False) and t.stage == "click":
            more = [u for u in workers if u["tag"] not in obs["selection"]]
            if more and t.data.get("added", 0) < MINE_ADD:
                t.data["added"] = t.data.get("added", 0) + 1
                hi.click(*hi.aim(obs, more[0]), shift=True)
                return
        w = workers[0]
        t.data["base"] = self.mine_base(mine, (w["x"], w["y"]), obs).center
        return self._mine_gather(obs, mine, t, tags)

    def _mine_split(self, obs: dict, mine: list[dict], t: Task, workers: list[dict]) -> None:
        """The game's start: the idle workers selected one at a time, each sent to the field with fewest workers (the
        pros' split; four SCVs sent to one field together had mined half the pros' minerals by 10 s, #55)."""
        g, hi = self.g, self.g.hi
        t.stage = "split"
        sent = t.data.setdefault("split", [])
        left = [u for u in workers if u["tag"] not in sent]
        if not left:
            g.task = None
            return
        w = left[0]
        if obs["selection"] != [w["tag"]]:
            t.data["clicks"] = t.data.get("clicks", 0) + 1
            if t.data["clicks"] > SPLIT_CLICKS:       # (it won't be selected alone: the rest go together, as before)
                t.data.update(nosplit=True, tags=[u["tag"] for u in left])
                t.stage = "start"
                return
            hi.click(*hi.aim(obs, w))
            return
        base = self.mine_base(mine, (w["x"], w["y"]), obs)
        m = self._pick_field(obs, mine, base, t)
        if m is None:
            t.data.update(nosplit=True, tags=[u["tag"] for u in left])
            t.data["base"] = base.center
            t.stage = "gather"
            return
        hi.right_click(*field_click(obs, m, base.center))
        sent.append(w["tag"])
        t.data["clicks"] = 0
        g.ignore_idle_until[w["tag"]] = hi.frame + 24 * 20

    def field_loads(self, mine: list[dict], fields: list) -> list[int]:
        """Workers heading to, waiting at or mining each field (by their order's target)."""
        loads = [0] * len(fields)
        for u in mine:
            if u["type"] != self.g.race.worker or u["order"] not in O.MINERAL_ORDERS or not fields:
                continue
            # (its order's target, else where it stands: one mining has no target at its field 2 times in 4)
            for at, near in (((u.get("tx"), u.get("ty")), FIELD_NEAR), ((u["x"], u["y"]), FIELD_BESIDE)):
                if at[0] is None:
                    continue
                k = min(range(len(fields)), key=lambda i: math.dist(fields[i], at))
                if math.dist(fields[k], at) < near:
                    loads[k] += 1
                    break
        return loads

    def _pick_field(self, obs: dict, mine: list[dict], base, t: Task):
        """The base's field on screen for workers sent to mine: the one with fewest workers on it, then the closest to
        its Command Center (every worker sent to the closest stood waiting behind the one mining it, #55); this task's
        own clicks counted too (an order shows a moment later). None: none of its fields on screen."""
        bases = self.my_bases(mine) or [self.g.main]
        fields = self._clear_fields(mine, [m for m in base.minerals if screen_of(obs, *m)] or
                                    [m for b in bases for m in b.minerals if screen_of(obs, *m)])
        if not fields:
            return None
        loads = self.field_loads(mine, fields)
        clicked = [tuple(f) for f in t.data.setdefault("fields", [])]
        k = min(range(len(fields)), key=lambda i: (loads[i] + clicked.count(tuple(fields[i])), math.dist(fields[i], base.center)))
        t.data["fields"].append(list(fields[k]))
        return fields[k]

    def _mine_selection_ok(self, obs: dict, mine: list[dict], tags: set) -> bool:
        """The selection is workers to send mining: SCVs, each one of the task's or one gathering already (none
        walking to build, scouting, pulled or blocking)."""
        g = self.g
        by_tag = {u["tag"]: u for u in mine}
        builders = {g.builder_of.get(r) for r in g.reserved} | ({g.parked.data.get("worker")} if g.parked else set())
        busy = builders | {g.scout} | g.pulled | g.block_units
        for tag in obs["selection"]:
            u = by_tag.get(tag)
            if u is not None and u["type"] in self.box_extras:
                continue
            if u is None or u["type"] != g.race.worker or tag in busy:
                return False
            if tag not in tags and u["order"] not in GATHERING | self.box_orders:
                return False
        return True

    def trim_selection(self, obs: dict, mine: list[dict]) -> bool:
        """A unit the box took with the workers that a mining order would move (a race's flyers): out of the
        selection by its portrait (True: that was this turn's action)."""
        return False

    def click_blockers(self, mine: list[dict]) -> list[tuple[int, int]]:
        """Own units a right-click on a mineral field may hit instead (a race's flyers kept over its bases)."""
        return []

    def _clear_fields(self, mine: list[dict], fields: list) -> list:
        """The fields with none of click_blockers over them (all of them if every one has)."""
        blockers = self.click_blockers(mine)
        return [m for m in fields if not any(math.dist(m, b) < 40 for b in blockers)] or fields

    def _mine_gather(self, obs: dict, mine: list[dict], t: Task, tags: set) -> None:
        """The selected workers right-click a mineral field of the base chosen for them (the closest to its
        Command Center: a field at a base without one means a long walk with every load). Its minerals off
        screen: the camera goes there first, the selection kept, and the field is clicked next turn: a
        minimap right-click on them was a move, and 41 of 188 mining tasks in human game 4 sent workers to
        stand idle by the minerals."""
        g, hi = self.g, self.g.hi
        if self.trim_selection(obs, mine):
            return
        base = next((b for b in g.map.bases if b.center == tuple(t.data["base"])), None) or g.main
        m = self._pick_field(obs, mine, base, t)
        if m is not None:
            hi.right_click(*field_click(obs, m, base.center))
        elif t.stage != "gather":
            hi.camera_minimap(*base.center)
            t.stage = "gather"
            return
        elif g.home_scvs:                             # (the camera didn't get there: the minimap as before)
            hi.minimap_right_click(*min(base.minerals, key=lambda m: math.dist(m, base.center)))
        # whether or not it worked, don't fixate on these workers: look again in 20 s
        for tag in tags:
            g.ignore_idle_until[tag] = hi.frame + 24 * 20
        g.task = None

    def select(self, obs: dict, b: dict) -> bool:
        """Get this building selected (hotkey, or look at it and click). True once it is."""
        g, hi = self.g, self.g.hi
        if obs["selection"] == [b["tag"]]:
            g.select_misses.pop(b["tag"], None)
            return True
        hk = g.hotkeyed.get(b["tag"], -1)
        if hk >= 0:
            hi.hotkey_recall(hk)
            return False
        p = screen_of(obs, b["x"], b["y"])
        if p:
            # a building its clicks don't select (live: the bridge's box for it missed its centre) is left
            # alone for a while, the task ended, so the macro takes another or decides again
            g.select_misses[b["tag"]] = g.select_misses.get(b["tag"], 0) + 1
            if g.select_misses[b["tag"]] > SELECT_TRIES:
                g.select_misses.pop(b["tag"], None)
                g.unselectable[b["tag"]] = hi.frame + UNSELECTABLE_FOR
                g.say(f"can't select its {g.race.names.get(b['type'], b['type'])} at ({b['x']}, {b['y']}): left alone for a while")
                g.task = None
                return False
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
            LOG.order(g.hi.side, hi.frame, f"{d['act']}:{d['id']}", "dropped", "its building is gone", by="executor")
            g.task = None
            return
        if not self.select(obs, b):
            return
        act, uid = d["act"], d["id"]
        if act == TRAIN:
            if hi.train(uid).accepted:
                g.producing[b["tag"]] = (g.producing.get(b["tag"], []) + [uid])[-5:]
                g.ordered_at[b["tag"]] = obs["now"]
        elif act == RESEARCH:
            if hi.research(uid).accepted:
                g.researched.add(uid)
                g.researched_at[uid] = hi.frame
        elif act == UPGRADE:
            if hi.upgrade(uid).accepted:
                g.upgraded[uid] += 1
        elif act == BUILD and not self.command_build(obs, b, uid):
            return
        g.task = None

    def command_build(self, obs: dict, b: dict, uid: int) -> bool:
        """A building built by a selected building (a race's add-on): False to come back next turn."""
        raise NotImplementedError(f"{self.g.race.letter}: no building builds {uid}")

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
            picked = [u["tag"] for u in mine if u["tag"] in obs["selection"] and u["type"] == g.race.worker]
            # the box may have caught a busy worker too (an SCV walking past to place a building:
            # sent to gas, its build was lost; 120 of 176 games had a turret that didn't start):
            # then select again
            if any(not self.free_worker(obs, u) for u in mine if u["tag"] in obs["selection"]):
                picked = []
            if picked:
                hi.right_click(*rp)
                g.gas_workers.setdefault(d["tag"], set()).update(picked)
            t.stage = "start"
            d["tries"] = d.get("tries", 0) + 1
            if d["tries"] >= 5:                 # look again in 20 s rather than keep trying now
                g.ignore_idle_until[d["tag"]] = hi.frame + 24 * 20
                g.task = None
            return
        workers = [u for u in mine if self.free_worker(obs, u) and u["order"] not in O.GAS_ORDERS]
        if not workers:                         # (asked again every turn, it took Gary's hands for good)
            g.ignore_idle_until[d["tag"]] = hi.frame + 24 * 20
            g.task = None
            return
        workers = [u for u in workers if u["order"] in STANDING] or workers    # (where a box finds them)
        w = min(workers, key=lambda u: math.dist((u["x"], u["y"]), (ref["x"], ref["y"])))
        p = hi.aim(obs, w)
        if p is None:
            hi.camera_minimap(w["x"], w["y"])
            return
        x, y = p
        # a small drag box where it will be: a click on a mining worker can select its mineral field
        hi.box(max(0, x - 6), max(0, y - 6), min(hi.p.viewport[0] - 1, x + 6), min(hi.p.viewport[1] - 1, y + 6))
        t.stage = "send"
