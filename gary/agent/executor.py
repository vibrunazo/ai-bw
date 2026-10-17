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
from gary.decision_log import LOG
from gary.interface import screen_of
from gary.mapinfo import Base

CC, SCV = T.CC, T.SCV
COST = {unit: minerals for unit, (minerals, _gas) in T.COST.items()}
SIZE = T.SIZE                                              # tiles
SELECT_TRIES = 6                      # clicks on a building that select nothing before it's left alone...
UNSELECTABLE_FOR = 60 * 24            # ...for this long (live, one took 1,335 clicks and froze production)
HK_CC = [4, 5]                        # one hotkey per Command Center (buildings can't share one)
HK_RAX = [6, 7, 8, 9]
STRAY = 12 * 32                              # home_scvs: a worker this far from every Command Center is astray
MINE_CLICKS = 4                               # home_scvs: clicks to get an idle worker alone (a box takes marines too)
FIGHT_CALM = 10 * 24                          # home_scvs: pulled or attack-moving workers are left alone this long
                                              # after zerg were last at Gary's bases (a fight's defenders)
SATURATED, UNDER = 1.8, 1.0                   # balance: workers a mineral patch at a base that gives some away,
                                              # and at one that takes them
TRANSFER_EVERY = 10 * 24                      # balance: bases' saturation looked at this often
DANGER_FOR = 20 * 24                          # balance: a base with zerg fighters seen near this recently isn't sent to
GATHERING = set(range(79, 92))                # orders: to, at, mining and returning minerals or gas
ZERG_FIGHTERS = {37, 38, 39, 43, 103, 46, 47, 44, 45, 62}
ATTACK_MOVE_ORDER, PATROL_ORDER = 14, 152
RALLY_DOWN, RALLY_OUT = 285 * 24, 450 * 24   # rally_ramp: inside the main's ramp until 4:45, at its
                                             # bottom until 7:30, then in front of the natural (pros' marines
                                             # at 3:00-4:00: the ramp's top 33%, the main 30%; at 5:00-6:00
                                             # its bottom 30-34%, the natural 17-19%; Gary's rally in front
                                             # of the natural from the start put a quarter of its marines in
                                             # the open)
BUSY_ORDERS = {30, 31, 33}            # placing / constructing a building
BAD_FOR = 60 * 24                     # a spot where a build failed (not for money) isn't tried again for this long
LOST_ORDER = 6 * 24                   # a reserved spot with no SCV on its way after this: the order was lost
RESUME_AFTER = 24                     # an unfinished building nobody has built for a second gets an SCV
RESUME_RETRY = 24 * 4                 # ...and if that one doesn't arrive either, another after 4 s
IDLE = 3                              # order "PlayerGuard": standing around
MINING_STILL = 87                     # order "MiningMinerals": at its mineral field (lets a build order go)
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
TASK_TIMEOUT = 24 * 40                # build, hotkey, mine and resume tasks give up after 40 s
COMMAND_TIMEOUT = 24 * 20             # commands and gas after 20 s
CREEP_BUILDINGS = {131, 132, 133, 143, 144, 146}   # Hatchery, Lair, Hive, Creep, Spore and Sunken Colony
CREEP_RX, CREEP_RY = 320, 200         # creep spreads up to this far from one (pixels, an ellipse; OpenBW's
                                      # spread_creep), a tile every few seconds



def _to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """The distance from p to the segment a-b."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / ((dx * dx + dy * dy) or 1)))
    return math.dist(p, (ax + t * dx, ay + t * dy))

class ScriptedExecutor:
    def __init__(self, g: Gary):
        self.g = g
        self.next_sweep = 0                           # next look for idle workers off screen
        self.stalled_since: dict[int, int] = {}       # unfinished building -> since when nobody builds it
        self._terrain_cache = None                    # see terrain()
        self._ramp = None                             # rerally: (spot inside the main's ramp, its bottom), or False
        self.danger: dict[tuple, int] = {}            # balance: base center -> frame zerg fighters were last seen near it
        self.next_transfer = 0
        # the map's own creep (map knowledge, read now: at the start of a game, before any zerg's)
        creep = getattr(g.hi.game, "creep", None)
        self._map_creep = creep() if creep else None
        for kind in ("build", "hotkey_building", "mine", "resume", "rerally", "transfer"):
            g.task_runners[kind] = self._run_basic
        g.task_runners["command"] = self._run_command
        g.task_runners["gas"] = self._run_command

    # --- chores, in the arbiter's order ------------------------------------------------------

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
            cx, cy = tx * 32 + SIZE[ut][0] * 16, ty * 32 + SIZE[ut][1] * 16
            built = any(u["type"] == ut and math.dist((u["x"], u["y"]), (cx, cy)) < 48 for u in mine)
            age = g.hi.frame - g.reserved_at.get(r, 0)
            heading = any(u["type"] == SCV and u["order"] in BUSY_ORDERS and "tx" in u
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
                if me is not None and me["minerals"] < T.COST[ut][0]:
                    g.say(f"its {T.NAMES.get(ut, ut)} failed for money: again")
                    LOG.order("T", g.hi.frame, f"{T.NAMES.get(ut, ut)} at {(tx, ty)}", "retried", "failed for money",
                              by="executor")
                elif g.hi.frame - g.failed_at.get(r, -10 ** 9) < BAD_FOR:
                    g.bad_spots[r] = g.hi.frame              # (twice: somewhere else)
                    g.say(f"couldn't place its {T.NAMES.get(ut, ut)} there twice: elsewhere")
                    LOG.order("T", g.hi.frame, f"{T.NAMES.get(ut, ut)} at {(tx, ty)}", "retried",
                              "the game wouldn't place it there twice: elsewhere", by="executor")
                else:
                    # the game's pathfinder can leave an SCV stuck short of the spot (the order then
                    # fails): the same SCV again from where it stands, a new order clears that
                    g.failed_at[r] = g.hi.frame
                    g.retry_with[ut] = builder["tag"]
                    g.say(f"its {T.NAMES.get(ut, ut)} didn't start: again, with the same SCV")
                    LOG.order("T", g.hi.frame, f"{T.NAMES.get(ut, ut)} at {(tx, ty)}", "retried",
                              "its SCV stopped and nothing started: again, the same SCV", by="executor")
            elif not built and age < 24 * 40:
                g.say(f"its {T.NAMES.get(ut, ut)} order was lost: again")
                LOG.order("T", g.hi.frame, f"{T.NAMES.get(ut, ut)} at {(tx, ty)}", "retried",
                          "no SCV on its way to it: the order was lost", by="executor")
            g.reserved_at.pop(r, None)
            g.builder_of.pop(r, None)
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
        ccs = [(c["x"], c["y"]) for c in mine if c["type"] == CC]
        return bool(ccs) and min(math.dist((u["x"], u["y"]), c) for c in ccs) > STRAY

    def idle_mining(self, obs: dict, done: list[dict]) -> bool:
        """Idle workers on screen go back to mining (game start, after building something)."""
        g = self.g
        idle = [u for u in done if u["type"] == SCV and self._idle(u, done) and screen_of(obs, u["x"], u["y"])
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
        away = [u for u in done if u["type"] == SCV and self._idle(u, done) and u["tag"] not in busy
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
            if b["type"] in g.production_types and b["type"] != CC and hk is not None and hk >= 0 \
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
        n = sum(1 for u in mine if u["type"] == SCV and u["order"] in GATHERING and math.dist((u["x"], u["y"]), base.center) < 10 * 32)
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
        miners = sorted((u for u in mine if u["type"] == SCV and u["order"] in GATHERING and u["tag"] not in g.pulled | g.block_units
                         and math.dist((u["x"], u["y"]), src.center) < 10 * 32),
                        key=lambda u: math.dist((u["x"], u["y"]), line))[:k]
        g.task = Task("transfer", data={"tags": [u["tag"] for u in miners], "to": dst.center}, started=hi.frame)
        g.say(f"transfers {len(miners)} workers to the base at {dst.center[0] // 32},{dst.center[1] // 32}")
        return True

    def _task_transfer(self, obs: dict, mine: list[dict], t: Task) -> None:
        """A box around the chosen workers (at their mineral line), then a minimap right-click on the
        other base's minerals."""
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
        if sel and all(u["type"] == SCV for u in sel) and len(sel) >= 2:
            base = next(b for b in g.map.bases if b.center == tuple(t.data["to"]))
            hi.minimap_right_click(*min(base.minerals, key=lambda m: math.dist(m, base.center)))
            g.task = None
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
        return u["type"] == SCV and u["completed"] and u["order"] not in BUSY_ORDERS \
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
            self.fail_build(t, "its SCV is gone")
            return False
        if t.data.get("resume") == "wait_arrival":           # walking to another base: until it's there
            w = next(u for u in mine if u["tag"] == t.data["worker"])
            center = g.base(t.where).center
            walking = w["order"] == MOVE and math.dist((w.get("tx", 0), w.get("ty", 0)), center) < 6 * 32
            if (math.dist((w["x"], w["y"]), center) > 6 * 32 and g.hi.frame - t.data["walking"] < TRAVEL_FOR
                    and (walking or g.hi.frame - t.data["walking"] < 2 * 24)):
                return False                                  # (stopped short: back to it now, to send it again)
        else:
            budget = obs["me"]["minerals"] - sum(COST[ut] for ut, *_ in g.reserved)
            if budget < COST[t.unit_type] and g.hi.frame - t.data["walking"] < PARK_FOR:
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
        if len(sel) > 1 and all(u["type"] == SCV for u in sel):
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
                 if w not in seen or seen[w]["order"] in T.GAS_ORDERS or seen[w]["order"] == RESET_HARVEST}
        alive |= {t for t, u in seen.items() if u["type"] == T.SCV and u["order"] in T.GAS_ORDERS
                  and u.get("order_target") == refinery}
        g.gas_workers[refinery] = alive
        return len(alive)

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
            if any(abs(tx - rx) < SIZE[rt][0] + 1 and abs(ty - ry) < SIZE[rt][1] + 1 for rt, rx, ry in self.g.reserved) \
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
        w, h = SIZE[unit_type]
        cx, cy = near[0] // 32, near[1] // 32
        terrain = self.terrain()
        level = int(terrain[1][level_at[1] // 8, level_at[0] // 8]) if terrain and level_at else None
        cands = sorted((math.dist((tx * 32 + w * 16, ty * 32 + h * 16), near), tx, ty)
                       for ty in range(cy - 8, cy + 8) for tx in range(cx - 8, cx + 8))
        for same_level in ((True, False) if level is not None else (False,)):
            for _, tx, ty in cands:
                if any(abs(tx - rx) < SIZE[rt][0] + 1 and abs(ty - ry) < SIZE[rt][1] + 1 for rt, rx, ry in self.g.reserved) \
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
        # (these two whether the terrain is known or not: live, without it, they were skipped, and
        # turrets stood where both Command Centers' Comsats go: no scan against lurkers, human game 2)
        if self.on_addon_spot(unit_type, tx, ty) or self.creep_coming(unit_type, tx, ty):
            return False
        terrain = self.terrain()
        if terrain is None:
            return True
        w, h = SIZE[unit_type]
        return bool(terrain[0][ty:ty + h, tx:tx + w].all())

    def on_addon_spot(self, unit_type: int, tx: int, ty: int) -> bool:
        """The building would stand where one of Gary's buildings' add-on goes (a Comsat Station by
        the natural's Command Center was asked for every second for minutes: its bunker stood there)."""
        g = self.g
        w, h = SIZE[unit_type]
        parents = set(T.ADDON_PARENT.values())
        spots = [((u["x"] - SIZE[u["type"]][0] * 16) // 32, (u["y"] - SIZE[u["type"]][1] * 16) // 32)
                 for u in g.hi.observe()["units"] if u["owner"] == g.slot and u["type"] in parents]
        spots += [(rx, ry) for rt, rx, ry in g.reserved if rt in parents]
        ox, oy = T.ADDON_OFFSET
        return any(tx < px + ox + 2 and px + ox < tx + w and ty < py + oy + 2 and py + oy < ty + h for px, py in spots)

    def creep_coming(self, unit_type: int, tx: int, ty: int) -> bool:
        """Creep under the building, now or soon: the game accepts a build on fogged creep it hasn't
        seen, and it fails once the SCV gets there. The map's own creep (Colosseum II has a strip of
        it across each natural's choke) is map knowledge, read at the start; the zerg's spreads
        around its creep buildings, as seen (the ellipse it can reach)."""
        w, h = SIZE[unit_type]
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
                return self.fail_build(t, "stuck for 40 s")
            g.task = None
            return
        {"build": self._task_build, "hotkey_building": self._task_hotkey, "mine": self._task_mine,
         "resume": self._task_resume, "rerally": self._task_rerally, "transfer": self._task_transfer}[t.kind](obs, mine, t)

    def _run_command(self, obs: dict, mine: list[dict], t: Task) -> None:
        if self.g.hi.frame - t.started > COMMAND_TIMEOUT:
            LOG.order("T", self.g.hi.frame, t.kind, "dropped", f"not done in {COMMAND_TIMEOUT // 24} s", by="executor")
            self.g.task = None
            return
        (self._task_command if t.kind == "command" else self._task_gas)(obs, mine, t)

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
                    any(u["type"] == CC and math.dist((u["x"], u["y"]), g.natural.center) < 4 * 32 for u in mine):
                # no free SCV in the main (they mine at the natural, or died there): one from the
                # natural (13 Barracks in a row failed for it in one game)
                t.data["from"] = "natural"
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
                self.fail_build(t, "its SCV was pulled or is gone")
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
                self.fail_build(t, "its SCV is gone")
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
                spot = self.town_hall_spot(base, t.data["worker"]) if t.unit_type == CC else \
                    self.spot_near(t.unit_type, t.data["near"], t.data["worker"], t.data.get("level_at"),
                                   t.data.get("avoid")) if "near" in t.data else \
                    self.spot(t.unit_type, base, t.data["worker"])
                if not spot:
                    self.fail_build(t, "no spot for it")
                    return
                t.data["spot"] = spot
            spot = t.data["spot"]
            w, h = SIZE[t.unit_type]
            p = screen_of(obs, spot[0] * 32 + 16, spot[1] * 32 + 16)
            if not p:                                   # the spot is off screen: look there first
                if t.data.get("looked"):
                    self.fail_build(t, "its spot is off screen after a look")
                    return
                t.data["looked"] = True
                hi.camera_minimap(spot[0] * 32 + w * 16, spot[1] * 32 + h * 16)
                return
            budget = obs["me"]["minerals"] - sum(COST[ut] for ut, *_ in g.reserved)
            if budget < COST[t.unit_type]:
                # the worker walks there while the money comes (as pros send it: it arrives with
                # the money, not after it)
                if not t.data.get("money_wait"):
                    t.data["money_wait"] = True
                    t.data["walking"] = hi.frame
                    if t.unit_type == T.REFINERY:
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
            if hi.build(t.unit_type, *p).accepted:
                g.reserved.append((t.unit_type, *spot))
                g.reserved_at[(t.unit_type, *spot)] = hi.frame
                g.builder_of[(t.unit_type, *spot)] = t.data["worker"]
                if getattr(g, "builder_back", False) and t.unit_type != T.REFINERY:
                    t.stage = "back"            # (a Refinery's builder goes on to gas by itself)
                else:
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
        workers = [u for u in mine if u["tag"] in tags and screen_of(obs, u["x"], u["y"])]
        if not workers:
            g.task = None
            return
        ccs = [(c["x"], c["y"]) for c in mine if c["type"] == CC]
        if t.stage == "start" and g.balance and ccs and \
                all(min(math.dist((u["x"], u["y"]), c) for c in ccs) > STRAY for u in workers):
            hi.ctrl_click(*hi.aim(obs, workers[0]))   # (out in the field: every worker on the screen, at once)
            t.stage = "send"
            return
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
        # those back to mining; select an idle one with a click instead
        if not obs["selection"] or not set(obs["selection"]) <= tags:
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
        # the closest mineral field to the Command Center of the nearest base Gary owns (a field
        # at a base without a Command Center means a long walk with every load)
        w = workers[0]
        bases = self.my_bases(mine) or [g.main]
        base = self.mine_base(mine, (w["x"], w["y"]), obs)
        fields = [m for m in base.minerals if screen_of(obs, *m)] or \
                 [m for b in bases for m in b.minerals if screen_of(obs, *m)]
        if fields:
            m = min(fields, key=lambda m: math.dist(m, base.center))
            hi.right_click(*screen_of(obs, *m))
        elif g.home_scvs:                             # (away from the base: its minerals on the minimap)
            hi.minimap_right_click(*min(base.minerals, key=lambda m: math.dist(m, base.center)))
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
            scvs = [u for u in mine if u["type"] == SCV and u["completed"] and u["tag"] not in g.pulled | g.block_units
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
                g.say(f"can't select its {T.NAMES.get(b['type'], b['type'])} at ({b['x']}, {b['y']}): left alone for a while")
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
            LOG.order("T", hi.frame, f"{d['act']}:{d['id']}", "dropped", "its building is gone", by="executor")
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
                g.researched_at[uid] = hi.frame
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
        workers = [u for u in mine if self.free_worker(obs, u) and u["order"] not in T.GAS_ORDERS]
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
