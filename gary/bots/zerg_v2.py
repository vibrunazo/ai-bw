"""Zerg v2: Zerg v1's army (gary/bots/zerg_v1.py) on a real economy, like a pro's zergling opening.

v1 kept the dummy's 9-pool on 9 drones: with nothing behind its zerglings, every wait and retreat
just gave Gary time. This one builds what a pro Zerg opens with, through the human interface:

  12 hatch   drones to 12, then a hatchery at its natural
  pool       then the Spawning Pool, then an Extractor; 3 drones on gas once it's done
  speed      Metabolic Boost (zergling speed) at the pool as soon as there are 100 gas
  drones     to 18 over both bases, then zerglings without pause from every hatchery (overlords
             when supply runs short, one at a time per hatchery); the gas drones back to minerals
             once speed is started; a macro hatchery in the main whenever it has 350 minerals and
             no larva to spend them on (4 hatcheries at most)
  rally      both hatcheries rally to a gathering point in front of its natural, where v1's army
             takes them in groups of 12 to its staging point outside Gary's natural
  all in     until then its groups wait at the staging point; once speed is done and it has 20
             zerglings (or at 6:00 at the latest), it musters every zergling there (groups of 6), then
             (80% there, or 25 s on) every group attacks Gary's start at once, without judging and without retreating, and new
             zerglings follow in groups of 6 (its home still defended): a zergling opening's
             timing, before Gary's bio is up

    python -m eval.vs_rush --versions v025 --opponent v2 --seeds 1 2 3 4

It makes what a zergling opening should (4 hatcheries, speed, 35-50 zerglings by 8-10 min on
Fighting Spirit). Against Gary v0.11 (48 seeds, 12 min), before the all-in: Gary lost none at
either timing (won 14-15); with it (going at about 6:00 with 14-20 zerglings, reinforcements
streaming): lost 2 (immediate) and 0 (from 7:00), won 15-16. The dummy's 9-pool, hitting at 3:00
before any bunker, beats Gary 5 times in 48: zerglings alone don't break marines, medics and
bunkers once they're up (a Zerg goes lurkers or mutalisks then).
"""

from __future__ import annotations

import math

from gary import commands as C
from gary.bots.zerg_rush import DRONE, HATCH, LING, MINING, OVERLORD, POOL
from gary.bots.zerg_v1 import ZergV1
from gary.interface import screen_of

EXTRACTOR, METABOLIC_BOOST = 149, 27
SPEED_FRAMES = 100 * 24         # Metabolic Boost's research time (about)
ALL_IN_LINGS, ALL_IN_LATEST = 20, 6 * 60 * 24
MUSTER = 25 * 24                # the all-in goes when 80% of the zerglings are at the staging point, or after this
HATCHERIES = {131, 132, 133}
HATCH_AT = 12                   # drones before the natural's hatchery
DRONES = 18
GAS = 3
GATHER = 5 * 32                 # the gathering point: this far in front of the natural
IDLE_ORDERS = {2, 3, 49}        # Guard, PlayerGuard: standing; Follow: a right-click that hit a unit
BUILD_TRIES = 10                # a building that doesn't show up is ordered again this often (a unit on its spot)
IDLE_TRIES = 6                  # boxes around an idle drone before it's left be for 20 s
MINING_ORDERS = set(range(79, 92))     # moving to, waiting at, mining, returning minerals and gas


class ZergV2(ZergV1):
    attack_ratio = patient_ratio = float("inf")       # no attack before the timing: everyone waits for it

    def __init__(self, hi, mapinfo):
        super().__init__(hi, mapinfo)
        self.main_base = mapinfo.base_near(self.home)
        self.nat_base = mapinfo.natural_of(self.main_base)
        w, h = mapinfo.size
        nc = self.nat_base.center
        d = math.dist(nc, (w / 2, h / 2)) or 1
        self.gather = (int(nc[0] + (w / 2 - nc[0]) / d * GATHER), int(nc[1] + (h / 2 - nc[1]) / d * GATHER))
        self.ordered: dict[str, int] = {}        # build step -> frame ordered
        self.gas: set[int] = set()
        self.rallied_at: set[int] = set()
        self.overlords_at: list[int] = []
        self.task = ""                           # a multi-action step under way
        self.drone_for = None
        self.all_in = False
        self.walking: dict[int, int] = {}       # drone -> frame until which it is walking home (or left be)
        self.idle_tries: dict[int, int] = {}    # idle drone -> boxes around it so far
        self.select_tries = 0                   # boxes around the drone for a step so far
        self.builds_seen: dict[str, int] = {}   # build step -> tries
        self.builder: dict[str, int] = {}       # build step -> the drone ordered to
        self.gas_target = None                  # the extractor a gas drone goes to (None: the first seen)

    def gather_point(self):
        return self.gather

    def act(self) -> None:
        hi = self.hi
        if hi.pending:
            return
        obs = hi.observe()
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == hi.slot]
        self._timing(mine)
        if self.army(obs, mine):                              # (hotkeys and the minimap, mostly)
            return
        count = lambda t: sum(1 for u in mine if u["type"] == t)
        hatches = [u for u in mine if u["type"] in HATCHERIES]
        pool = next((u for u in mine if u["type"] == POOL), None)
        extractor = next((u for u in mine if u["type"] == EXTRACTOR), None)
        drones = [u for u in mine if u["type"] == DRONE and u["completed"]]
        eggs_drones = count(DRONE) + sum(1 for u in mine if u["type"] == 36)    # (eggs: maybe drones)

        # a step under way goes on
        if self.task:
            return self._continue(obs, mine, me)
        if self._mine_idle(obs, mine, drones, hatches):
            return
        if self._build_order(obs, mine, me, hatches, pool, extractor, drones):
            return
        # larvae: overlords when supply runs short, then what this Zerg makes
        if me["minerals"] < 50:
            return
        want = self._want(me, mine, hatches, pool, eggs_drones)
        if want is None or me["supply_used"] + (0 if want == OVERLORD else 1) > me["supply_max"]:
            return
        self._larva(obs, mine, want)

    def _go(self, mine) -> bool:
        """Time for the all-in: speed done and ALL_IN_LINGS zerglings, or ALL_IN_LATEST."""
        lings_now = sum(1 for u in mine if u["type"] == LING)
        speed_done = "speed" in self.ordered and self.hi.frame - self.ordered["speed"] >= SPEED_FRAMES
        return (speed_done and lings_now >= ALL_IN_LINGS) or self.hi.frame >= ALL_IN_LATEST

    def _timing(self, mine) -> None:
        """The all-in: muster everyone at the staging point once it's time (_go), then everyone goes."""
        hi = self.hi
        lings_now = sum(1 for u in mine if u["type"] in self.sizes)
        if not self.all_in:
            if self._go(mine):
                self.all_in, self.muster_from = "muster", hi.frame     # everyone to the staging point first
                self.wave = 6
                self.log(f"all in: {lings_now} zerglings, mustering")
        elif self.all_in == "muster":
            staged = sum(len(self.groups[k]) for k, p in self.phase.items() if p == "staged")
            if staged >= 0.8 * lings_now or hi.frame - self.muster_from >= MUSTER:
                self.all_in = "go"
                self.attack_ratio = self.patient_ratio = 0.0
                self.retreat_ratio = float("inf")
                self.surround = False
                self.log(f"all in: go with {staged} of {lings_now}")

    def _mine_idle(self, obs, mine, drones, hatches) -> bool:
        """Idle drones mine, at the base they stand at (True: it acted)."""
        hi = self.hi
        idle = [u for u in drones if (u["order"] in IDLE_ORDERS and u["tag"] not in self.gas
                                      # (once speed is started, the gas drones go back to minerals)
                                      or self._gas_done() and u["order"] in (81, 82, 84))
                and hi.frame >= self.walking.get(u["tag"], 0)]
        if idle:
            u = idle[0]
            if self._look(obs, (u["x"], u["y"])):
                return True
            sel = [s for s in mine if s["tag"] in obs["selection"]]
            # (idle drones clump: all of them; the box can take larvae and the overlords above too)
            if sel and any(s["tag"] in {i["tag"] for i in idle} for s in sel) \
                    and all(s["type"] in (DRONE, C.LARVA, OVERLORD) for s in sel):
                base = min((self.main_base, self.nat_base), key=lambda b: math.dist(b.center, (u["x"], u["y"])))
                if not any(h["completed"] and math.dist((h["x"], h["y"]), base.center) < 160 for h in hatches) \
                        or math.dist(base.center, (u["x"], u["y"])) > 10 * 32:
                    # away from a working base: walk home first (a minimap order), mine once there
                    hi.minimap_right_click(*self.main_base.minerals[0])
                    for s in sel:
                        self.walking[s["tag"]] = hi.frame + 20 * 24
                    return True
                # (a field with an overlord over it: the click would hit the overlord, the drones follow it)
                lords = [(o["x"], o["y"]) for o in mine if o["type"] == OVERLORD]
                clear = [m for m in base.minerals if not any(math.dist(m, lo) < 40 for lo in lords)] or base.minerals
                m = min(clear, key=lambda m: math.dist(m, (u["x"], u["y"])))
                p = screen_of(obs, *m)
                if p is None:
                    hi.minimap_right_click(*m)
                    for s in sel:
                        self.walking[s["tag"]] = hi.frame + 10 * 24
                    return True
                hi.right_click(*p)
                return True
            self.idle_tries[u["tag"]] = self.idle_tries.get(u["tag"], 0) + 1
            if self.idle_tries[u["tag"]] > IDLE_TRIES:          # can't get it alone: leave it be a while
                self.idle_tries[u["tag"]] = 0
                self.walking[u["tag"]] = hi.frame + 20 * 24
            x, y = screen_of(obs, u["x"], u["y"])
            hi.box(max(0, x - 12), max(0, y - 12), x + 12, y + 12)
            return True
        return False

    def _build_order(self, obs, mine, me, hatches, pool, extractor, drones) -> bool:
        """The buildings, gas, speed and rallies (True: it acted)."""
        hi = self.hi
        # a building ordered that doesn't show up within 30 s: order it again (3 tries)
        have = {"hatch": len(hatches) >= 2, "pool": pool is not None, "extractor": extractor is not None}
        for step, ok in have.items():
            f = self.ordered.get(step)
            if f is not None and f >= 0 and not ok and hi.frame - f > 30 * 24 and self.builds_seen.get(step, 0) < BUILD_TRIES:
                self.builds_seen[step] = self.builds_seen.get(step, 0) + 1
                del self.ordered[step]
        # the build: natural hatchery at 12, pool, extractor
        if "hatch" not in self.ordered and len(drones) >= HATCH_AT and me["minerals"] >= 250:
            self._start_build("hatch", HATCH, self.nat_base.tile, obs, drones)
            return True
        if "hatch" in self.ordered and "pool" not in self.ordered and me["minerals"] >= 200:
            spot = self._pool_spot()
            if spot:
                self._start_build("pool", POOL, spot, obs, drones)
                return True
        if "pool" in self.ordered and "extractor" not in self.ordered and me["minerals"] >= 50 and self.main_base.geysers:
            gx, gy = self.main_base.geysers[0]
            self._start_build("extractor", EXTRACTOR, ((gx - 64) // 32, (gy - 32) // 32), obs, drones)
            return True
        # gas
        if extractor and extractor["completed"] and len(self.gas & {u["tag"] for u in drones}) < GAS \
                and not self._gas_done():
            miners = [u for u in drones if u["order"] == MINING and u["tag"] not in self.gas
                      and math.dist((u["x"], u["y"]), self.main_base.center) < 12 * 32]
            if miners:
                self.task, self.drone_for = "gas", self._pick(miners, obs)["tag"]
                self._continue(obs, mine, me)
                return True
        if self._macro_hatch(obs, mine, me, hatches, drones):
            return True
        # speed
        if pool and pool["completed"] and "speed" not in self.ordered and me["gas"] >= 100 and me["minerals"] >= 100:
            if self._look(obs, (pool["x"], pool["y"])):
                return True
            if obs["selection"] == [pool["tag"]]:
                if hi.upgrade(METABOLIC_BOOST).accepted:
                    self.ordered["speed"] = hi.frame
                    self.gas.clear()                          # (enough gas: back to minerals as they idle)
                return True
            hi.click(*screen_of(obs, pool["x"], pool["y"]))
            return True
        return self._rallies(obs, hatches, pool)

    def _macro_hatch(self, obs, mine, me, hatches, drones) -> bool:
        """A macro hatchery in the main: money it can't spend, no larva to spend it on (4 at most)."""
        hi = self.hi
        larvae_now = sum(1 for u in mine if u["type"] == C.LARVA)
        pending = [k for k, f in self.ordered.items() if k.startswith("macro") and f >= 0 and hi.frame - f < 60 * 24]
        if len(hatches) >= 2 and len(hatches) < 4 and me["minerals"] >= 350 and larvae_now == 0 and not pending:
            spot = self._spot_near(HATCH, self.home)
            if spot:
                self._start_build(f"macro{len(hatches)}_{hi.frame}", HATCH, spot, obs, drones)
                return True
        return False

    def _rallies(self, obs, hatches, pool) -> bool:
        """Every finished hatchery rallies to the gathering point (once the pool is there)."""
        hi = self.hi
        for h in hatches:
            if h["completed"] and h["tag"] not in self.rallied_at and pool:
                if self._look(obs, (h["x"], h["y"])):
                    return True
                if obs["selection"] == [h["tag"]]:
                    hi.minimap_right_click(*self.gather)
                    self.rallied_at.add(h["tag"])
                    return True
                top = screen_of(obs, h["x"], h["y"] - 24)
                if top:
                    hi.click(*top)
                return True
        return False

    def _gas_done(self) -> bool:
        """The gas drones go back to minerals once speed is started (100 gas is all this Zerg needs)."""
        return "speed" in self.ordered

    def _want(self, me, mine, hatches, pool, eggs_drones):
        """What the next larva becomes: an overlord when supply runs short, drones to DRONES, then
        zerglings (None: nothing now)."""
        hi = self.hi
        count = lambda t: sum(1 for u in mine if u["type"] == t)
        self.overlords_at = [f for f in self.overlords_at if hi.frame - f < 24 * 25]
        want = None
        if self._overlord_due(me, hatches):
            want = OVERLORD
        elif eggs_drones < DRONES and (not pool or not pool["completed"] or count(LING) >= 6):
            want = DRONE
        elif pool and pool["completed"]:
            want = LING
        return want

    def _overlord_due(self, me, hatches) -> bool:
        """Supply runs short: an overlord (one at a time per finished hatchery)."""
        return me["supply_max"] - me["supply_used"] <= 2 * len(hatches) and me["supply_max"] < 200 \
            and me["minerals"] >= 100 and len(self.overlords_at) < max(1, len([h for h in hatches if h["completed"]]))

    def _larva(self, obs, mine, want) -> None:
        """A larva (selected with a tiny box) becomes want."""
        hi = self.hi
        larvae = [u for u in mine if u["type"] == C.LARVA]
        if not larvae:
            return
        sel = [u for u in mine if u["tag"] in obs["selection"]]
        if sel and all(u["type"] == C.LARVA for u in sel):
            if hi.morph(want).accepted and want == OVERLORD:
                self.overlords_at.append(hi.frame)
            return
        on = [lv for lv in larvae if screen_of(obs, lv["x"], lv["y"])]
        if not on:
            lv = larvae[(hi.frame // 48) % len(larvae)]
            hi.camera_minimap(lv["x"], lv["y"])
            return
        clear = [lv for lv in on if not any(u["type"] != C.LARVA and abs(u["x"] - lv["x"]) < 12
                                            and abs(u["y"] - lv["y"]) < 12 for u in mine)]
        lv = (clear or on)[(hi.frame // 48) % len(clear or on)]
        x, y = screen_of(obs, lv["x"], lv["y"])
        hi.box(max(0, x - 3), max(0, y - 3), x + 3, y + 3)

    def _spot_near(self, unit_type: int, at, r0: int = 4, r1: int = 14):
        """The first spot (top-left tile) for unit_type in rings around a point, or None."""
        cx, cy = at[0] // 32, at[1] // 32
        for r in range(r0, r1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r and self.hi.game.can_place(self.hi.slot, unit_type, cx + dx, cy + dy, 0):
                        return cx + dx, cy + dy
        return None

    # --- multi-action steps --------------------------------------------------------------------

    def _start_build(self, step: str, unit_type: int, tile, obs, drones) -> None:
        """A drone from the main (one mining) builds unit_type at tile (top-left)."""
        miners = [u for u in drones if u["order"] == MINING and math.dist((u["x"], u["y"]), self.main_base.center) < 12 * 32] \
            or drones
        if not miners:
            return
        self.task, self.drone_for = f"build:{step}:{unit_type}:{tile[0]}:{tile[1]}", self._pick(miners, obs)["tag"]
        self._continue(obs, [], obs["me"])

    def _pick(self, drones: list[dict], obs) -> dict:
        """A drone a tiny box takes alone: none of the Zerg's units (overlords above, larvae) on it,
        else any (in turn)."""
        others = [u for u in obs["units"] if u["owner"] == self.hi.slot and u["type"] != DRONE]
        alone = [d for d in drones if not any(abs(o["x"] - d["x"]) < 24 and abs(o["y"] - d["y"]) < 24 for o in others)]
        return (alone or drones)[(self.hi.frame // 48) % len(alone or drones)]

    def _continue(self, obs, mine, me) -> None:
        hi = self.hi
        drone = next((u for u in obs["units"] if u["tag"] == self.drone_for), None)
        if drone is None:
            self.task = ""
            return
        if obs["selection"] != [drone["tag"]]:                 # select the drone (a tiny box)
            if self._look(obs, (drone["x"], drone["y"])):
                return
            self.select_tries += 1
            if self.select_tries > IDLE_TRIES:                 # it won't come alone: another drone, later
                self.select_tries, self.task = 0, ""
                return
            x, y = screen_of(obs, drone["x"], drone["y"])
            hi.box(max(0, x - 2), max(0, y - 2), x + 2, y + 2)
            return
        self.select_tries = 0
        if self.task == "gas":
            ext = next((u for u in obs["units"] if u["owner"] == hi.slot and u["type"] == EXTRACTOR
                        and self.gas_target in (None, u["tag"])), None)
            if ext is None:
                self.task = ""
                return
            if self._look(obs, (ext["x"], ext["y"])):
                return
            hi.right_click(*screen_of(obs, ext["x"], ext["y"]))
            self.gas.add(drone["tag"])
            self.task = ""
            return
        _, step, unit_type, tx, ty = self.task.split(":")
        unit_type, tx, ty = int(unit_type), int(tx), int(ty)
        p = (tx * 32 + 16, ty * 32 + 16)
        if self._look(obs, p):
            return
        if drone["order"] in MINING_ORDERS:                 # a drone busy mining ignores a build order:
            hi.right_click(*screen_of(obs, *p))             # walk it toward the spot first
            return
        if hi.build(unit_type, *screen_of(obs, *p), order=C.ORDER_DRONE_START_BUILD).accepted:
            self.ordered[step] = hi.frame
            self.builder[step] = drone["tag"]
        else:
            self.ordered.setdefault(f"{step}_tries", 0)
            self.ordered[f"{step}_tries"] += 1
            if self.ordered[f"{step}_tries"] > 20:          # can't: give it up
                self.ordered[step] = -1
        self.task = ""

    def _look(self, obs, p) -> bool:
        """The camera on p (True if it had to move: that was this turn's action)."""
        if screen_of(obs, int(p[0]), int(p[1])):
            return False
        self.hi.camera_minimap(int(p[0]), int(p[1]))
        return True
