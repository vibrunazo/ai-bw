"""Gary v0.9, Terran: v0.8 that holds early pools (#2).

v0.8 never noticed an early Spawning Pool: its learned macro kept planning an expansion, gas and
tech while the zerglings came, and it met the first wave with three marines and no bunker. v0.9
does what a player does against an early pool:

  detect   a Spawning Pool seen before 3:00 while only one hatchery is known, zerglings before
           4:00, or 6 at once before 5:00 (from the scout or at home)
  bunker   at the top of the main ramp (gary/terrain.py; only where the map analysis is reliable,
           else between the main and its natural), Barracks rallied just behind it; marines load
           in (v0.7)
  hold     the army model doesn't send the marines anywhere: they stay behind the bunker (the
           fight model still fights with them)
  macro    the standard answer to a one-base all-in, not the learned macro (whose likely picks,
           expansion, gas and tech, are what it mustn't do now, and the rest fall below its
           threshold: v0.9 built nothing for minutes): a depot when supply runs low, marines and
           SCVs nonstop, a second and third Barracks when money piles up. Until it's safe: from
           5:00, a finished bunker and as many marines as half the most zerglings seen at once, 6
           at least. A one-base all-in that keeps coming keeps Gary home
  load     marines go into the bunker as soon as it's done, from anywhere in the main, not when
           the zerglings are already there (v0.4's rule)
  repair   2 SCVs repair the bunker when it's damaged and zerglings are near

Everything else is v0.8.

    python -m gary.bots.terran_v09 --map path/to/map.scx --minutes 12 --style 1
    python -m gary.bots.terran_v09 --live --style 1
"""

from __future__ import annotations

import argparse
import math

from gary import terran as T
from gary.bots import terran_v01 as v01
from gary.bots.terran_v01 import Task
from gary.bots.terran_v02 import BUILD, EXPAND, TRAIN, WAIT, latest_model
from gary.bots.terran_v03 import latest_army_model
from gary.bots.terran_v08 import TerranGaryV8, latest_rl_fight_model
from gary.env import LIVE_COMMAND_DELAY
from gary.interface import screen_of
from gary.policy.army import ArmyModel, is_building, is_worker, supply_x2
from gary.policy.fight_cmd import CommandModel
from gary.policy.macro import MacroModel
from gary.terrain import map_analysis

POOL, LING = 142, 37
HATCHERIES = {131, 132, 133}
POOL_BY = 3 * 60 * 24                # a Spawning Pool seen before 3:00 with one hatchery is an early pool...
LINGS_BY = 4 * 60 * 24               # ...and so are zerglings before 4:00
MANY_LINGS, MANY_LINGS_BY = 6, 5 * 60 * 24   # ...or this many at once before 5:00
SAFE_MARINES = 6
SAFE_FROM = 5 * 60 * 24              # the first wave comes around 4:00: not safe before this
RUSH_SCVS = 20                       # SCVs to keep making while defending (one base)
RUSH_RAX = 3
RAX_MONEY = 200                      # minerals left over: another Barracks
DEFENSE = {T.SCV, T.MARINE, T.DEPOT, T.RAX, T.BUNKER}
REPAIR_ORDER = 34
REPAIRERS = 2
THREAT = 9 * 32                      # zerglings this close to the bunker: repair it
LOAD_FROM = 40 * 32                  # during the rush, marines this far from the bunker go in
BUNKER_RETRY = 20 * 24               # a bunker order that didn't take: try again this much later


class TerranGaryV9(TerranGaryV8):
    version = "v0.9"

    def __init__(self, hi, mapinfo, *args, **kwargs):
        super().__init__(hi, mapinfo, *args, **kwargs)
        self.rush = False                # an early pool was seen: defend first
        self.rush_over = False
        self.hatcheries_seen: set[int] = set()
        self.most_lings = 0              # the most zerglings seen at once
        self.next_bunker = 0
        self.next_repair = 0
        analysis = map_analysis(hi.game, mapinfo)["starts"]
        mine = min(analysis.values(), key=lambda a: math.dist(a["start"], self.main.center), default=None)
        if mine and mine["reliable"] and math.dist(mine["start"], self.main.center) < 64:
            # the bunker goes just inside the main, a little up from the ramp's top
            top, bottom = mine["ramp_top"], mine["ramp_bottom"]
            d = math.dist(top, bottom) or 1
            self.bunker_spot = (int(top[0] + (top[0] - bottom[0]) / d * 48), int(top[1] + (top[1] - bottom[1]) / d * 48))
        else:                            # no trusted ramp: between the main and its natural
            self.bunker_spot = (int(self.main.center[0] * 0.6 + self.natural.center[0] * 0.4),
                                int(self.main.center[1] * 0.6 + self.natural.center[1] * 0.4))

    # --- noticing the rush, and when it's over ------------------------------------------------

    def act(self) -> None:
        hi = self.hi
        if not hi.pending and not self.rush_over:
            obs = hi.observe()
            enemy = [u for u in obs["units"] if u["owner"] not in (self.slot, 11)]
            self.hatcheries_seen |= {u["tag"] for u in enemy if u["type"] in HATCHERIES}
            lings = sum(1 for u in enemy if u["type"] == LING)
            self.most_lings = max(self.most_lings, lings)
            if not self.rush and ((hi.frame <= POOL_BY and any(u["type"] == POOL for u in enemy)
                                   and len(self.hatcheries_seen) <= 1) or
                                  (hi.frame <= LINGS_BY and lings) or (hi.frame <= MANY_LINGS_BY and lings >= MANY_LINGS)):
                self.rush = True
                # marines gather a few tiles behind the bunker, toward the main (not on its spot,
                # where they'd keep the SCV from placing it)
                bx, by = self.bunker_spot
                mx, my = self.main.center
                d = math.dist((bx, by), (mx, my)) or 1
                self.rally = (int(bx + (mx - bx) / d * 4 * 32), int(by + (my - by) / d * 4 * 32))
                self._say("early pool: bunker at the ramp, marines first, no expansion or tech yet")
            if self.rush:
                mine = [u for u in obs["units"] if u["owner"] == self.slot]
                bunker = any(u["type"] == T.BUNKER and u["completed"] for u in mine)
                seen = {u["tag"] for u in mine if u["type"] == T.MARINE}
                tags = {u["tag"] for u in mine}
                inside = set().union(set(), *(m for b, m in self.loaded.items() if b in tags)) - tags
                marines = len(seen) + len(inside)   # the ones in the bunker aren't seen
                if hi.frame >= SAFE_FROM and bunker and marines >= max(SAFE_MARINES, self.most_lings // 2):
                    self.rush, self.rush_over = False, True
                    self._say("the early pool is held: back to the usual plan")
        super().act()

    # --- the army stays home during the rush, in the bunker ------------------------------------

    @property
    def bunker_from(self) -> int:
        return LOAD_FROM if self.rush else super().bunker_from

    def _bunker_wanted(self, b: dict, enemy: list[dict]) -> bool:
        return self.rush or super()._bunker_wanted(b, enemy)

    def _army(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        if self.rush:                    # the army model's moves leave the bunker empty: none now
            done = [u for u in done if u["type"] not in self.army_types]
        return super()._army(obs, mine, done)

    # --- macro: defense only during the rush, the bunker first --------------------------------

    def _plan(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict):
        if self.rush and (act == EXPAND or act not in (TRAIN, BUILD) or uid not in DEFENSE):
            return None
        return super()._plan(act, uid, obs, mine, done, me)

    def _macro(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> None:
        if self.rush and not self._bunker_coming(mine) and any(u["type"] == T.RAX and u["completed"] for u in mine):
            budget = me["minerals"] - sum(v01.COST[ut] for ut, *_ in self.reserved)
            if budget >= T.COST[T.BUNKER][0] - 40 and self.hi.frame >= self.next_bunker:
                self.next_bunker = self.hi.frame + BUNKER_RETRY
                self.task = Task("build", T.BUNKER, "main", data={"near": self.bunker_spot}, started=self.hi.frame)
                self._say("builds a bunker at the ramp")
            return                       # save for the bunker
        if self.rush and self.goal is None:
            self.goal = self._rush_goal(obs, mine, done, me)
            if self.goal is None:
                return
            self._say(f"next: {self.goal['name']} (rush defense)")
        super()._macro(obs, mine, done, me)

    def _rush_goal(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        """The next build against a one-base all-in, in the macro's goal format."""
        goal = lambda act, uid, name: {"act": act, "id": uid, "p": 1.0, "since": self.hi.frame, "name": name}
        coming = lambda ut: any(u["type"] == ut and not u["completed"] for u in mine) or \
            any(r == ut for r, *_ in self.reserved)
        if me["supply_max"] - me["supply_used"] < 3 * T.SUPPLY[T.MARINE] and not coming(T.DEPOT) and \
                self._plan(BUILD, T.DEPOT, obs, mine, done, me) not in (None, WAIT):
            return goal(BUILD, T.DEPOT, "Supply Depot")
        if self._plan(TRAIN, T.MARINE, obs, mine, done, me) not in (None, WAIT):
            return goal(TRAIN, T.MARINE, "Marine")
        if sum(1 for u in mine if u["type"] == T.SCV) < RUSH_SCVS and self._plan(TRAIN, T.SCV, obs, mine, done, me) not in (None, WAIT):
            return goal(TRAIN, T.SCV, "SCV")
        budget = me["minerals"] - sum(v01.COST[ut] for ut, *_ in self.reserved)
        if budget >= RAX_MONEY and sum(1 for u in mine if u["type"] == T.RAX) < RUSH_RAX and not coming(T.RAX):
            return goal(BUILD, T.RAX, "Barracks")
        return None

    def _bunker_coming(self, mine: list[dict]) -> bool:
        return any(u["type"] == T.BUNKER for u in mine) or any(ut == T.BUNKER for ut, *_ in self.reserved) or \
            (self.task is not None and self.task.kind == "build" and self.task.unit_type == T.BUNKER)

    # --- repair -------------------------------------------------------------------------------

    def _pull_workers(self, obs: dict, mine: list[dict]) -> bool:
        return self._repair(obs, mine) or super()._pull_workers(obs, mine)

    def _repair(self, obs: dict, mine: list[dict]) -> bool:
        """A damaged bunker with zerglings near: the 2 nearest free SCVs repair it."""
        hi = self.hi
        if hi.frame < self.next_repair:
            return False
        bunker = next((u for u in mine if u["type"] == T.BUNKER and u["completed"] and u.get("hp_max")
                       and u["hp"] < u["hp_max"]), None)
        if not bunker:
            return False
        fighters = [u for u in obs["units"] if u["owner"] not in (self.slot, 11) and supply_x2(u["type"]) > 0
                    and not is_worker(u["type"]) and not is_building(u["type"])]
        if not any(math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])) < THREAT for u in fighters):
            return False
        repairing = [u for u in mine if u["type"] == T.SCV and u.get("order") == REPAIR_ORDER
                     and u.get("order_target") == bunker["tag"]]
        if len(repairing) >= REPAIRERS:
            return False
        scvs = sorted((u for u in mine if u["type"] == T.SCV and u["completed"] and u["tag"] not in {r["tag"] for r in repairing}),
                      key=lambda u: math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])))[:REPAIRERS - len(repairing)]
        if not scvs:
            return False
        self.next_repair = hi.frame + 24 * 3
        self.task = Task("repair", data={"bunker": bunker["tag"], "x": bunker["x"], "y": bunker["y"],
                                         "scvs": [u["tag"] for u in scvs]}, started=hi.frame)
        return True

    def _run_task(self, obs: dict, mine: list[dict]) -> None:
        t = self.task
        if t.kind != "repair":
            return super()._run_task(obs, mine)
        hi, d = self.hi, t.data
        if hi.frame - t.started > 24 * 3:
            self.task = None
            return
        units = [u for u in mine if u["tag"] in set(d["scvs"])]
        if not units:
            self.task = None
            return
        if t.stage == "start":
            if not screen_of(obs, d["x"], d["y"]) or not all(screen_of(obs, u["x"], u["y"]) for u in units):
                if d.get("camera_moved"):
                    units = [u for u in units if screen_of(obs, u["x"], u["y"])]
                    if not units or not screen_of(obs, d["x"], d["y"]):
                        self.task = None
                        return
                else:
                    d["camera_moved"] = True
                    self._camera_to(int(d["x"]), int(d["y"]))
                    return
            pts = [screen_of(obs, u["x"], u["y"]) for u in units]
            hi.box(max(0, min(p[0] for p in pts) - 12), max(0, min(p[1] for p in pts) - 12),
                   min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + 12), min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + 12))
            t.stage = "order"
            return
        p = screen_of(obs, d["x"], d["y"])
        if p and set(obs["selection"]) & set(d["scvs"]):
            hi.right_click(*p)                   # an SCV's right-click on a damaged building: repair
            self._say(f"{len(set(obs['selection']) & set(d['scvs']))} SCVs repair the bunker")
        self.task = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", help="a .scm/.scx map (or a pre-1.18 replay) for headless play")
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", default="gary_v09.rep")
    ap.add_argument("--command-delay", type=int, default=LIVE_COMMAND_DELAY)
    ap.add_argument("--style", type=int, help="build-style cluster to steer the macro toward")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--live", action="store_true", help="play in a live SC:R client via the bridge")
    ap.add_argument("--pipe", default=r"\\.\pipe\gary_scr")
    ap.add_argument("--profile", default="pro", choices=sorted(v01.PROFILES))
    ap.add_argument("--pov", help="where to save the point-of-view log (live)")
    ap.add_argument("--fight-model", help="fight command model (default: v0.8's)")
    args = ap.parse_args()
    macro, army = MacroModel.load(latest_model()), ArmyModel.load(latest_army_model())
    command = CommandModel.load(args.fight_model or latest_rl_fight_model())
    make = lambda hi, mapinfo: TerranGaryV9(hi, mapinfo, macro, army, command, args.style, verbose=not args.quiet)
    if args.live:
        v01.play_live(args.pipe, args.minutes, args.profile, args.pov or "gary_v09_live.pov.jsonl", make_bot=make)
    else:
        if not args.map:
            ap.error("--map is required unless --live")
        v01.play(args.map, args.minutes, args.seed, args.save, command_delay=args.command_delay,
                 make_bot=make, name="Gary v0.9 (T)")


if __name__ == "__main__":
    main()
