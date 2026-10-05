"""Gary v0.9, Terran: v0.8 that holds early pools (#2).

v0.8 never noticed an early Spawning Pool: its learned macro kept planning an expansion, gas and
tech while the zerglings came, and it met the first wave with three marines and no bunker. v0.9
does what a player does against an early pool:

  detect   a Spawning Pool seen before 3:00 while only one hatchery is known, zerglings before
           4:00, or 6 at once before 5:00 (from the scout or at home)
  bunker   on the main's level by the top of the main ramp (gary/terrain.py; only where the map
           analysis is reliable, else between the main and its natural), Barracks rallied just
           behind it
  macro    until the bunker stands with 4 marines and no zerglings are at home: the standard answer
           to a one-base all-in, not the learned macro (whose likely picks, expansion, gas and
           tech, are what it mustn't do now, and the rest fall below its threshold: v0.9 built
           nothing for minutes): depots ahead of production, marines and SCVs nonstop, a second
           Barracks on spare money. Then the learned macro again, gas and tech included, only no
           expansion while Gary holds
  hold     until Gary's army is half again the biggest army the Zerg showed: no expansion, the
           army model doesn't send the army anywhere (the fight model still fights with it), and
           marines go into the bunker as soon as it's done, from anywhere in the main, not when the
           zerglings are already there (v0.4's rule). Leaving the bunker for a counter-attack is
           what lost the defended games
  reinforce  a marine out of a Barracks while zerglings are at the base gets an attack-move to
           the fight: walking the rally path (a plain move) it would die without firing
  repair   2 SCVs repair the bunker when it's damaged and zerglings are near

Everything else is v0.8.

    python -m gary.bots.terran_v09 --map path/to/map.scx --minutes 12 --style 1
    python -m gary.bots.terran_v09 --live --style 1
"""

from __future__ import annotations

import argparse
import math

from gary import commands as C
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
RUSH_MARINES = 4                     # in or at the finished bunker: the rush build can end...
CALM = 20 * 24                       # ...once no zerg fighter was at home for this long
HOLD_ARMY, HOLD_RATIO = 6, 1.5       # Gary holds until its army supply is this, and this many times
                                     # the biggest zerg army seen at once
HOME = 12 * 32                       # zerg fighters this close to a Command Center or bunker: at home
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
        self.rush = False                # an early pool was seen: the rush build
        self.hold = False                # ...and the army stays home
        self.rush_over = False
        self.hatcheries_seen: set[int] = set()
        self.most_enemy = 0.0            # the biggest zerg army (supply) seen at once
        self.calm_since = 0              # since when no zerg fighter was at home
        self.known_army: set[int] | None = None   # fighters Gary has seen (for _reinforce)
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
            fighters = self._fighters(obs)
            self.most_enemy = max(self.most_enemy, sum(supply_x2(u["type"]) for u in fighters) / 2)
            if not self.rush and not self.hold and ((hi.frame <= POOL_BY and any(u["type"] == POOL for u in enemy)
                                                     and len(self.hatcheries_seen) <= 1) or
                                                    (hi.frame <= LINGS_BY and lings) or
                                                    (hi.frame <= MANY_LINGS_BY and lings >= MANY_LINGS)):
                self.rush = self.hold = True
                # marines gather a few tiles behind the bunker, toward the main (not on its spot,
                # where they'd keep the SCV from placing it)
                bx, by = self.bunker_spot
                mx, my = self.main.center
                d = math.dist((bx, by), (mx, my)) or 1
                self.rally = (int(bx + (mx - bx) / d * 4 * 32), int(by + (my - by) / d * 4 * 32))
                self._say("early pool: bunker at the ramp, marines first, no expansion or tech yet")
            if self.hold:
                mine = [u for u in obs["units"] if u["owner"] == self.slot]
                if self._near_home(mine, fighters):
                    self.calm_since = hi.frame
                bunker = any(u["type"] == T.BUNKER and u["completed"] for u in mine)
                tags = {u["tag"] for u in mine}
                inside = set().union(set(), *(m for b, m in self.loaded.items() if b in tags)) - tags
                marines = sum(1 for u in mine if u["type"] == T.MARINE) + len(inside)   # the ones in the bunker aren't seen
                army = sum(supply_x2(u["type"]) for u in mine if u["type"] in self.army_types) / 2 + len(inside)
                if self.rush and bunker and marines >= RUSH_MARINES and hi.frame - self.calm_since >= CALM:
                    self.rush = False
                    self._say(f"bunker up with {marines} marines, no zerglings at home: gas and tech again, still home")
                if not self.rush and army >= max(HOLD_ARMY, HOLD_RATIO * self.most_enemy):
                    self.hold, self.rush_over = False, True
                    self._say(f"the early pool is held: army {army:.0f} vs the zerg's {self.most_enemy:.0f} at most, back to the usual plan")
        super().act()

    def _fighters(self, obs: dict) -> list[dict]:
        return [u for u in obs["units"] if u["owner"] not in (self.slot, 11) and supply_x2(u["type"]) > 0
                and not is_worker(u["type"]) and not is_building(u["type"])]

    def _near_home(self, mine: list[dict], fighters: list[dict]) -> list[dict]:
        """The zerg fighters at Gary's base: near a Command Center or a bunker."""
        home = [(u["x"], u["y"]) for u in mine if u["type"] in (T.CC, T.BUNKER)]
        return [e for e in fighters if any(math.dist((e["x"], e["y"]), h) < HOME for h in home)]

    # --- the army stays home during the rush, in the bunker ------------------------------------

    @property
    def bunker_from(self) -> int:
        return LOAD_FROM if self.hold else super().bunker_from

    def _bunker_wanted(self, b: dict, enemy: list[dict]) -> bool:
        return self.hold or super()._bunker_wanted(b, enemy)

    def _army(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        if self.hold:                    # the army model's moves leave the bunker empty: none now
            done = [u for u in done if u["type"] not in self.army_types]
        return super()._army(obs, mine, done)

    # --- reinforce ----------------------------------------------------------------------------

    def _fight(self, obs: dict, mine: list[dict]) -> bool:
        return self._reinforce(obs, mine) or super()._fight(obs, mine)

    def _reinforce(self, obs: dict, mine: list[dict]) -> bool:
        """Fighters just out of a Barracks while zerg fighters are at the base: attack-move them to
        the fight (on the minimap). The rally is a plain move, and a moving marine doesn't shoot."""
        army = [u for u in mine if u["type"] in self.army_types and u["completed"]]
        if self.known_army is None:
            self.known_army = {u["tag"] for u in army}
        new = [u for u in army if u["tag"] not in self.known_army]
        self.known_army |= {u["tag"] for u in army}
        if not new:
            return False
        at_home = self._near_home(mine, self._fighters(obs))
        if not at_home:
            return False
        cx, cy = sum(u["x"] for u in new) / len(new), sum(u["y"] for u in new) / len(new)
        target = min(at_home, key=lambda e: math.dist((e["x"], e["y"]), (cx, cy)))
        self.task = Task("reinforce", data={"tags": [u["tag"] for u in new], "x": target["x"], "y": target["y"]},
                         started=self.hi.frame)
        return True

    # --- macro: defense only during the rush, the bunker first --------------------------------

    def _plan(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict):
        if self.rush and (act == EXPAND or act not in (TRAIN, BUILD) or uid not in DEFENSE):
            return None
        if self.hold and act == EXPAND:
            return None
        return super()._plan(act, uid, obs, mine, done, me)

    def _macro(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> None:
        if self.rush and not self._bunker_coming(mine) and any(u["type"] == T.RAX and u["completed"] for u in mine):
            budget = me["minerals"] - sum(v01.COST[ut] for ut, *_ in self.reserved)
            if budget >= T.COST[T.BUNKER][0] - 40 and self.hi.frame >= self.next_bunker:
                self.next_bunker = self.hi.frame + BUNKER_RETRY
                self.task = Task("build", T.BUNKER, "main", data={"near": self.bunker_spot, "level_at": self.main.center},
                                 started=self.hi.frame)
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
        doable = lambda act, uid: self._plan(act, uid, obs, mine, done, me) not in (None, WAIT)
        # a depot takes longer than a marine or an SCV: start it while each producer has 2 left
        producers = sum(1 for u in done if u["type"] in (T.CC, T.RAX))
        if me["supply_max"] - me["supply_used"] < 2 * producers and me["supply_max"] < 200 and \
                not coming(T.DEPOT) and doable(BUILD, T.DEPOT):
            return goal(BUILD, T.DEPOT, "Supply Depot")
        if doable(TRAIN, T.MARINE):
            return goal(TRAIN, T.MARINE, "Marine")
        if sum(1 for u in mine if u["type"] == T.SCV) < RUSH_SCVS and doable(TRAIN, T.SCV):
            return goal(TRAIN, T.SCV, "SCV")
        budget = me["minerals"] - sum(v01.COST[ut] for ut, *_ in self.reserved)
        if budget >= RAX_MONEY and sum(1 for u in mine if u["type"] == T.RAX) < RUSH_RAX and not coming(T.RAX) and \
                doable(BUILD, T.RAX):
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
        if t.kind == "reinforce":
            return self._task_reinforce(obs, mine, t)
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


    def _task_reinforce(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Camera on the new fighters, drag-select them, attack-move on the minimap to the fight."""
        hi, d = self.hi, t.data
        units = [u for u in mine if u["tag"] in set(d["tags"])]
        if hi.frame - t.started > 24 * 3 or not units:
            self.task = None
            return
        if t.stage == "start":
            on_screen = [u for u in units if screen_of(obs, u["x"], u["y"])]
            if not on_screen:
                if d.get("camera_moved"):
                    self.task = None
                    return
                d["camera_moved"] = True
                self._camera_to(int(units[0]["x"]), int(units[0]["y"]))
                return
            pts = [screen_of(obs, u["x"], u["y"]) for u in on_screen]
            hi.box(max(0, min(p[0] for p in pts) - 12), max(0, min(p[1] for p in pts) - 12),
                   min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + 12), min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + 12))
            t.stage = "order"
            return
        chosen = set(obs["selection"]) & set(d["tags"])
        if chosen:
            hi.minimap_command(C.ORDER_ATTACK_MOVE, int(d["x"]), int(d["y"]))
            self._say(f"reinforce: {len(chosen)} new {'units' if len(chosen) > 1 else 'unit'} attack-move to the fight")
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
