"""Gary v0.4, Terran: v0.3 plus a learned fight model controlling units in skirmishes (#2).

Whenever enemy fighters are near Gary's units, the fight model (gary/policy/fight.py, trained on
pro TvZ skirmishes) looks at every unit around the fight and says what each of Gary's should do:
attack which unit, move where, attack-move, go back to mining, repair (right-click an own unit),
stop, hold, stim, return cargo, or nothing, drawn from the model's probabilities (sampling plays
better than always taking the likeliest). Gary groups units given the same order and carries the
orders out with human hands, one group per turn: camera on the fight, select (drag box, or click
and shift-clicks), then right-click the target, A-click, or the hotkey. Units already doing what
they were told are left alone. This replaces v0.3's scripted worker pull; the army model still
moves the army clusters that aren't in a fight.

    python -m gary.bots.terran_v04 --map path/to/map.scx --minutes 12 --style 1
    python -m gary.bots.terran_v04 --live --style 1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from gary import commands as C
from gary.bots import terran_v01 as v01
from gary.bots.terran_v01 import Task
from gary.bots.terran_v02 import latest_model
from gary.bots.terran_v03 import TerranGaryV3, latest_army_model
from gary.env import LIVE_COMMAND_DELAY
from gary.interface import HumanInterface, screen_of, unit_name
from gary.mapinfo import MapInfo
from gary.policy.army import ArmyModel, is_building, is_worker, supply_x2
from gary.policy.fight import ACTIONS, FIGHT_AROUND, MAX_UNITS, FightModel, snapshot
from gary.policy.macro import MacroModel
from gary.version import announcement

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASK_EVERY = 12                       # frames between fight decisions (half a second)
INTERRUPTIBLE = {"mine", "gas", "hotkey_building"}   # chores a fight may interrupt
RESOURCES = {176, 177, 178, 188, 110}
ACT_IF_NONE_BELOW = 0.2              # act on a unit when the model gives "nothing" less than this
                                     # (with 0.2 Gary acts on about as many units as pros do: 22% vs 27%)
SAME_POINT = 96                      # moves to points this close count as the same order
BOX_MARGIN = 20                      # screen pixels around the group's units when drag-selecting
BUNKER, MARINE = 125, 0
BUNKER_THREAT = 9 * 32               # enemy fighters this close to a bunker: load it
BUNKER_FROM = 10 * 32                # with marines from this far
BUNKER_SLOTS = 4
REPEAT_FRAMES = 48                   # a unit isn't given the same order again within two seconds
IDLE_ORDERS = {2, 3}                 # Guard, PlayerGuard
HOLD_ORDERS = {107, 177}             # HoldPosition, MedicHoldPosition
HARVEST_ORDERS = set(range(79, 91))  # Harvest1 .. ReturnMinerals: already mining or on gas
DEST_NOISE = 96                      # sampling: spread of move destinations (pixels)
PRIORITY = {"attack_unit": 0, "gather": 1, "own_unit": 1, "stim": 1, "hold": 2, "return_cargo": 2,
            "attack_move": 3, "stop": 3, "move": 4}
# Enemy HP is hidden unless the unit is selected (the game's UI); the model learned with true HP,
# so unseen enemies count as full HP.
MAX_HP = {37: 35, 38: 80, 39: 400, 41: 40, 42: 200, 43: 120, 44: 150, 45: 120, 46: 80, 47: 25, 50: 60,
          62: 250, 103: 125, 35: 25, 36: 200, 40: 30}


def latest_fight_model(matchup: str = "TvZ", race: str = "T") -> Path:
    found = sorted((REPO_ROOT / "runs" / "fight").glob(f"{matchup}_{race}_*/model.pt"))
    if not found:
        raise SystemExit(f"no trained fight model in runs/fight (python -m train.fight --race {race})")
    return found[-1]


class TerranGaryV4(TerranGaryV3):
    version = "v0.4"
    own_radius, max_units = FIGHT_AROUND, MAX_UNITS      # the fight snapshot, as the model learned it

    def __init__(self, hi: HumanInterface, mapinfo: MapInfo, model: MacroModel, army_model: ArmyModel,
                 fight_model: FightModel, style: int | None = None, verbose: bool = True):
        super().__init__(hi, mapinfo, model, army_model, style, verbose)
        self.fight_model = fight_model
        self.announce = announcement(self.version, style, {"macro": getattr(model, "path", "?"),
                                                           "army": getattr(army_model, "path", "?"),
                                                           "fight": getattr(fight_model, "path", "?")})
        self.flip = (self.main.center[0] > mapinfo.size[0] / 2, self.main.center[1] > mapinfo.size[1] / 2)
        self.next_fight = 0
        self.fight_center: tuple[float, float] | None = None
        self.told: dict[int, tuple[tuple, int]] = {}    # unit tag -> (order key, frame given)
        self.macro_turn = False      # after each fight command, production gets a turn
        self.loaded: dict[int, set[int]] = {}           # bunker tag -> marines Gary sent into it
        self.next_load = 0
        # sample (the default): draw each unit's action, target and destination from the model's
        # distributions; it plays better in pro scenarios than acting on fixed thresholds
        # (eval/scenarios.py: -207 vs -253). False: the most likely action when "nothing" is
        # unlikely (ACT_IF_NONE_BELOW). rl_log, when a list, gets every decision carried out
        # (reinforcement learning, train/fight_rl.py).
        self.sample = True
        self.rng = np.random.default_rng(0)
        self.rl_log: list[dict] | None = None

    # --- fights ----------------------------------------------------------------------------

    def act(self) -> None:
        """Fights come first: when enemy fighters are near, the fight model is asked before
        anything else, and may interrupt chores (sending workers to mine or gas, hotkeys) that
        are cheap to pick up again. After each fight command production gets one turn, so it goes
        on during fights."""
        hi = self.hi
        if self.macro_turn and self.task is None and not hi.pending:
            self.macro_turn = False              # fights don't starve production: this turn goes
            obs = hi.observe()                   # to buildings and production, not to chores
            mine = [u for u in obs["units"] if u["owner"] == self.slot]   # (refilling gas, mining)
            self._release_reservations(mine)
            if not self._resume_construction(mine):
                self._macro(obs, mine, [u for u in mine if u["completed"]], obs["me"])
            return
        if not hi.pending and self.setup_done and hi.frame >= self.next_fight and                 (self.task is None or self.task.kind in INTERRUPTIBLE):
            obs = hi.observe()
            self.army_tracker.remember(obs)
            mine = [u for u in obs["units"] if u["owner"] == self.slot]
            task = self.task
            self.task = None
            if self._pull_workers(obs, mine) or self._bunker_load(obs, mine) or self._fight(obs, mine):
                return
            self.task = task
        super().act()

    def _worker_defense(self, obs: dict, mine: list[dict]) -> bool:
        return False                             # the fight model decides what workers do (_army)

    def _pull_workers(self, obs: dict, mine: list[dict]) -> bool:
        """Later versions: a worker pull decided before the fight model (v0.7). True if Gary acted."""
        return False

    # --- bunkers ---------------------------------------------------------------------------

    def _bunker_load(self, obs: dict, mine: list[dict]) -> bool:
        """Zerg fighters coming at a finished bunker that isn't full: put the nearest marines in it
        (right-click the bunker), as players do. Marines inside are hidden, so Gary counts as loaded
        the marines it sent that are inside (not seen) or still on their way. True if Gary acted."""
        hi = self.hi
        if hi.frame < self.next_load:
            return False
        seen = {u["tag"]: u for u in mine}
        enemy = [u for u in obs["units"] if u["owner"] not in (self.slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        for b in mine:
            if b["type"] != BUNKER or not b["completed"]:
                continue
            if not self._bunker_wanted(b, enemy):
                continue
            sent = self.loaded.setdefault(b["tag"], set())
            sent &= {t for t in sent if t not in seen or seen[t].get("order_target") == b["tag"]}
            free = BUNKER_SLOTS - len(sent)
            marines = sorted((u for u in mine if u["type"] == MARINE and u["completed"] and u["tag"] not in sent
                              and math.dist((u["x"], u["y"]), (b["x"], b["y"])) < self.bunker_from),
                             key=lambda u: math.dist((u["x"], u["y"]), (b["x"], b["y"])))[:free]
            if free <= 0 or not marines:
                continue
            self.next_load = hi.frame + 24 * 2
            self.task = Task("load", data={"bunker": b["tag"], "x": b["x"], "y": b["y"],
                                           "marines": [u["tag"] for u in marines]}, started=hi.frame)
            return True
        return False

    bunker_from = BUNKER_FROM

    def _bunker_wanted(self, b: dict, enemy: list[dict]) -> bool:
        """Whether marines should be in this bunker now: when enemy fighters come near it."""
        return any(math.dist((e["x"], e["y"]), (b["x"], b["y"])) < BUNKER_THREAT for e in enemy)

    def _task_load(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Drag-select the marines (camera on them first if needed), camera on the bunker if it isn't
        on screen, right-click the bunker."""
        hi, d = self.hi, t.data
        units = [u for u in mine if u["tag"] in set(d["marines"])]
        if not units:
            self.task = None
            return
        if t.stage == "start":
            on_screen = [u for u in units if screen_of(obs, u["x"], u["y"])]
            if not on_screen:
                if d.get("camera_moved"):
                    self.task = None
                    return
                d["camera_moved"] = True             # on the marines nearest the bunker
                self._camera_to(int(units[0]["x"]), int(units[0]["y"]))
                return
            pts = [screen_of(obs, u["x"], u["y"]) for u in on_screen]
            m = BOX_MARGIN
            hi.box(max(0, min(p[0] for p in pts) - m), max(0, min(p[1] for p in pts) - m),
                   min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + m),
                   min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + m))
            t.stage = "order"
            return
        chosen = set(obs["selection"]) & {u["tag"] for u in units}
        p = screen_of(obs, d["x"], d["y"])
        if chosen and not p and not d.get("to_bunker"):
            d["to_bunker"] = True
            self._camera_to(int(d["x"]), int(d["y"]))
            return
        if chosen and p:
            hi.right_click(*p)
            self.loaded.setdefault(d["bunker"], set()).update(chosen)
            self._say(f"loads {len(chosen)} marine{'s' * (len(chosen) != 1)} into the bunker")
        self.task = None

    def _snapshot(self, obs: dict):
        """The fight around Gary's units as the fight model sees it (rows, tags, center, units by
        tag), or None when there's no fight; asked at most every ASK_EVERY frames."""
        hi = self.hi
        if hi.frame < self.next_fight:
            return None
        self.next_fight = hi.frame + ASK_EVERY
        units = [dict(u) for u in obs["units"]]
        for u in units:                          # hidden enemy HP: assume full
            if u["owner"] not in (self.slot, 11) and u.get("hp") is None:
                u["hp"], u["shields"] = MAX_HP.get(u["type"], 100), 0
        snap = snapshot({**obs, "units": units}, self.slot, self.flip, supply_x2, is_worker, is_building,
                        self.own_radius, self.max_units)
        if snap is None:
            self.fight_center = None
            return None
        rows, tags, center = snap
        self.fight_center = center
        return rows, tags, center, {u["tag"]: u for u in units}

    def _fight(self, obs: dict, mine: list[dict]) -> bool:
        hi = self.hi
        snap = self._snapshot(obs)
        if snap is None:
            return False
        rows, tags, center, by_tag = snap
        acts, targets, dests = self.fight_model.predict(rows, obs["frame"] * 42 / 1000)
        groups: dict[tuple, list[int]] = {}
        points: dict[str, list[tuple[int, int]]] = {"move": [], "attack_move": []}
        picked: dict[int, tuple[int, int, int, tuple[float, float]]] = {}   # tag -> (row, action, target, dest)
        for i, tag in enumerate(tags):
            if rows[i][1] != 1:
                continue
            if self.sample:
                p = acts[i].astype(np.float64)
                ai = int(self.rng.choice(len(p), p=p / p.sum()))
                if ai == 0:
                    continue
            elif acts[i][0] >= ACT_IF_NONE_BELOW and acts[i][0] >= acts[i][1:].max():
                continue                         # "nothing" is likely and the model's first choice
            else:                                # act when the model doesn't expect "nothing",
                ai = int(acts[i][1:].argmax()) + 1   # with its best real action
            a = ACTIONS[ai]
            j, dest = -1, (0.0, 0.0)
            if a == "other":
                continue
            u = by_tag[tag]
            if self._already(a, u):
                continue
            if a in ("attack_unit", "gather", "own_unit"):
                if self.sample:
                    p = targets[i].astype(np.float64)
                    j = int(self.rng.choice(len(p), p=p / p.sum()))
                else:
                    j = int(targets[i].argmax())
                t = tags[j]
                if t == tag or u.get("order_target") == t:
                    continue                     # already on it
                if a == "gather" and rows[j][0] not in RESOURCES:
                    continue                     # gathering means minerals, a geyser or a refinery
                key = (a, t)
            elif a in ("move", "attack_move"):
                # like a player: the whole group goes to one point (the median of where the
                # model sends each unit), not every unit to its own
                dx, dy = dests[i] + (self.rng.normal(0, DEST_NOISE, 2) if self.sample else 0)
                dest = (float(dx), float(dy))
                dx, dy = (-dx if self.flip[0] else dx), (-dy if self.flip[1] else dy)
                points[a].append((int(u["x"] + dx), int(u["y"] + dy)))
                key = (a,)
            else:
                key = (a,)
            groups.setdefault(key, []).append(tag)
            picked[tag] = (i, ai, j, dest)
        for a, pts in points.items():
            if (a,) in groups:
                groups[(a, int(np.median([p[0] for p in pts])), int(np.median([p[1] for p in pts])))] = groups.pop((a,))
        for key in list(groups):                 # drop units just given the same order
            groups[key] = [t for t in groups[key] if not self._just_told(t, key)]
            if not groups[key]:
                del groups[key]
        if not groups:
            return False
        key = min(groups, key=lambda k: (PRIORITY.get(k[0], 5), -len(groups[k])))
        for tag in groups[key][:12]:
            self.told[tag] = (key, hi.frame)
        if self.rl_log is not None:
            done = [picked[t] for t in groups[key][:12]]
            self.rl_log.append({"rows": rows, "time": obs["frame"] * 42 / 1000,
                                "unit": [d[0] for d in done], "act": [d[1] for d in done],
                                "target": [d[2] for d in done], "dest": [d[3] for d in done]})
        self.task = Task("fight_cmd", data={"key": key, "tags": groups[key][:12], "center": center},
                         started=hi.frame)
        return True

    def _just_told(self, tag: int, key: tuple) -> bool:
        last = self.told.get(tag)
        if not last or self.hi.frame - last[1] >= REPEAT_FRAMES or last[0][0] != key[0]:
            return False
        if len(key) == 3:                        # moves: the same if to about the same point
            return math.dist(last[0][1:], key[1:]) < SAME_POINT
        return last[0] == key

    @staticmethod
    def _already(a: str, u: dict) -> bool:
        """Whether the unit is already doing what the model says (the order would change nothing)."""
        order = u.get("order")
        return (a == "gather" and order in HARVEST_ORDERS) or (a == "hold" and order in HOLD_ORDERS) or             (a == "stop" and order in IDLE_ORDERS) or (a == "return_cargo" and not u.get("carrying"))

    def _army(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        if self._fight(obs, mine):
            return True
        if self.fight_center is not None:        # the army model leaves units in a fight alone
            cx, cy = self.fight_center
            done = [u for u in done if math.dist((u["x"], u["y"]), (cx, cy)) > 12 * 32]
        return super()._army(obs, mine, done)

    def _run_task(self, obs: dict, mine: list[dict]) -> None:
        t = self.task
        if t.kind == "load":
            if self.hi.frame - t.started > 24 * 3:
                self.task = None
                return
            return self._task_load(obs, mine, t)
        if t.kind == "fight_cmd":
            if self.hi.frame - t.started > 24 * 3:
                self.task = None
            else:
                self._task_fight(obs, mine, t)
            self.macro_turn = self.task is None
            return
        return super()._run_task(obs, mine)

    def _task_fight(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Select the group (or keep the selection if it already is the group), with the camera on
        the group, then give the order."""
        hi, d = self.hi, t.data
        units = [u for u in mine if u["tag"] in set(d["tags"])]
        if not units:
            self.task = None
            return
        sel = set(obs["selection"])
        want = {u["tag"] for u in units}
        if t.stage == "start":
            # the group is already selected (kiting the same units): straight to the order
            if (want <= sel and len(sel) <= len(want) + 2) or (sel and sel <= want and len(sel) >= 0.7 * len(want)):
                t.stage = "order"
            else:
                on_screen = [u for u in units if screen_of(obs, u["x"], u["y"])]
                if len(on_screen) < len(units) and not d.get("camera_moved"):
                    d["camera_moved"] = True             # once: then take whoever is on screen
                    self._camera_to(int(sum(u["x"] for u in units) / len(units)),
                                    int(sum(u["y"] for u in units) / len(units)))
                    return
                if not on_screen:
                    self.task = None
                    return
                # a generous box: units move while Gary reacts (what it sees is 0.3 s old)
                pts = [screen_of(obs, u["x"], u["y"]) for u in on_screen]
                m = BOX_MARGIN
                hi.box(max(0, min(p[0] for p in pts) - m), max(0, min(p[1] for p in pts) - m),
                       min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + m),
                       min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + m))
                t.stage = "order"
                return
        if not sel & want:
            self.task = None
            return
        want = sel & want
        key = d["key"]
        a = key[0]
        if a in ("attack_unit", "gather", "own_unit"):
            target = next((u for u in obs["units"] if u["tag"] == key[1]), None)
            p = target and screen_of(obs, target["x"], target["y"])
            if p:
                hi.right_click(*p)
                self._say(f"fight: {len(sel & want)} {self._names(units)} -> {a.replace('_', ' ')} "
                          f"{unit_name(target['type'])}")
            elif target and a == "attack_unit":   # off screen: attack-move there on the minimap
                hi.minimap_command(C.ORDER_ATTACK_MOVE, target["x"], target["y"])
                self._say(f"fight: {len(want)} {self._names(units)} -> attack move (minimap)")
            elif target:
                hi.minimap_right_click(target["x"], target["y"])
                self._say(f"fight: {len(want)} {self._names(units)} -> move (minimap)")
        elif a in ("move", "attack_move"):
            p = screen_of(obs, key[1], key[2])
            if p and a == "move":
                hi.right_click(*p)
            elif p:
                hi.order_click(C.ORDER_ATTACK_MOVE, *p)
            elif a == "move":                    # off screen: on the minimap, like a player would
                hi.minimap_right_click(key[1], key[2])
            else:
                hi.minimap_command(C.ORDER_ATTACK_MOVE, key[1], key[2])
            self._say(f"fight: {len(sel & want)} {self._names(units)} -> {a.replace('_', ' ')}")
        else:
            {"stop": hi.stop, "hold": hi.hold, "stim": hi.stim, "return_cargo": hi.return_cargo}[a]()
            self._say(f"fight: {len(sel & want)} {self._names(units)} -> {a.replace('_', ' ')}")
        self.task = None

    def _camera_to(self, x: int, y: int) -> None:
        """The camera onto a map point: a saved screen (F2-F4) if one shows it, a key press, else
        the minimap."""
        hi = self.hi
        vw, vh = hi.p.viewport
        for n, (lx, ly) in hi.camera_locations.items():
            if lx + 32 <= x <= lx + vw - 32 and ly + 32 <= y <= ly + vh - 32:
                if hi.camera != (lx, ly):
                    hi.camera_location(n)
                    return
        hi.camera_minimap(x, y)

    @staticmethod
    def _names(units: list[dict]) -> str:
        kinds = {unit_name(u["type"]) for u in units}
        return "/".join(sorted(kinds))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", help="a .scm/.scx map (or a pre-1.18 replay) for headless play")
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", default="gary_v04.rep")
    ap.add_argument("--command-delay", type=int, default=LIVE_COMMAND_DELAY)
    ap.add_argument("--style", type=int, help="build-style cluster to steer the macro toward")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--live", action="store_true", help="play in a live SC:R client via the bridge")
    ap.add_argument("--pipe", default=r"\\.\pipe\gary_scr")
    ap.add_argument("--profile", default="pro", choices=sorted(v01.PROFILES))
    ap.add_argument("--pov", help="where to save the point-of-view log (live)")
    args = ap.parse_args()
    macro, army = MacroModel.load(latest_model()), ArmyModel.load(latest_army_model())
    fight = FightModel.load(latest_fight_model())
    make = lambda hi, mapinfo: TerranGaryV4(hi, mapinfo, macro, army, fight, args.style, verbose=not args.quiet)
    if args.live:
        v01.play_live(args.pipe, args.minutes, args.profile, args.pov or "gary_v04_live.pov.jsonl", make_bot=make)
    else:
        if not args.map:
            ap.error("--map is required unless --live")
        v01.play(args.map, args.minutes, args.seed, args.save, command_delay=args.command_delay,
                 make_bot=make, name="Gary v0.4 (T)")


if __name__ == "__main__":
    main()
