"""The fight (layer L4, docs/CONTROL.md): what Gary's units do when enemy fighters are near.

A fight layer is a model plus rules. Models (one command per half second, carried out with human
hands: camera on the fight, drag box or the current selection, then right-click / A-click / key):

  per_unit  the fight model (gary/policy/fight.py; Gary v0.4): what each unit should do, regrouped
            into one command per order
  command   the command model (gary/policy/fight_cmd.py; v0.5+): one command at a time to a
            selection, as a player gives it; with memory of its recent commands (v0.6+), trained
            further by reinforcement learning (v0.8+). Options: a guard rail that drops its plain
            moves and stops (v0.7), and keeping workers for the rules (v0.7+)

Rules (scripted reflexes), run before the model ("before") or before it in the army's turn too
("in_fight"):

  pull       enemy fighters at a base outnumber the army there: about two SCVs per attacker
             attack-move to them, back to mining after 5 s of calm (v0.3's rule; v0.7+)
  load       zerg fighters near a finished bunker that isn't full: the nearest marines go in
             (v0.4+); the posture can ask for it before the enemy is there (v0.9)
  repair     a damaged bunker with zerglings near: the 2 nearest SCVs repair it (v0.9)
  reinforce  fighters just out of a Barracks while zerg fighters are at the base attack-move to
             the fight: the rally is a plain move, and a moving marine doesn't shoot (v0.9)
"""

from __future__ import annotations

import math

import numpy as np

from gary import commands as C
from gary import terran as T
from gary.agent.gary import Gary, Task
from gary.interface import screen_of, unit_name
from gary.policy.army import is_building, is_worker, supply_x2
from gary.policy.fight import ACTIONS, FIGHT_AROUND, snapshot
from gary.policy.fight import MAX_UNITS as PER_UNIT_MAX
from gary.policy.fight_cmd import MAX_UNITS as COMMAND_MAX, MOVES, OWN_RADIUS, POINTER
from gary.policy.fight_memory import history_inputs

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
LOAD_FROM = 40 * 32                  # ...or this far, when the posture wants bunkers loaded early
BUNKER_SLOTS = 4
REPEAT_FRAMES = 48                   # a unit isn't given the same order again within two seconds
IDLE_ORDERS = {2, 3}                 # Guard, PlayerGuard
HOLD_ORDERS = {107, 177}             # HoldPosition, MedicHoldPosition
HARVEST_ORDERS = set(range(79, 91))  # Harvest1 .. ReturnMinerals: already mining or on gas
DEST_NOISE = 96                      # sampling: spread of move destinations (pixels)
PRIORITY = {"attack_unit": 0, "gather": 1, "own_unit": 1, "stim": 1, "hold": 2, "return_cargo": 2,
            "attack_move": 3, "stop": 3, "move": 4}
# Enemy HP is hidden unless the unit is selected (the game's UI); the models learned with true HP,
# so unseen enemies count as full HP.
MAX_HP = {37: 35, 38: 80, 39: 400, 41: 40, 42: 200, 43: 120, 44: 150, 45: 120, 46: 80, 47: 25, 50: 60,
          62: 250, 103: 125, 35: 25, 36: 200, 40: 30}
THREAT_PX = 10 * 32                  # enemy fighters this close to a Command Center threaten its workers
CALM_FRAMES = 24 * 5                 # pulled workers go back to mining after 5 s without a threat
REPAIR_ORDER = 34
REPAIRERS = 2
REPAIR_THREAT = 9 * 32               # zerglings this close to the bunker: repair it
HOME = 12 * 32                       # zerg fighters this close to a Command Center or bunker: at home


def enemy_fighters(obs: dict, slot: int) -> list[dict]:
    return [u for u in obs["units"] if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
            and not is_worker(u["type"]) and not is_building(u["type"])]


def near_home(mine: list[dict], fighters: list[dict]) -> list[dict]:
    """The enemy fighters at Gary's base: near a Command Center or a bunker."""
    home = [(u["x"], u["y"]) for u in mine if u["type"] in (T.CC, T.BUNKER)]
    return [e for e in fighters if any(math.dist((e["x"], e["y"]), h) < HOME for h in home)]


def box_around(hi, pts: list[tuple[int, int]], m: int) -> None:
    hi.box(max(0, min(p[0] for p in pts) - m), max(0, min(p[1] for p in pts) - m),
           min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + m), min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + m))


class FightLayer:
    def __init__(self, g: Gary, model, kind: str, before: list = (), in_fight: list = (),
                 guard_rail: bool = False, workers: bool = True, seed: int = 0):
        self.g = g
        self.model = model
        self.kind = kind                         # "per_unit" or "command"
        self.before = list(before)               # rules asked first in a fight
        self.in_fight = list(in_fight)           # rules asked right before the model
        self.guard_rail = guard_rail             # drop the model's plain moves and stops (v0.7)
        self.workers = workers                   # False: workers belong to the rules
        self.own_radius, self.max_units = (FIGHT_AROUND, PER_UNIT_MAX) if kind == "per_unit" else (OWN_RADIUS, COMMAND_MAX)
        self.next_fight = 0
        self.center: tuple[float, float] | None = None
        self.told: dict[int, tuple[tuple, int]] = {}    # unit tag -> (order key, frame given)
        # sample (the default): draw each command from the model's distributions; it plays better
        # in pro scenarios than acting on fixed thresholds (eval/scenarios.py: -207 vs -253). False:
        # the most likely action when "nothing" is unlikely (ACT_IF_NONE_BELOW). rl_log, when a
        # list, gets every decision (reinforcement learning, train/fight_cmd_rl.py).
        self.sample = True
        self.temperature = 1.0                   # < 1: draw commands closer to the model's likeliest
        self.rng = np.random.default_rng(seed)
        self.rl_log: list[dict] | None = None
        g.task_runners["fight_cmd"] = self._run_fight_cmd

    def block(self, obs: dict, mine: list[dict]) -> bool:
        """The fight's own turn: the rules, then the model. True if Gary acted."""
        for rule in self.before:
            if rule.step(obs, mine):
                return True
        return self.decide(obs, mine)

    def decide(self, obs: dict, mine: list[dict]) -> bool:
        for rule in self.in_fight:
            if rule.step(obs, mine):
                return True
        return self._per_unit(obs, mine) if self.kind == "per_unit" else self._command(obs, mine)

    # --- the models -----------------------------------------------------------------------------

    def _snapshot(self, obs: dict):
        """The fight around Gary's units as the model sees it (rows, tags, center, units by tag),
        or None when there's no fight; asked at most every ASK_EVERY frames."""
        g = self.g
        if g.hi.frame < self.next_fight:
            return None
        self.next_fight = g.hi.frame + ASK_EVERY
        units = [dict(u) for u in obs["units"]]
        for u in units:                          # hidden enemy HP: assume full
            if u["owner"] not in (g.slot, 11) and u.get("hp") is None:
                u["hp"], u["shields"] = MAX_HP.get(u["type"], 100), 0
        snap = snapshot({**obs, "units": units}, g.slot, g.flip, supply_x2, is_worker, is_building,
                        self.own_radius, self.max_units)
        if snap is None:
            self.center = None
            return None
        rows, tags, center = snap
        self.center = center
        return rows, tags, center, {u["tag"]: u for u in units}

    def _per_unit(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        snap = self._snapshot(obs)
        if snap is None:
            return False
        rows, tags, center, by_tag = snap
        acts, targets, dests = self.model.predict(rows, obs["frame"] * 42 / 1000)
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
                dx, dy = (-dx if g.flip[0] else dx), (-dy if g.flip[1] else dy)
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
        g.task = Task("fight_cmd", data={"key": key, "tags": groups[key][:12], "center": center},
                      started=hi.frame)
        return True

    def _hand_state(self, obs: dict, center) -> np.ndarray:
        """What Gary's hands have to give (HANDS_FEATURES): APM tokens left, production waiting,
        time since the last macro action, camera on the fight."""
        g = self.g
        hi = g.hi
        me = obs.get("me") or {}
        idle = sum(1 for u in obs["units"] if u["owner"] == g.slot and u["type"] in g.production_types
                   and u["completed"] and not u.get("queue"))
        waiting = min(idle, 4) / 4 if me.get("minerals", 0) >= 50 else 0.0
        since = min(1.0, (hi.frame - hi.last_macro_frame) * 42 / 1000 / 10)
        on_fight = 1.0 if screen_of(obs, *center) else 0.0
        return np.array([hi.tokens / hi.p.apm_capacity, waiting, since, on_fight], np.float32)

    def _command(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        snap = self._snapshot(obs)
        if snap is None:
            return False
        rows, tags, center, by_tag = snap
        hist = None
        if self.model.memory:                    # what Gary's hands did in the last seconds
            hist = history_inputs(tags, rows, center, g.flip, obs["frame"], g.hi.commands)
        hands = self._hand_state(obs, center) if getattr(self.model, "hands", False) else None
        cmd = self.model.decide(rows, obs["frame"] * 42 / 1000, self.rng, self.temperature, hist, hands)
        if self.rl_log is not None:              # every decision, "nothing" too (train/fight_cmd_rl.py)
            self.rl_log.append({"rows": rows, "time": obs["frame"] * 42 / 1000, "hist": hist, "hands": hands,
                                **getattr(self.model, "last", {"type": 0, "select": [], "target": -1, "cell": -1})})
        if cmd is None:
            return False
        a = ACTIONS[cmd["type"]]
        if a == "other" or (self.guard_rail and a in ("move", "stop")):
            return False
        if a in POINTER and cmd["target"] < 0:
            return False
        if cmd["type"] in POINTER:
            t = tags[cmd["target"]]
            if a == "gather" and rows[cmd["target"]][0] not in RESOURCES:
                return False                     # gathering means minerals, a geyser or a refinery
            key = (a, t)
        elif cmd["type"] in MOVES:
            dx, dy = cmd["dest"]
            dx, dy = (-dx if g.flip[0] else dx), (-dy if g.flip[1] else dy)
            key = (a, int(center[0] + dx), int(center[1] + dy))
        else:
            key = (a,)
        chosen = [tags[i] for i in cmd["select"] if self.workers or by_tag[tags[i]]["type"] != T.SCV]
        chosen = [t for t in chosen if not self._already(a, by_tag[t]) and not self._just_told(t, key)
                  and not (len(key) == 2 and by_tag[t].get("order_target") == key[1])]
        if not chosen:
            return False
        for t in chosen:
            self.told[t] = (key, g.hi.frame)
        g.task = Task("fight_cmd", data={"key": key, "tags": chosen, "center": center}, started=g.hi.frame)
        if self.rl_log:
            self.rl_log[-1]["executed"] = True   # (decisions Gary didn't carry out cost no action)
        return True

    def _just_told(self, tag: int, key: tuple) -> bool:
        last = self.told.get(tag)
        if not last or self.g.hi.frame - last[1] >= REPEAT_FRAMES or last[0][0] != key[0]:
            return False
        if len(key) == 3:                        # moves: the same if to about the same point
            return math.dist(last[0][1:], key[1:]) < SAME_POINT
        return last[0] == key

    @staticmethod
    def _already(a: str, u: dict) -> bool:
        """Whether the unit is already doing what the model says (the order would change nothing)."""
        order = u.get("order")
        return (a == "gather" and order in HARVEST_ORDERS) or (a == "hold" and order in HOLD_ORDERS) or \
            (a == "stop" and order in IDLE_ORDERS) or (a == "return_cargo" and not u.get("carrying"))

    # --- carrying a command out ---------------------------------------------------------------

    def _run_fight_cmd(self, obs: dict, mine: list[dict], t: Task) -> None:
        g = self.g
        if g.hi.frame - t.started > 24 * 3:
            g.task = None
        else:
            self._task_fight(obs, mine, t)
        g.macro_turn = g.task is None            # after each fight command, production gets a turn

    def _task_fight(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Select the group (or keep the selection if it already is the group), with the camera on
        the group, then give the order."""
        g, hi, d = self.g, self.g.hi, t.data
        units = [u for u in mine if u["tag"] in set(d["tags"])]
        if not units:
            g.task = None
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
                    g.camera_to(int(sum(u["x"] for u in units) / len(units)),
                                int(sum(u["y"] for u in units) / len(units)))
                    return
                if not on_screen:
                    g.task = None
                    return
                # a generous box: units move while Gary reacts (what it sees is 0.3 s old)
                box_around(hi, [screen_of(obs, u["x"], u["y"]) for u in on_screen], BOX_MARGIN)
                t.stage = "order"
                return
        if not sel & want:
            g.task = None
            return
        want = sel & want
        key = d["key"]
        a = key[0]
        if a in ("attack_unit", "gather", "own_unit"):
            target = next((u for u in obs["units"] if u["tag"] == key[1]), None)
            p = target and screen_of(obs, target["x"], target["y"])
            if p:
                hi.right_click(*p)
                g.say(f"fight: {len(sel & want)} {_names(units)} -> {a.replace('_', ' ')} "
                      f"{unit_name(target['type'])}")
            elif target and a == "attack_unit":   # off screen: attack-move there on the minimap
                hi.minimap_command(C.ORDER_ATTACK_MOVE, target["x"], target["y"])
                g.say(f"fight: {len(want)} {_names(units)} -> attack move (minimap)")
            elif target:
                hi.minimap_right_click(target["x"], target["y"])
                g.say(f"fight: {len(want)} {_names(units)} -> move (minimap)")
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
            g.say(f"fight: {len(sel & want)} {_names(units)} -> {a.replace('_', ' ')}")
        else:
            {"stop": hi.stop, "hold": hi.hold, "stim": hi.stim, "return_cargo": hi.return_cargo}[a]()
            g.say(f"fight: {len(sel & want)} {_names(units)} -> {a.replace('_', ' ')}")
        g.task = None


def _names(units: list[dict]) -> str:
    kinds = {unit_name(u["type"]) for u in units}
    return "/".join(sorted(kinds))


# --- rules ------------------------------------------------------------------------------------

class PullRule:
    """When enemy fighters at a base outnumber Gary's army there, pull a box of workers off the
    minerals to fight them (what players do against an early zergling attack); send them back to
    mining once the base has been calm for 5 s."""

    def __init__(self, g: Gary):
        self.g = g
        self.next_pull = 0
        g.task_runners["pull"] = self._run
        g.task_runners["unpull"] = self._run

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0]
        for cc in (u for u in mine if u["type"] == T.CC and u["completed"]):
            near = [u for u in enemy if math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < THREAT_PX]
            if not near:
                continue
            ours = sum(supply_x2(u["type"]) for u in mine if u["type"] in g.army_types and u["completed"]
                       and math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < THREAT_PX + 4 * 32)
            g.calm_since = hi.frame
            theirs = sum(supply_x2(u["type"]) for u in near)
            alive = {u["tag"] for u in mine}
            g.pulled &= alive
            if theirs >= 2 and theirs > ours and len(g.pulled) < 2 * len(near) + 1 and hi.frame >= self.next_pull:
                self.next_pull = hi.frame + 24 * 3              # one pull at a time
                tx = sum(u["x"] for u in near) // len(near)
                ty = sum(u["y"] for u in near) // len(near)
                g.task = Task("pull", data={"cc": (cc["x"], cc["y"]), "to": (tx, ty), "n": 2 * len(near) + 1},
                              started=hi.frame)
                g.say(f"pulls workers: {len(near)} enemy units at the base, outnumbering the army there")
                return True
            return False
        if g.pulled and hi.frame - g.calm_since > CALM_FRAMES:
            g.task = Task("unpull", started=hi.frame)
            return True
        return False

    def _run(self, obs: dict, mine: list[dict], t: Task) -> None:
        if self.g.hi.frame - t.started > 24 * 8:
            self.g.task = None
            return
        (self._pull if t.kind == "pull" else self._unpull)(obs, mine, t)

    def _pull(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi, d = self.g, self.g.hi, t.data
        if t.stage == "selected":
            workers = [u["tag"] for u in mine if u["tag"] in obs["selection"] and u["type"] == T.SCV]
            if workers:
                hi.minimap_command(C.ORDER_ATTACK_MOVE, *d["to"])
                g.pulled.update(workers)
            g.task = None
            return
        cx, cy = d["cc"]
        if not screen_of(obs, cx, cy):
            hi.camera_minimap(cx, cy)
            return
        miners = [u for u in mine if u["type"] == T.SCV and u["tag"] not in g.pulled
                  and math.dist((u["x"], u["y"]), (cx, cy)) < 8 * 32 and screen_of(obs, u["x"], u["y"])]
        if not miners:
            g.task = None
            return
        # about two workers per attacker, the ones closest to the attackers
        tx, ty = d["to"]
        miners.sort(key=lambda u: math.dist((u["x"], u["y"]), (tx, ty)))
        box_around(hi, [screen_of(obs, u["x"], u["y"]) for u in miners[:min(10, d["n"] - len(g.pulled))]], 6)
        t.stage = "selected"

    def _unpull(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Pulled workers back to the minerals of the nearest base."""
        g, hi = self.g, self.g.hi
        back = [u for u in mine if u["tag"] in g.pulled]
        if not back:
            g.pulled.clear()
            g.task = None
            return
        if t.stage == "selected":
            sel = [u for u in back if u["tag"] in obs["selection"]]
            if sel:
                w = sel[0]
                base = min(g.map.bases, key=lambda b: math.dist(b.center, (w["x"], w["y"])))
                fields = [m for m in base.minerals if screen_of(obs, *m)]
                if fields:
                    hi.right_click(*screen_of(obs, *fields[0]))
                else:
                    hi.minimap_right_click(*base.minerals[0])
                g.pulled -= {u["tag"] for u in sel}
            t.stage = "start"
            t.data["boxes"] = t.data.get("boxes", 0) + 1
            if t.data["boxes"] >= 3:
                g.pulled.clear()
                g.task = None
            return
        cx = sorted(u["x"] for u in back)[len(back) // 2]
        cy = sorted(u["y"] for u in back)[len(back) // 2]
        here = [u for u in back if screen_of(obs, u["x"], u["y"])]
        if not here:
            hi.camera_minimap(cx, cy)
            return
        box_around(hi, [screen_of(obs, u["x"], u["y"]) for u in here], 6)
        t.stage = "selected"


class LoadRule:
    """Zerg fighters coming at a finished bunker that isn't full: put the nearest marines in it
    (right-click the bunker), as players do. The posture can ask for bunkers loaded before the
    enemy is there, from anywhere in the main (preload_bunkers). Marines inside are hidden, so
    Gary counts as loaded the marines it sent that are inside (not seen) or still on their way."""

    def __init__(self, g: Gary):
        self.g = g
        self.next_load = 0
        g.task_runners["load"] = self._run

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        if hi.frame < self.next_load:
            return False
        early = g.posture.preload_bunkers
        reach = LOAD_FROM if early else BUNKER_FROM
        seen = {u["tag"]: u for u in mine}
        enemy = enemy_fighters(obs, g.slot)
        for b in mine:
            if b["type"] != BUNKER or not b["completed"]:
                continue
            if not (early or any(math.dist((e["x"], e["y"]), (b["x"], b["y"])) < BUNKER_THREAT for e in enemy)):
                continue
            sent = g.loaded.setdefault(b["tag"], set())
            sent &= {t for t in sent if t not in seen or seen[t].get("order_target") == b["tag"]}
            free = BUNKER_SLOTS - len(sent)
            marines = sorted((u for u in mine if u["type"] == MARINE and u["completed"] and u["tag"] not in sent
                              and math.dist((u["x"], u["y"]), (b["x"], b["y"])) < reach),
                             key=lambda u: math.dist((u["x"], u["y"]), (b["x"], b["y"])))[:free]
            if free <= 0 or not marines:
                continue
            self.next_load = hi.frame + 24 * 2
            g.task = Task("load", data={"bunker": b["tag"], "x": b["x"], "y": b["y"],
                                        "marines": [u["tag"] for u in marines]}, started=hi.frame)
            return True
        return False

    def _run(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Drag-select the marines (camera on them first if needed), camera on the bunker if it isn't
        on screen, right-click the bunker."""
        g, hi, d = self.g, self.g.hi, t.data
        if hi.frame - t.started > 24 * 3:
            g.task = None
            return
        units = [u for u in mine if u["tag"] in set(d["marines"])]
        if not units:
            g.task = None
            return
        if t.stage == "start":
            on_screen = [u for u in units if screen_of(obs, u["x"], u["y"])]
            if not on_screen:
                if d.get("camera_moved"):
                    g.task = None
                    return
                d["camera_moved"] = True             # on the marines nearest the bunker
                g.camera_to(int(units[0]["x"]), int(units[0]["y"]))
                return
            box_around(hi, [screen_of(obs, u["x"], u["y"]) for u in on_screen], BOX_MARGIN)
            t.stage = "order"
            return
        chosen = set(obs["selection"]) & {u["tag"] for u in units}
        p = screen_of(obs, d["x"], d["y"])
        if chosen and not p and not d.get("to_bunker"):
            d["to_bunker"] = True
            g.camera_to(int(d["x"]), int(d["y"]))
            return
        if chosen and p:
            hi.right_click(*p)
            g.loaded.setdefault(d["bunker"], set()).update(chosen)
            g.say(f"loads {len(chosen)} marine{'s' * (len(chosen) != 1)} into the bunker")
        g.task = None


class RepairRule:
    """A damaged bunker with zerglings near: the 2 nearest free SCVs repair it."""

    def __init__(self, g: Gary):
        self.g = g
        self.next_repair = 0
        g.task_runners["repair"] = self._run

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        if hi.frame < self.next_repair:
            return False
        bunker = next((u for u in mine if u["type"] == T.BUNKER and u["completed"] and u.get("hp_max")
                       and u["hp"] < u["hp_max"]), None)
        if not bunker:
            return False
        fighters = enemy_fighters(obs, g.slot)
        if not any(math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])) < REPAIR_THREAT for u in fighters):
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
        g.task = Task("repair", data={"bunker": bunker["tag"], "x": bunker["x"], "y": bunker["y"],
                                      "scvs": [u["tag"] for u in scvs]}, started=hi.frame)
        return True

    def _run(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi, d = self.g, self.g.hi, t.data
        if hi.frame - t.started > 24 * 3:
            g.task = None
            return
        units = [u for u in mine if u["tag"] in set(d["scvs"])]
        if not units:
            g.task = None
            return
        if t.stage == "start":
            if not screen_of(obs, d["x"], d["y"]) or not all(screen_of(obs, u["x"], u["y"]) for u in units):
                if d.get("camera_moved"):
                    units = [u for u in units if screen_of(obs, u["x"], u["y"])]
                    if not units or not screen_of(obs, d["x"], d["y"]):
                        g.task = None
                        return
                else:
                    d["camera_moved"] = True
                    g.camera_to(int(d["x"]), int(d["y"]))
                    return
            box_around(hi, [screen_of(obs, u["x"], u["y"]) for u in units], 12)
            t.stage = "order"
            return
        p = screen_of(obs, d["x"], d["y"])
        if p and set(obs["selection"]) & set(d["scvs"]):
            hi.right_click(*p)                   # an SCV's right-click on a damaged building: repair
            g.say(f"{len(set(obs['selection']) & set(d['scvs']))} SCVs repair the bunker")
        g.task = None


class ReinforceRule:
    """Fighters just out of a Barracks while zerg fighters are at the base: attack-move them to
    the fight (on the minimap). The rally is a plain move, and a moving marine doesn't shoot."""

    def __init__(self, g: Gary):
        self.g = g
        self.known: set[int] | None = None       # fighters Gary has seen
        g.task_runners["reinforce"] = self._run

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        army = [u for u in mine if u["type"] in g.army_types and u["completed"]]
        if self.known is None:
            self.known = {u["tag"] for u in army}
        new = [u for u in army if u["tag"] not in self.known]
        self.known |= {u["tag"] for u in army}
        if not new:
            return False
        at_home = near_home(mine, enemy_fighters(obs, g.slot))
        if not at_home:
            return False
        cx, cy = sum(u["x"] for u in new) / len(new), sum(u["y"] for u in new) / len(new)
        target = min(at_home, key=lambda e: math.dist((e["x"], e["y"]), (cx, cy)))
        g.task = Task("reinforce", data={"tags": [u["tag"] for u in new], "x": target["x"], "y": target["y"]},
                      started=g.hi.frame)
        return True

    def _run(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Camera on the new fighters, drag-select them, attack-move on the minimap to the fight."""
        g, hi, d = self.g, self.g.hi, t.data
        units = [u for u in mine if u["tag"] in set(d["tags"])]
        if hi.frame - t.started > 24 * 3 or not units:
            g.task = None
            return
        if t.stage == "start":
            on_screen = [u for u in units if screen_of(obs, u["x"], u["y"])]
            if not on_screen:
                if d.get("camera_moved"):
                    g.task = None
                    return
                d["camera_moved"] = True
                g.camera_to(int(units[0]["x"]), int(units[0]["y"]))
                return
            box_around(hi, [screen_of(obs, u["x"], u["y"]) for u in on_screen], 12)
            t.stage = "order"
            return
        chosen = set(obs["selection"]) & set(d["tags"])
        if chosen:
            hi.minimap_command(C.ORDER_ATTACK_MOVE, int(d["x"]), int(d["y"]))
            g.say(f"reinforce: {len(chosen)} new {'units' if len(chosen) > 1 else 'unit'} attack-move to the fight")
        g.task = None


RULES = {"pull": PullRule, "load": LoadRule, "repair": RepairRule, "reinforce": ReinforceRule}
