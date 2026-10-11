"""The fight (layer L4, docs/CONTROL.md): what Gary's units do when enemy fighters are near.

A fight layer is a model plus rules. Each decision is an intent (intents.py) that the motor layer
(motor.py) carries out with Gary's hands. Models (one command per half second):

  per_unit  the fight model (gary/policy/fight.py; Gary v0.4): what each unit should do, regrouped
            into one command per order
  roles     no model: a role per unit (roles.py: fire, close in, pull back, retreat), by hand rules
            for now, each role's skill choosing where to go
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
  unload     zerg fighters at the base, none near a loaded bunker (the main's, from an early-pool
             scare, while the natural is attacked): let its marines out and send them to the fight
  ramp       asked to (g.hold_ramp), the army in the main holds the main's ramp from its top
"""

from __future__ import annotations

import math

import numpy as np

from gary import terran as T
from gary.agent.gary import Gary
from gary.agent.intents import Intent
from gary.interface import screen_of
from gary.policy.army import is_building, is_worker, supply_x2
from gary.policy.fight import ACTIONS, FIGHT_AROUND, snapshot
from gary.policy.fight import MAX_UNITS as PER_UNIT_MAX
from gary.policy.fight_cmd import MAX_UNITS as COMMAND_MAX, MOVES, OWN_RADIUS, POINTER
from gary.policy.fight_memory import history_inputs

ASK_EVERY = 12                       # frames between fight decisions (half a second)
INTERRUPTIBLE = {"mine", "gas", "hotkey_building"}   # chores a fight may interrupt
FIGHT_TASKS = {"fight_cmd", "load", "pull", "unpull", "repair", "reinforce", "unload"}
RESOURCES = {176, 177, 178, 188, 110}
ACT_IF_NONE_BELOW = 0.2              # act on a unit when the model gives "nothing" less than this
                                     # (with 0.2 Gary acts on about as many units as pros do: 22% vs 27%)
SAME_POINT = 96                      # moves to points this close count as the same order
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
MINERALS = {176, 177, 178}
MINERAL_LINE_PX = 4 * 32             # pull(mineral_line): zerg this close to a base's minerals are in its mineral line
THREAT_PX = 10 * 32                  # enemy fighters this close to a Command Center threaten its workers
CALM_FRAMES = 24 * 5                 # pulled workers go back to mining after 5 s without a threat
REPAIR_ORDER = 34
REPAIRERS = 2
REPAIR_THREAT = 9 * 32               # zerglings this close to the bunker: repair it
SAFE_AROUND = 8 * 32                 # safe rally: the zerg and Gary's fighters this close to the fight compared
REPAIR_PAST = 64                     # surround: the move point past the bunker's center (pixels; it's 96x64)
HOME = 12 * 32                       # zerg fighters this close to a Command Center or bunker: at home
PACK_PX, PACK_MIN = 4 * 32, 3        # reinforce to_pack: fighters with this many (itself too) this close
UNLOAD_CLEAR = 12 * 32               # unload: no enemy fighter this close to the bunker
UNLOAD_EVERY = 24 * 10
HOLD_ORDER = 107
MOVE_ORDER = 6
RAMP_SLACK = 40                      # ramp hold: a unit this close to its spot is there
RAMP_IN = 2 * 32                     # ...zerg fighters this much nearer the main's center than the top are in the main
RAMP_NEAR = 9 * 32                   # ...auto: zerg fighters this close to the ramp's top or bottom ask for it
RAMP_FRONT_TYPES = {34, 32}          # ...medics and firebats on the top, the others just behind
RAMP_BLOCKERS = 4                    # ...units wanted on the top (SCVs fill up the medics and firebats)
RAMP_GAP = 20                        # ...the top's line: a place per this much of its width (pixels)
RAMP_GATHER = 4 * 32                 # ...front units farther from the top go there together first
RAMP_MAX_SPAN = 8 * 32               # ...looked for walls this far each way
RAMP_ON_SPOT = 12                    # ...a front unit this close to its place is there
RAMP_SCVS_FROM = 12 * 32             # ...SCVs this close to the main's center (its mineral line) can go
RAMP_HOLD_TTL = 24 * 2               # ...the units holding it are the rule's this long, renewed every turn
RAMP_RANGE = {0: 160, 32: 48}        # ...a zergling this close (+32) to a unit reaches it (marine, firebat; else 32)


def enemy_fighters(obs: dict, slot: int) -> list[dict]:
    return [u for u in obs["units"] if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
            and not is_worker(u["type"]) and not is_building(u["type"])]


def near_home(mine: list[dict], fighters: list[dict]) -> list[dict]:
    """The enemy fighters at Gary's base: near a Command Center or a bunker."""
    home = [(u["x"], u["y"]) for u in mine if u["type"] in (T.CC, T.BUNKER)]
    return [e for e in fighters if any(math.dist((e["x"], e["y"]), h) < HOME for h in home)]


def command_intent(key: tuple, tags: list[int], center) -> Intent:
    """A fight model's command as an intent: ("move", x, y), ("attack_unit", tag), ("stim",)..."""
    target = key[1] if len(key) == 2 else (key[1], key[2]) if len(key) == 3 else None
    return Intent(key[0].upper(), list(tags), target, {"key": key, "tags": tags, "center": center}, "fight")


class FightLayer:
    def __init__(self, g: Gary, model, kind: str, before: list = (), in_fight: list = (),
                 guard_rail: bool = False, workers: bool = True, seed: int = 0, ctrl_click: bool = False,
                 watch: int = 0, role_options: dict | None = None):
        self.g = g
        self.model = model
        self.kind = kind                         # "per_unit", "command" or "roles"
        self.before = list(before)               # rules asked first in a fight
        self.in_fight = list(in_fight)           # rules asked right before the model
        self.guard_rail = guard_rail             # drop the model's plain moves and stops (v0.7)
        self.workers = workers                   # False: workers belong to the rules
        self.ctrl_click = ctrl_click             # select a group that is every unit of one type on
                                                 # screen with a ctrl+click, not a drag box
        self.own_radius, self.max_units = (FIGHT_AROUND, PER_UNIT_MAX) if kind == "per_unit" else (OWN_RADIUS, COMMAND_MAX)
        self.watch = watch                       # enemy fighters this close make a fight (0: the model's 8 tiles)
        self.roles = None
        if kind in ("roles", "role_override"):
            from gary.agent.roles import RoleLearned, RoleOverride, RoleRules
            cls = RoleOverride if kind == "role_override" else RoleLearned
            self.roles = cls(g, model, **(role_options or {})) if model is not None else RoleRules(g, **(role_options or {}))
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

    def block(self, obs: dict, mine: list[dict]) -> str | None:
        """The fight's own turn: the rules, then the model. Who acted ("rule:pull", "fight"...), or
        None."""
        for rule in self.before:
            if rule.step(obs, mine):
                return f"rule:{rule.name}"
        return self.decide(obs, mine)

    def decide(self, obs: dict, mine: list[dict]) -> str | None:
        for rule in self.in_fight:
            if rule.step(obs, mine):
                return f"rule:{rule.name}"
        if self.kind in ("roles", "role_override"):
            snap = self._snapshot(obs)
            intent = self.roles.decide(obs, snap)
            if intent is None:
                return None
            self.g.motor.start(intent)
            return "roles"
        acted = self._per_unit(obs, mine) if self.kind == "per_unit" else self._command(obs, mine)
        return "fight" if acted else None

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
                        self.own_radius, self.max_units, self.watch or None)
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
        g.motor.start(command_intent(key, groups[key][:12], center))
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
        chosen = g.leases.free(chosen, "fight")
        if not chosen:
            return False
        for t in chosen:
            self.told[t] = (key, g.hi.frame)
        g.motor.start(command_intent(key, chosen, center))
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

# --- rules ------------------------------------------------------------------------------------

class PullRule:
    """When enemy fighters at a base outnumber Gary's army there, pull a box of workers off the
    minerals to fight them (what players do against an early zergling attack); send them back to
    mining once the base has been calm for 5 s. Options (manifest "rule_options": {"pull": ...}):

      ratio          pull when the zerg there outweigh the army more than this (supply)
      delay_s        ...and have, without a break, for this long
      per_enemy      workers pulled per enemy unit (and one more)
      attacked_only  only zerg attacking something of Gary's count, not zerg hovering near
      max_share      never more than this share of the base's workers pulled (None: no cap)
      mineral_line   only zerg in the mineral line count (attacking an SCV, or by the base's
                     minerals), not zerg at a bunker or a building further out: there pros keep
                     mining (the fast pools on replayed pros: Gary walked 10 SCVs out to the
                     bunker at the main's exit and lost them all; pros in the same games pulled
                     0-4 and lost none)

    In the home defenses Gary pulled where the pros didn't (44 of 239: small threats, about 3
    zerg against 3 of its own) and, where both did, sooner (3.0 s after the takeover, the pros'
    7.6 s, median), more at once (7.5, the pros' 6.5) and longer (89 SCV-seconds, the pros' 62)."""

    name = "pull"

    def __init__(self, g: Gary, ratio: float = 1.0, delay_s: float = 0, per_enemy: int = 2,
                 attacked_only: bool = False, max_share: float | None = None, mineral_line: bool = False):
        self.g = g
        self.mineral_line = mineral_line
        self.max_share = max_share
        self.next_pull = 0
        self.ratio, self.delay, self.per_enemy, self.attacked_only = ratio, int(delay_s * 24), per_enemy, attacked_only
        self.threat_since: int | None = None

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0]
        for cc in (u for u in mine if u["type"] == T.CC and u["completed"]):
            near = [u for u in enemy if math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < THREAT_PX]
            if self.attacked_only and not any(u.get("order_target") in {m["tag"] for m in mine} for u in near):
                near = []
            if self.mineral_line:
                scvs = {m["tag"] for m in mine if m["type"] == T.SCV}
                fields = [(r["x"], r["y"]) for r in obs["units"] if r["type"] in MINERALS
                          and math.dist((r["x"], r["y"]), (cc["x"], cc["y"])) < THREAT_PX]
                near = [u for u in near if u.get("order_target") in scvs
                        or any(math.dist((u["x"], u["y"]), f) < MINERAL_LINE_PX for f in fields)]
            if not near:
                continue
            ours = sum(supply_x2(u["type"]) for u in mine if u["type"] in g.army_types and u["completed"]
                       and math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < THREAT_PX + 4 * 32)
            g.calm_since = hi.frame
            theirs = sum(supply_x2(u["type"]) for u in near)
            alive = {u["tag"] for u in mine}
            g.pulled &= alive
            threat = theirs >= 2 and theirs > self.ratio * ours
            if not threat:
                self.threat_since = None
            elif self.threat_since is None:
                self.threat_since = hi.frame
            n = self.per_enemy * len(near) + 1
            if self.max_share is not None:
                workers = sum(1 for u in mine if u["type"] == T.SCV and (u["tag"] in g.pulled
                              or math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < 8 * 32))
                n = min(n, max(1, int(self.max_share * workers)))
            if threat and hi.frame - self.threat_since >= self.delay and len(g.pulled) < n and hi.frame >= self.next_pull:
                self.next_pull = hi.frame + 24 * 3              # one pull at a time
                tx = sum(u["x"] for u in near) // len(near)
                ty = sum(u["y"] for u in near) // len(near)
                g.motor.start(Intent("PULL", [], (tx, ty), {"cc": (cc["x"], cc["y"]), "to": (tx, ty), "n": n},
                                     "rule:pull"))
                g.say(f"pulls workers: {len(near)} enemy units at the base, outnumbering the army there")
                return True
            return False
        self.threat_since = None
        if g.pulled and hi.frame - g.calm_since > CALM_FRAMES:
            g.motor.start(Intent("UNPULL", sorted(g.pulled), None, {}, "rule:pull"))
            return True
        return False


class LoadRule:
    """Zerg fighters coming at a finished bunker that isn't full: put the nearest marines in it
    (right-click the bunker), as players do. The posture can ask for bunkers loaded before the
    enemy is there, from anywhere in the main (preload_bunkers). Marines inside are hidden, so
    Gary counts as loaded the marines it sent that are inside (not seen) or still on their way."""

    name = "load"

    def __init__(self, g: Gary, always: bool = False):
        self.g = g
        self.always = always                     # option: every finished bunker kept loaded (as pros
        self.next_load = 0                       # man the natural's), not only once zerg are near

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        if hi.frame < self.next_load:
            return False
        early = g.posture.preload_bunkers or self.always
        seen = {u["tag"]: u for u in mine}
        enemy = enemy_fighters(obs, g.slot)
        for b in mine:
            if b["type"] != BUNKER or not b["completed"]:
                continue
            threat = any(math.dist((e["x"], e["y"]), (b["x"], b["y"])) < BUNKER_THREAT for e in enemy)
            released = b["tag"] in g.release_bunkers       # (posture v5: let out once the rush is over)
            if released and not threat:
                # (marines still walking in stay counted: once inside, they're let out too)
                inside = {t for t in g.loaded.get(b["tag"], set()) if t not in seen}
                if inside:
                    g.loaded[b["tag"]] -= inside
                    self.next_load = hi.frame + 24 * 2
                    g.motor.start(Intent("UNLOAD", [], b["tag"], {"bunker": b["tag"], "x": b["x"], "y": b["y"]},
                                         "rule:load"))
                    return True
            pre = early and not released
            if not (pre or threat):
                continue
            reach = LOAD_FROM if pre else BUNKER_FROM
            sent = g.loaded.setdefault(b["tag"], set())
            sent &= {t for t in sent if t not in seen or seen[t].get("order_target") == b["tag"]}
            free = BUNKER_SLOTS - len(sent)
            marines = sorted((u for u in mine if u["type"] == MARINE and u["completed"] and u["tag"] not in sent
                              and math.dist((u["x"], u["y"]), (b["x"], b["y"])) < reach),
                             key=lambda u: math.dist((u["x"], u["y"]), (b["x"], b["y"])))[:free]
            if free <= 0 or not marines:
                continue
            self.next_load = hi.frame + 24 * 2
            g.motor.start(Intent("LOAD", [u["tag"] for u in marines], b["tag"],
                                 {"bunker": b["tag"], "x": b["x"], "y": b["y"],
                                  "marines": [u["tag"] for u in marines]}, "rule:load"))
            return True
        return False


class RepairRule:
    """A damaged bunker with zerglings near: the 2 (option: n) nearest free SCVs repair it. Option
    surround: as pros do, they first move past the bunker, then repair it, so they spread around it
    instead of queueing on the side they came from."""

    name = "repair"

    def __init__(self, g: Gary, n: int = REPAIRERS, surround: bool = False):
        self.g = g
        self.n = n
        self.surround = surround
        self.next_repair = 0

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
        if len(repairing) >= self.n:
            return False
        scvs = sorted((u for u in mine if u["type"] == T.SCV and u["completed"] and u["tag"] not in {r["tag"] for r in repairing}),
                      key=lambda u: math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])))[:self.n - len(repairing)]
        if not scvs:
            return False
        self.next_repair = hi.frame + 24 * 3
        data = {"bunker": bunker["tag"], "x": bunker["x"], "y": bunker["y"], "scvs": [u["tag"] for u in scvs]}
        if self.surround and len(scvs) > 1:
            # past the bunker: its far side from the SCVs, just beyond its edge
            cx, cy = sum(u["x"] for u in scvs) / len(scvs), sum(u["y"] for u in scvs) / len(scvs)
            d = math.dist((cx, cy), (bunker["x"], bunker["y"])) or 1
            ux, uy = (bunker["x"] - cx) / d, (bunker["y"] - cy) / d
            data["past"] = (int(bunker["x"] + ux * REPAIR_PAST), int(bunker["y"] + uy * REPAIR_PAST))
        g.motor.start(Intent("REPAIR", [u["tag"] for u in scvs], bunker["tag"], data, "rule:repair"))
        return True


class ReinforceRule:
    """Fighters just out of a Barracks while zerg fighters are at the base: attack-move them to
    the fight (on the minimap). The rally is a plain move, and a moving marine doesn't shoot.
    Option to_pack: to Gary's nearest bunch of fighters instead, when there is one (one by one into
    the zerg, they died alone: in 48 games on Python, 48% of Gary's army deaths had no other of its
    fighters within 6 tiles, pros' 12%; those were 13 s old, median). Option safe: while the zerg
    there outweigh Gary's fighters already at the fight, to the main's mineral line instead (as
    pros re-rally: the new ones gather behind the SCVs until there are enough)."""

    name = "reinforce"

    def __init__(self, g: Gary, to_pack: bool = False, safe: bool = False):
        self.g = g
        self.to_pack = to_pack
        self.safe = safe
        self.known: set[int] | None = None       # fighters Gary has seen

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
        if g.ramp_point is not None:                        # (holding the ramp: to its back spot)
            target = {"x": g.ramp_point[0], "y": g.ramp_point[1]}
        elif self.safe and self._outweighed(army, new, at_home, target):
            spot = self._mineral_line(obs, mine)
            if spot is not None:
                target = {"x": spot[0], "y": spot[1]}
        elif self.to_pack:
            old = [u for u in army if u["tag"] not in {n["tag"] for n in new}]
            packed = [u for u in old if sum(1 for v in old if math.dist((u["x"], u["y"]), (v["x"], v["y"])) < PACK_PX) >= PACK_MIN]
            if packed:
                target = min(packed, key=lambda u: math.dist((u["x"], u["y"]), (target["x"], target["y"])))
        g.motor.start(Intent("REINFORCE", [u["tag"] for u in new], (target["x"], target["y"]),
                             {"tags": [u["tag"] for u in new], "x": target["x"], "y": target["y"]}, "rule:reinforce"))
        return True


    def _outweighed(self, army: list[dict], new: list[dict], at_home: list[dict], target: dict) -> bool:
        """The zerg around the nearest one at home outweigh Gary's fighters already near them."""
        p = (target["x"], target["y"])
        fresh = {u["tag"] for u in new}
        theirs = sum(supply_x2(e["type"]) for e in at_home if math.dist((e["x"], e["y"]), p) < SAFE_AROUND)
        ours = sum(supply_x2(u["type"]) for u in army
                   if u["tag"] not in fresh and math.dist((u["x"], u["y"]), p) < SAFE_AROUND)
        return theirs > ours

    def _mineral_line(self, obs: dict, mine: list[dict]) -> tuple[int, int] | None:
        """Between the main's Command Center and its minerals: the SCVs mining there are a wall."""
        g = self.g
        ccs = [u for u in mine if u["type"] == T.CC and u["completed"]]
        if not ccs:
            return None
        cc = min(ccs, key=lambda u: math.dist((u["x"], u["y"]), g.main.center))
        fields = [u for u in obs["units"] if u["type"] in MINERALS and math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < 10 * 32]
        if not fields:
            return None
        mx, my = sum(u["x"] for u in fields) / len(fields), sum(u["y"] for u in fields) / len(fields)
        return int((cc["x"] + mx) / 2), int((cc["y"] + my) / 2)


class UnloadRule:
    """Zerg fighters at Gary's base, none near a loaded bunker: its marines sit out the fight (the
    main's bunker from an early-pool scare, while the zerg hit the natural). Let them out (select
    the bunker, U), then attack-move them to the fight. Not while the posture wants bunkers loaded."""

    name = "unload"

    def __init__(self, g: Gary):
        self.g = g
        self.next_unload = 0
        self.out: tuple[set, int] | None = None     # marines just let out, until when to wait for them

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        fighters = enemy_fighters(obs, g.slot)
        if self.out is not None:                    # out of the bunker: to the fight
            inside, until = self.out
            out = [u for u in mine if u["tag"] in inside]
            at_home = near_home(mine, fighters)
            if not out and hi.frame <= until:
                return False
            self.out = None
            if not out or not at_home:
                return False
            cx, cy = sum(u["x"] for u in out) / len(out), sum(u["y"] for u in out) / len(out)
            target = min(at_home, key=lambda e: math.dist((e["x"], e["y"]), (cx, cy)))
            g.motor.start(Intent("REINFORCE", [u["tag"] for u in out], (target["x"], target["y"]),
                                 {"tags": [u["tag"] for u in out], "x": target["x"], "y": target["y"]}, "rule:unload"))
            return True
        if hi.frame < self.next_unload or g.posture.preload_bunkers or not near_home(mine, fighters):
            return False
        seen = {u["tag"] for u in mine}
        for b in mine:
            if b["type"] != BUNKER or not b["completed"]:
                continue
            inside = {t for t in g.loaded.get(b["tag"], set()) if t not in seen}
            if not inside or any(math.dist((e["x"], e["y"]), (b["x"], b["y"])) < UNLOAD_CLEAR for e in fighters):
                continue
            self.next_unload = hi.frame + UNLOAD_EVERY
            g.loaded[b["tag"]] = set()
            self.out = (inside, hi.frame + 24 * 3)
            g.motor.start(Intent("UNLOAD", [], b["tag"], {"bunker": b["tag"], "x": b["x"], "y": b["y"]}, "rule:unload"))
            return True
        return False


class RampRule:
    """Hold the main's ramp from its top, when asked (g.hold_ramp: a higher decision, or option auto:
    on one base, zerg fighters seen at the ramp): the army in the main goes up, medics and firebats
    on the top (they take the hits; zerglings come up a few at a time), the others just behind
    (gary/terrain.py: ramp_top, ramp_inside), in a line from wall to wall; with fewer medics and
    firebats than places in it (one per RAMP_GAP of its width; `blockers` at least, 0: none), SCVs
    from the main's mineral line fill the top (as pros do: the zerglings can't get past them, and
    come up one or two at a time). Each holds position once there, or where it is once a zergling
    reaches it (a moving unit doesn't shoot). Those units are the rule's while it holds
    (g.ramp_units; the roles, the army and the chores leave them alone); new fighters rally to its
    back spot (g.ramp_point). With zerg fighters in the main the ramp is lost: the rule lets go (its
    SCVs are pulled workers then: they fight, and mine again once it's calm)."""

    name = "ramp"

    def __init__(self, g: Gary, auto: bool = False, blockers: int = RAMP_BLOCKERS):
        self.g = g
        self.auto = auto
        self.blockers = blockers
        self.scvs: set[int] = set()              # the SCVs on the top
        self.points = None                       # (top, inside, distance from the main's center to the top) or False
        g.hold_ramp = False
        g.ramp_units = set()

    def _points(self):
        if self.points is None:
            self.points = False
            g = self.g
            try:
                from gary.terrain import map_analysis
                a = map_analysis(g.hi.game, g.map)
                r = a["starts"][min(range(len(g.map.starts)), key=lambda k: math.dist(g.map.starts[k], g.main.center))]
                if r["reliable"]:
                    top = tuple(r["ramp_top"])
                    self.points = (top, tuple(r["ramp_inside"]), tuple(r["ramp_bottom"]), math.dist(top, g.main.center))
            except Exception:                    # (no terrain in this game: no ramp hold)
                pass
        return self.points or None

    def _let_go(self, mine: list[dict]) -> bool:
        g = self.g
        g.leases.release(g.ramp_units)
        if self.scvs:                            # to the pull rule: they fight, then mine once it's calm
            g.pulled |= self.scvs & {u["tag"] for u in mine}
            g.calm_since = g.hi.frame
        g.ramp_units, g.ramp_point, self.scvs = set(), None, set()
        return False

    def _places(self, top, inside) -> list[tuple[int, int]]:
        """The top's line: across the ramp where the route narrows to it, from wall to wall
        (walkable ground), places RAMP_GAP apart. Computed once."""
        if getattr(self, "places", None) is None:
            d = math.dist(top, inside) or 1
            across = (-(top[1] - inside[1]) / d, (top[0] - inside[0]) / d)
            game = self.g.hi.game
            reach = []
            for side in (1, -1):             # walkable ground from the top to the wall, each way
                k = 0
                while k < RAMP_MAX_SPAN and game.walkable(int(top[0] + across[0] * (k + 4) * side),
                                                          int(top[1] + across[1] * (k + 4) * side)):
                    k += 4
                reach.append(k)
            span = reach[0] + reach[1]
            n = max(1, round(span / RAMP_GAP))
            start = -reach[1] + span / n / 2
            self.places = [(int(top[0] + across[0] * (start + span / n * k)), int(top[1] + across[1] * (start + span / n * k)))
                           for k in range(n)]
        return self.places

    def _line(self, front: list[dict], top, inside) -> dict:
        """Each front unit's place in the top's line, the places nearest the middle first, each to
        the nearest unit still without one; extra units on the top itself."""
        if not front:
            return {}
        places = sorted(self._places(top, inside), key=lambda q: math.dist(q, top))
        out, left = {}, list(front)
        for q in places:
            if not left:
                break
            u = min(left, key=lambda u: math.dist((u["x"], u["y"]), q))
            out[u["tag"]] = q
            left.remove(u)
        for u in left:
            out[u["tag"]] = top
        return out

    def _blockers(self, mine: list[dict], front: list[dict], top, wanted: int) -> list[dict]:
        """SCVs for the top, up to `wanted` front units: those already there, then the nearest mining
        in the main."""
        g = self.g
        want = wanted - len(front) if self.blockers else 0
        if want <= 0:
            self.scvs = set()
            return []
        scvs = [u for u in mine if u["type"] == T.SCV and u["completed"]]
        kept = [u for u in scvs if u["tag"] in self.scvs][:want]
        main = g.main.center
        free = sorted((u for u in scvs if u["tag"] not in self.scvs and u["tag"] != g.scout and u["tag"] not in g.pulled
                       and u["order"] in HARVEST_ORDERS | {3} and math.dist((u["x"], u["y"]), main) < RAMP_SCVS_FROM),
                      key=lambda u: math.dist((u["x"], u["y"]), top))
        chosen = kept + free[:want - len(kept)]
        self.scvs = {u["tag"] for u in chosen}
        return chosen

    def step(self, obs: dict, mine: list[dict]) -> bool:
        g = self.g
        p = self._points()
        if p is None:
            return self._let_go(mine)
        top, inside, bottom, d_top = p
        main = g.main.center
        fighters = enemy_fighters(obs, g.slot)
        if self.auto:
            one_base = not any(u["type"] == T.CC and math.dist((u["x"], u["y"]), main) > d_top for u in mine)
            g.hold_ramp = one_base and any(math.dist((e["x"], e["y"]), top) < RAMP_NEAR or
                                           math.dist((e["x"], e["y"]), bottom) < RAMP_NEAR for e in fighters)
        if not g.hold_ramp or any(math.dist((e["x"], e["y"]), main) < d_top - RAMP_IN for e in fighters):
            return self._let_go(mine)
        army = [u for u in mine if u["type"] in g.army_types and u["completed"]
                and math.dist((u["x"], u["y"]), main) <= d_top + RAMP_SLACK]
        if not army:
            return self._let_go(mine)
        front = [u for u in army if u["type"] in RAMP_FRONT_TYPES]
        front += self._blockers(mine, front, top, max(self.blockers, len(self._places(top, inside))))
        back = [u for u in army if u["type"] not in RAMP_FRONT_TYPES]
        g.ramp_units, g.ramp_point = {u["tag"] for u in front + back}, inside
        now = g.hi.frame
        g.leases.take(g.ramp_units, Intent("HOLD", list(g.ramp_units), None, {}, "rule:ramp", ttl=RAMP_HOLD_TTL))
        reached = lambda u: any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) <= RAMP_RANGE.get(u["type"], 32) + 32
                                for e in fighters)
        told = getattr(self, "told", {})
        self.told = told
        fresh = lambda u, kind: now - told.get(u["tag"], ("", -10 ** 9))[1] >= 24 or told[u["tag"]][0] != kind
        spots = {**self._line(front, top, inside), **{u["tag"]: inside for u in back}}
        on_top = {u["tag"] for u in front}

        def at(u) -> bool:
            """On its place: a front unit on it, or near it and done walking (others in the way); one
            behind near its spot."""
            d = math.dist((u["x"], u["y"]), spots[u["tag"]])
            if u["tag"] in on_top:
                return d <= RAMP_ON_SPOT or (d <= RAMP_SLACK and u.get("order") != MOVE_ORDER)
            return d <= RAMP_SLACK

        def hold(units: list[dict]) -> bool:
            units = [u for u in units if u.get("order") != HOLD_ORDER and fresh(u, "hold")][:12]
            if not units:
                return False
            for u in units:
                told[u["tag"]] = ("hold", now)
            tags = [u["tag"] for u in units]
            g.motor.start(Intent("HOLD", tags, None, {"key": ("hold",), "tags": tags,
                                                      "center": (units[0]["x"], units[0]["y"])}, "rule:ramp"))
            return True

        def move(units: list[dict]) -> bool:
            # not there, not reached, not already walking there (told less than a second ago, or moving)
            units = [u for u in units if not at(u) and not reached(u) and fresh(u, "move")
                     and not (told.get(u["tag"], ("",))[0] == "move" and u.get("order") == MOVE_ORDER)]
            if not units:
                return False
            far = [u for u in units if u["tag"] in on_top and math.dist((u["x"], u["y"]), top) > RAMP_GATHER]
            if len(far) > 1:                     # the front far away: all to the top at once, the line once there
                units, spot = far[:12], top
            else:                                # (near it: one by one, each to its place in the line)
                spot = spots[units[0]["tag"]]
                units = [u for u in units if spots[u["tag"]] == spot][:12]
            for u in units:
                told[u["tag"]] = ("move", now)
            tags = [u["tag"] for u in units]
            key = ("move", int(spot[0]), int(spot[1]))
            g.motor.start(Intent("MOVE", tags, spot, {"key": key, "tags": tags, "center": spot}, "rule:ramp"))
            return True

        # a zergling on it: hold (shoot) now; then the ones behind (one command), the top's line, arrivals
        return hold([u for u in front + back if reached(u)]) or move(back) or move(front) or \
            hold([u for u in front + back if at(u)])


RULES = {"ramp": RampRule, "pull": PullRule, "load": LoadRule, "repair": RepairRule, "reinforce": ReinforceRule, "unload": UnloadRule}
