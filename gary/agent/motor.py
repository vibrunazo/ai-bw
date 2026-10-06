"""The motor layer (L5, docs/CONTROL.md): skills that turn an intent into clicks over time.

"direct" (Gary v0.1-v0.9's way of carrying decisions out, provider `direct/v1`): one skill per
intent kind, each a task on Gary's hands:

  command      the fight model's commands (MOVE, ATTACK_MOVE, ATTACK_UNIT, GATHER, OWN_UNIT, STOP,
               HOLD, STIM, RETURN_CARGO), and the roles (FIRE, CLOSE_IN, PULL_BACK, RETREAT: roles.py
               picks the point): select the group (or keep the selection when it already is the
               group), camera on it if needed, then right-click / A-click / the key
  PULL         camera on the base, drag-box the miners nearest the attackers, attack-move there
  UNPULL       drag-box the pulled workers, right-click minerals
  LOAD         drag-select the marines (camera on them if needed), camera on the bunker,
               right-click it
  REPAIR       camera on the bunker and the SCVs, drag-box them, right-click the bunker
  REINFORCE    camera on the new units, drag-box them, attack-move on the minimap
  ARMY_MOVE, ARMY_ATTACK   drag-box the cluster 12 at a time, order each box on the minimap
  SCOUT        pick a worker once and give it a hotkey; recall it, right-click on the minimap
  ATTACK_WAVE  gather the waiting army with drag boxes, attack-move on the minimap

Every order a skill gives takes its units' leases for the intent (intents.py).
"""

from __future__ import annotations

import math

from gary import commands as C
from gary import terran as T
from gary.agent.gary import Gary, Task
from gary.agent.intents import Intent, Leases
from gary.interface import screen_of, unit_name

BOX_MARGIN = 20                      # screen pixels around the group's units when drag-selecting
SCOUT_KEY = 0
HK_ARMY = [1, 2]
COMMANDS = {"MOVE", "ATTACK_MOVE", "ATTACK_UNIT", "GATHER", "OWN_UNIT", "STOP", "HOLD", "STIM", "RETURN_CARGO"}
ROLES = {"FIRE", "CLOSE_IN", "PULL_BACK", "RETREAT"}           # roles.py: carried out as commands
TASK_OF = {**{k: "fight_cmd" for k in COMMANDS | ROLES}, "PULL": "pull", "UNPULL": "unpull", "LOAD": "load",
           "REPAIR": "repair", "REINFORCE": "reinforce", "ARMY_MOVE": "army_order", "ARMY_ATTACK": "army_order",
           "SCOUT": "scout", "ATTACK_WAVE": "attack"}


def box_around(hi, pts: list[tuple[int, int]], m: int) -> None:
    hi.box(max(0, min(p[0] for p in pts) - m), max(0, min(p[1] for p in pts) - m),
           min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + m), min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + m))


def _names(units: list[dict]) -> str:
    kinds = {unit_name(u["type"]) for u in units}
    return "/".join(sorted(kinds))


class DirectMotor:
    def __init__(self, g: Gary):
        self.g = g
        g.leases = Leases(g)
        for kind, run in (("fight_cmd", self._run_command), ("pull", self._run_pull), ("unpull", self._run_pull),
                          ("load", self._run_load), ("repair", self._run_repair), ("reinforce", self._run_reinforce),
                          ("army_order", self._run_army_order), ("scout", self._run_scout),
                          ("attack", self._run_attack_wave)):
            g.task_runners[kind] = run

    def start(self, intent: Intent) -> None:
        """Carry the intent out: its skill becomes the task on Gary's hands."""
        g = self.g
        intent.issued = g.hi.frame
        g.event("intent", intent=intent.kind, source=intent.source, units=len(intent.units),
                target=list(intent.target) if isinstance(intent.target, tuple) else intent.target)
        g.task = Task(TASK_OF[intent.kind], data=intent.params, started=g.hi.frame, intent=intent)

    # --- the fight model's commands -------------------------------------------------------------

    def _run_command(self, obs: dict, mine: list[dict], t: Task) -> None:
        g = self.g
        if g.hi.frame - t.started > 24 * 3:
            g.task = None
        else:
            self._command(obs, mine, t)
        g.macro_turn = g.task is None            # after each fight command, production gets a turn

    def _command(self, obs: dict, mine: list[dict], t: Task) -> None:
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
                if g.fight.ctrl_click and self._ctrl_click(obs, on_screen):
                    t.stage = "order"
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
        ordered = True
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
            else:
                ordered = False
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
            what = t.intent.kind.lower().replace("_", " ") if t.intent is not None and t.intent.kind in ROLES \
                else a.replace("_", " ")
            g.say(f"fight: {len(sel & want)} {_names(units)} -> {what}")
        else:
            {"stop": hi.stop, "hold": hi.hold, "stim": hi.stim, "return_cargo": hi.return_cargo}[a]()
            g.say(f"fight: {len(sel & want)} {_names(units)} -> {a.replace('_', ' ')}")
        if ordered:
            g.leases.take(want, t.intent)
            if "bunker" in d:                    # a retreat into a bunker: they're its marines now
                g.loaded.setdefault(d["bunker"], set()).update(want)
        g.task = None

    def _ctrl_click(self, obs: dict, group: list[dict]) -> bool:
        """When the group is every unit of one type on screen (12 at most), select it with a
        ctrl+click on its unit nearest the group's middle: one click, and nothing else caught (a
        drag box around marines among SCVs takes the SCVs too). True if Gary clicked."""
        g = self.g
        kinds = {u["type"] for u in group}
        if len(kinds) != 1:
            return False
        kind = kinds.pop()
        same = {u["tag"] for u in obs["units"] if u["owner"] == g.slot and u["type"] == kind
                and screen_of(obs, u["x"], u["y"])}
        if same != {u["tag"] for u in group} or len(same) > 12:
            return False
        cx, cy = sum(u["x"] for u in group) / len(group), sum(u["y"] for u in group) / len(group)
        u = min(group, key=lambda u: math.dist((u["x"], u["y"]), (cx, cy)))
        g.hi.ctrl_click(*screen_of(obs, u["x"], u["y"]))
        return True

    # --- the rules' skills ----------------------------------------------------------------------

    def _run_pull(self, obs: dict, mine: list[dict], t: Task) -> None:
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
                g.leases.take(workers, t.intent)
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
                g.leases.take([u["tag"] for u in sel], t.intent)
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

    def _run_load(self, obs: dict, mine: list[dict], t: Task) -> None:
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
            g.leases.take(chosen, t.intent)
            g.say(f"loads {len(chosen)} marine{'s' * (len(chosen) != 1)} into the bunker")
        g.task = None

    def _run_repair(self, obs: dict, mine: list[dict], t: Task) -> None:
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
        chosen = set(obs["selection"]) & set(d["scvs"])
        if p and chosen:
            hi.right_click(*p)                   # an SCV's right-click on a damaged building: repair
            g.leases.take(chosen, t.intent)
            g.say(f"{len(chosen)} SCVs repair the bunker")
        g.task = None

    def _run_reinforce(self, obs: dict, mine: list[dict], t: Task) -> None:
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
            g.leases.take(chosen, t.intent)
            g.say(f"reinforce: {len(chosen)} new {'units' if len(chosen) > 1 else 'unit'} attack-move to the fight")
        g.task = None

    # --- the army's skills ------------------------------------------------------------------------

    def _run_army_order(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Drag-box the cluster's units (12 at most per box, like the real game) and give each box
        the order on the minimap, until all of them have it."""
        g, hi, d = self.g, self.g.hi, t.data
        if hi.frame - t.started > 24 * 10:
            g.task = None
            return
        if t.stage == "selected":
            sel = [s for s in obs["selection"] if s in set(d["tags"])]
            if sel:
                if d["attack"]:
                    hi.minimap_command(C.ORDER_ATTACK_MOVE, d["x"], d["y"])
                else:
                    hi.minimap_right_click(d["x"], d["y"])
                d["done"] += sel
                g.leases.take(sel, t.intent)
            t.stage = "start"
            return
        left = [u for u in mine if u["tag"] in set(d["tags"]) - set(d["done"])]
        if not left or d["boxes"] >= 4:
            if d["done"]:
                g.say(f"army ({len(d['done'])} units): {'attack' if d['attack'] else 'move'} "
                      f"to ({d['x']}, {d['y']}) ({d['p']:.0%} sure it should move; fight estimate {d['fight']:+.0%})")
            g.task = None
            return
        cx = sorted(u["x"] for u in left)[len(left) // 2]
        cy = sorted(u["y"] for u in left)[len(left) // 2]
        here = [u for u in left if screen_of(obs, u["x"], u["y"])]
        if not here:
            hi.camera_minimap(cx, cy)
            return
        box_around(hi, [screen_of(obs, u["x"], u["y"]) for u in here], 10)
        d["boxes"] += 1
        t.stage = "selected"

    def _run_scout(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi, d = self.g, self.g.hi, t.data
        if hi.frame - t.started > 24 * 10:
            g.task = None
            return
        if g.scout is None:                                # pick a worker, give it a hotkey
            if t.stage == "start":
                if not g.look_at(obs, "main"):
                    return
                w = g.executor.pick_worker(obs, mine)
                if w is None:
                    g.task = None
                    return
                x, y = screen_of(obs, w["x"], w["y"])
                hi.box(max(0, x - 5), max(0, y - 5), min(hi.p.viewport[0] - 1, x + 5), min(hi.p.viewport[1] - 1, y + 5))
                t.stage = "picked"
                return
            sel = [u for u in mine if u["tag"] in obs["selection"] and u["type"] == T.SCV]
            if len(sel) != 1:
                g.task = None                              # try again later
                return
            g.scout = sel[0]["tag"]
            hi.hotkey_set(SCOUT_KEY)
            t.stage = "go"
            return
        if obs["selection"] != [g.scout]:
            if t.stage == "recalled":
                g.task = None
                return
            hi.hotkey_recall(SCOUT_KEY)
            t.stage = "recalled"
            return
        hi.minimap_right_click(*d["to"])
        g.leases.take([g.scout], t.intent)
        if d.get("home"):
            g.army.scout_done = True
            g.say("scout found the enemy base, heading home")
        else:
            g.say(f"scouting {d['to']}")
        g.task = None

    def _run_attack_wave(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Gather marines at the rally point with drag boxes, attack-move."""
        g, hi = self.g, self.g.hi
        army = g.army
        if hi.frame - t.started > 24 * 40:
            g.task = None
            return
        waiting = [u for u in mine if u["type"] in g.army_types and u["completed"]
                   and u["tag"] not in army.army_sent]
        if not waiting:
            army.attacks_sent += 1
            g.task = None
            return
        # look where the waiting marines are (normally the rally point)
        cx = sorted(u["x"] for u in waiting)[len(waiting) // 2]
        cy = sorted(u["y"] for u in waiting)[len(waiting) // 2]
        if not screen_of(obs, cx, cy):
            hi.camera_minimap(cx, cy)
            return
        target = g.enemy_starts[army.attacks_sent % len(g.enemy_starts)]
        group = t.data.get("group", 0)
        if t.stage in ("start", "box"):
            here = [u for u in waiting if screen_of(obs, u["x"], u["y"])]
            if not here or group >= len(HK_ARMY) * 3:
                army.attacks_sent += 1
                g.task = None
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
            army.army_sent.update(sent)
            g.leases.take(sent, t.intent)
            t.data["group"] = group + 1
            t.stage = "box"
