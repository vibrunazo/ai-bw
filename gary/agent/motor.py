"""The motor layer (L5, docs/CONTROL.md): skills that turn an intent into clicks over time.

"direct" (Gary v0.1-v0.9's way of carrying decisions out, provider `direct/v1`): one skill per
intent kind, each a task on Gary's hands:

  command      the fight model's commands (MOVE, ATTACK_MOVE, ATTACK_UNIT, GATHER, OWN_UNIT, STOP,
               HOLD, STIM, RETURN_CARGO), and the roles (FIRE, CLOSE_IN, PULL_BACK, RETREAT: roles.py
               picks the point): select the group (or keep the selection when it already is the
               group), camera on it if needed, then right-click / A-click / the key. "exact"
               (a body block, block.py): exactly those units, by clicks on each where the
               game's hit test finds it (not a drag box), waiting to see the selection before
               clicking again (any_scv: whichever mining SCVs the clicks take); "place": a
               right-click exactly on a spot (nudged only off another unit)
  PULL         camera on the base, drag-box the miners nearest the attackers, attack-move there
  UNPULL       drag-box the pulled workers, right-click minerals
  LOAD         drag-select the marines (camera on them if needed), camera on the bunker,
               right-click it
  REPAIR       camera on the bunker and the SCVs, drag-box them, (surround: right-click the ground
               past the bunker, then) right-click the bunker
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
ROLES = {"FIRE", "CLOSE_IN", "PULL_BACK", "RETREAT", "BRACE", "REGROUP", "POST", "SKIRMISH",
         "REPOSITION"}  # roles.py: carried out as commands
TASK_OF = {**{k: "fight_cmd" for k in COMMANDS | ROLES}, "PULL": "pull", "UNPULL": "unpull", "LOAD": "load", "UNLOAD": "unload",
           "REPAIR": "repair", "REINFORCE": "reinforce", "ARMY_MOVE": "army_order", "ARMY_ATTACK": "army_order",
           "SCOUT": "scout", "ATTACK_WAVE": "attack", "SCAN": "scan"}


ENTER_TRANSPORT = 92
ARMY_BOXES = 4                       # an army order's drag boxes (12 units each) at most (a hunt's: HUNT_BOXES,
                                     # army.py; 8 for every order sent more units into fights, lurkers above all)
UNDER_CURSOR = 24                    # a unit this close to a click point (pixels, each axis) is clicked on
EXACT_TRIES = 3                      # exact selections clicked again this often before giving up
PLACE_SEARCH = 12                    # "place" moves look this far for a pixel that hits no other unit
GROUND_SEARCH = 160                  # attack-moves and moves look this far for open ground
OTHER_SIZE = {176: (2, 1), 177: (2, 1), 178: (2, 1), 188: (4, 2), 131: (4, 3), 132: (4, 3), 133: (4, 3)}   # tiles
PAST_FOR = 10                        # repair surround: frames the SCVs walk past the bunker before repairing
LOADING_FOR = 36                     # a load's marines are left alone this long...
LOADING_NEAR = 5 * 32                # ...those this close to its bunker


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
                          ("load", self._run_load), ("unload", self._run_unload), ("scan", self._run_scan), ("repair", self._run_repair), ("reinforce", self._run_reinforce),
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

    def _box_takes_loading(self, obs: dict, mine: list[dict], pts: list, want: set) -> bool:
        """Would a box around pts (screen) also take a unit of Gary's that is entering a bunker?"""
        x0, y0 = min(p[0] for p in pts) - BOX_MARGIN - 8, min(p[1] for p in pts) - BOX_MARGIN - 8
        x1, y1 = max(p[0] for p in pts) + BOX_MARGIN + 8, max(p[1] for p in pts) + BOX_MARGIN + 8
        until, bx, by = self.g.loading
        if self.g.hi.frame < until:                # (what Gary sees is a moment old: the load's own marines)
            for u in mine:
                p = screen_of(obs, u["x"], u["y"])
                if u["type"] == T.MARINE and p and x0 <= p[0] <= x1 and y0 <= p[1] <= y1 \
                        and math.dist((u["x"], u["y"]), (bx, by)) < LOADING_NEAR:
                    return True
        for u in mine:
            if u.get("order") == ENTER_TRANSPORT and u["tag"] not in want:
                p = screen_of(obs, u["x"], u["y"])
                if p and x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                    return True
        return False

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
        if d.get("exact"):                       # exactly these units (a body block): clicks, not a box
            if not self._exact(obs, units, sel, want, t):
                return
            want = sel = set(d["tags"])           # (SCVs: whichever the clicks took)
        elif t.stage == "start":
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
                if (g.fight.ctrl_click or d.get("ctrl_click")) and self._ctrl_click(obs, on_screen):
                    t.stage = "order"
                    return
                # a generous box: units move while Gary reacts (what it sees is 0.3 s old)
                pts = [self.g.hi.aim(obs, u) for u in on_screen]
                if self._box_takes_loading(obs, mine, pts, want):
                    g.task = None                        # (its order would call the load off)
                    return
                box_around(hi, pts, BOX_MARGIN)
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
        elif a == "place":
            # onto an exact spot (a body block, block.py): the click right there, or the nearest
            # pixel where it hits no other unit (the game's own hit test: a click on a unit would follow it)
            gx, gy = self._clear_click(want, key[1], key[2])
            p = screen_of(obs, gx, gy)
            if p:
                hi.right_click(*p)
            else:
                hi.minimap_right_click(key[1], key[2])
            g.say(f"fight: {len(want)} {_names(units)} -> placed")
        elif a in ("move", "attack_move"):
            # on the ground next to the point: a click on a unit there would be an attack on that one
            # unit (A-click) or a follow or attack (right-click): Gary's fighters chased single zerglings
            gx, gy = self._ground_near(obs, key[1], key[2])
            p = screen_of(obs, gx, gy)
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

    def _exact(self, obs: dict, units: list[dict], sel: set, want: set, t: Task) -> bool:
        """Exactly these units selected, the camera on them (and on a "place" order's spot): a click on
        the first, shift+clicks on the others, each where the game's hit test finds that unit; then
        wait to see the selection (what Gary sees is reaction_ms old); when some are already selected
        and nothing else, shift+clicks for the missing ones only (a shift+click on a selected unit
        would drop it). True once they are the selection (go on to the order)."""
        g, hi, d = self.g, self.g.hi, t.data
        if sel == want:
            t.stage = "order"
            return True
        if hi.frame < d.get("seen_at", -1):     # the clicks' selection isn't in sight yet
            return False
        # SCVs mining overlap (harvesters ignore collisions): a click takes one of them, not a chosen
        # one; as a player does, any SCVs do when the order allows it (any_scv: about as many; the
        # skill takes whichever it got, and sends for more if they're too few)
        types = lambda tags: sorted(next((u["type"] for u in obs["units"] if u["tag"] == x), -1) for x in tags)
        if d.get("any_scv") and sel and len(sel) <= len(want) + 1 and not sel & set(d.get("exclude", ())) and \
                all(next((u["type"] for u in obs["units"] if u["tag"] == x), -1) == g.race.worker for x in sel):
            d["tags"] = sorted(sel)
            if t.intent is not None:
                t.intent.units = sorted(sel)
            t.stage = "order"
            return True
        if d.get("clicked") is not None:
            d["tries"] = d.get("tries", 0) + 1
            if d["tries"] > EXACT_TRIES:
                g.task = None
                return False
        key = d["key"]
        points = [(u["x"], u["y"]) for u in units] + ([(key[1], key[2])] if key[0] == "place" else [])
        if not all(screen_of(obs, x, y) for x, y in points):
            if d.get("camera_moved"):
                g.task = None
                return False
            d["camera_moved"] = True
            g.camera_to(int(sum(x for x, _ in points) / len(points)), int(sum(y for _, y in points) / len(points)))
            d["seen_at"] = hi.frame + round(hi.p.reaction_ms * 24 / 1000) + 2     # (until it's seen there)
            return False
        add = sel and sel < want                 # some of them, nothing else: add the others
        todo = [u for u in units if u["tag"] not in sel] if add else units
        landed = hi.frame
        for i, u in enumerate(todo):
            p = screen_of(obs, *self._hit_point(obs, u))
            if p:
                landed = max(landed, hi.click(*p, shift=add or i > 0).land_frame)
        d["clicked"] = hi.frame
        d["seen_at"] = landed + round(hi.p.reaction_ms * 24 / 1000) + 1
        return False

    def _hit_point(self, obs: dict, u: dict) -> tuple[int, int]:
        """Where a click selects this unit: where it will be (aim), or the nearest pixel of its box there
        where the game's hit test finds it (not a neighbor in front of it)."""
        g = self.g
        a = g.hi.aim(obs, u)
        x, y = (obs["camera"]["x"] + a[0], obs["camera"]["y"] + a[1]) if a else (u["x"], u["y"])
        for r in range(0, 9, 2):
            for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
                if g.hi.game.unit_at(g.slot, x + dx, y + dy) == u["tag"]:
                    return x + dx, y + dy
        return x, y

    def _clear_click(self, movers, x: int, y: int) -> tuple[int, int]:
        """(x, y), or the nearest pixel within PLACE_SEARCH where a click hits no unit but the movers."""
        g = self.g
        ok = lambda px, py: g.hi.game.unit_at(g.slot, px, py) in (0, *movers)
        if ok(x, y):
            return x, y
        for r in range(1, PLACE_SEARCH + 1):
            for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)):
                if ok(x + dx, y + dy):
                    return x + dx, y + dy
        return x, y

    def _ground_near(self, obs: dict, x: int, y: int) -> tuple[int, int]:
        """The nearest point to (x, y), within GROUND_SEARCH, with no unit Gary sees under the cursor
        (as a player clicks the ground beside the zerglings, not on one); (x, y) if none is."""
        def half(u):                          # (a building by its footprint, as the game clicks it)
            w, h = T.SIZE.get(u["type"]) or OTHER_SIZE.get(u["type"], (0, 0))
            return (max(UNDER_CURSOR, w * 16 + 4), max(UNDER_CURSOR, h * 16 + 4))
        units = [(u["x"], u["y"], *half(u)) for u in obs["units"]]
        free = lambda px, py: all(abs(px - ux) > hx or abs(py - uy) > hy for ux, uy, hx, hy in units)
        if free(x, y):
            return x, y
        for r in range(16, GROUND_SEARCH + 1, 16):
            for k in range(16):
                t = k * math.pi / 8
                px, py = int(x + r * math.cos(t)), int(y + r * math.sin(t))
                if free(px, py):
                    return px, py
        return x, y

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
        g.hi.ctrl_click(*g.hi.aim(obs, u))
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
            workers = [u["tag"] for u in mine if u["tag"] in obs["selection"] and u["type"] == g.race.worker]
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
        # (not the ones placing or constructing a building, nor the scout: a builder pulled is a
        # build lost)
        miners = [u for u in mine if u["type"] == g.race.worker and u["tag"] not in g.pulled
                  and math.dist((u["x"], u["y"]), (cx, cy)) < 8 * 32 and screen_of(obs, u["x"], u["y"])
                  and u["order"] not in (30, 31, 33) and u["tag"] != g.scout
                  and (g.parked is None or g.parked.data.get("worker") != u["tag"])
                  and u["tag"] not in {g.builder_of.get(r) for r in g.reserved}]
        if not miners:
            g.task = None
            return
        # about two workers per attacker, the ones closest to the attackers
        tx, ty = d["to"]
        miners.sort(key=lambda u: math.dist((u["x"], u["y"]), (tx, ty)))
        box_around(hi, [self.g.hi.aim(obs, u) for u in miners[:min(10, d["n"] - len(g.pulled))]], 6)
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
                bases = (g.executor.my_bases(mine) if g.home_scvs and g.executor is not None else None) or g.map.bases
                base = g.executor.mine_base(mine, (w["x"], w["y"]), obs) if g.balance else \
                    min(bases, key=lambda b: math.dist(b.center, (w["x"], w["y"])))
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
        box_around(hi, [self.g.hi.aim(obs, u) for u in here], 6)
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
            box_around(hi, [self.g.hi.aim(obs, u) for u in on_screen], BOX_MARGIN)
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
            g.loading = (hi.frame + LOADING_FOR, d["x"], d["y"])
            # every marine the box took goes, not only the ones meant: counted too (5 went for 4 slots,
            # and the one left out was sent again every 2 s for the rest of the game, seed 108)
            went = {u["tag"] for u in mine if u["tag"] in set(obs["selection"]) and u["type"] == T.MARINE} | chosen
            g.loaded.setdefault(d["bunker"], set()).update(went)
            for tag in went:
                g.load_sent_at[tag] = hi.frame
            g.leases.take(chosen, t.intent)
            g.say(f"loads {len(chosen)} marine{'s' * (len(chosen) != 1)} into the bunker")
        g.task = None

    def _run_unload(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Camera on the bunker if needed, click it, U (unload all)."""
        g, hi, d = self.g, self.g.hi, t.data
        if hi.frame - t.started > 24 * 3 or not any(u["tag"] == d["bunker"] for u in mine):
            g.task = None
            return
        if obs["selection"] == [d["bunker"]]:
            hi.unload_all()                      # (OpenBW reports it as refused, and does it)
            g.say("unloads the bunker")
            g.task = None
            return
        p = screen_of(obs, d["x"], d["y"])
        if not p:
            if d.get("camera_moved"):
                g.task = None
                return
            d["camera_moved"] = True
            g.camera_to(int(d["x"]), int(d["y"]))
            return
        hi.click(*p)

    def _run_scan(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Camera on the Comsat Station if needed, click it, scan (on the minimap)."""
        g, hi, d = self.g, self.g.hi, t.data
        comsat = next((u for u in mine if u["tag"] == d["comsat"]), None)
        if hi.frame - t.started > 24 * 3 or comsat is None:
            g.task = None
            return
        if obs["selection"] == [d["comsat"]]:
            hi.minimap_command(C.ORDER_SCANNER_SWEEP, d["x"], d["y"])
            g.say(f"scans ({d['x']}, {d['y']})")
            g.task = None
            return
        p = screen_of(obs, comsat["x"], comsat["y"])
        if not p:
            if d.get("camera_moved"):
                g.task = None
                return
            d["camera_moved"] = True
            g.camera_to(int(comsat["x"]), int(comsat["y"]))
            return
        hi.click(*p)

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
            box_around(hi, [self.g.hi.aim(obs, u) for u in units], 12)
            t.stage = "order"
            return
        chosen = set(obs["selection"]) & set(d["scvs"])
        if t.stage == "order" and d.get("past") and chosen:
            q = screen_of(obs, *self._ground_near(obs, *d["past"]))
            if q:
                hi.right_click(*q)               # past the bunker first: they spread around it
                g.leases.take(chosen, t.intent)
                d["moved"] = hi.frame
                t.stage = "repair"
                return
        if t.stage == "repair" and hi.frame - d["moved"] < PAST_FOR:
            return
        p = screen_of(obs, d["x"], d["y"])
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
            box_around(hi, [self.g.hi.aim(obs, u) for u in on_screen], 12)
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
            if g.executor.army_select == "portrait" and sel and len(sel) < len(obs["selection"]) and not d.get("narrowed"):
                # the box took more than the order's units (drones and overlords at the rally point): a ctrl+click
                # on the portrait of the commonest of them keeps that type only (the player's way)
                from collections import Counter
                kinds = Counter(u["type"] for u in mine if u["tag"] in set(sel))
                if kinds:                     # (none: the order's selected units died this frame, round G12)
                    hi.ctrl_click_portrait(kinds.most_common(1)[0][0])
                    d["narrowed"] = True
                    return
            d["narrowed"] = False
            if sel:
                if d["attack"]:
                    hi.minimap_command(C.ORDER_ATTACK_MOVE, d["x"], d["y"])
                else:
                    hi.minimap_right_click(d["x"], d["y"])
                d["done"] += [s for s in sel if s not in d["done"]]   # a box also grabs units already ordered
                g.leases.take(sel, t.intent)
                # (which units each army order moved, and what ordered it: eval/attacks.py traces late arrivals by it)
                from gary.decision_log import LOG
                LOG.decision(hi.side, "motor", hi.frame, "army order", by=d.get("origin") or ("hunt" if d.get("hunt") else t.intent.source),
                             inputs={"tags": [int(s) for s in sel], "to": [int(d["x"]) // 32, int(d["y"]) // 32],
                                     "attack": bool(d["attack"])})
            t.stage = "start"
            return
        left = [u for u in mine if u["tag"] in set(d["tags"]) - set(d["done"])]
        if not left or d["boxes"] >= d.get("max_boxes", ARMY_BOXES):
            if d["done"]:
                g.say(f"army ({len(d['done'])} units): {'attack' if d['attack'] else 'move'} "
                      f"to ({d['x']}, {d['y']}) ({d['p']:.0%} sure it should move; fight estimate {d['fight']:+.0%})")
            g.task = None
            return
        cx = sorted(u["x"] for u in left)[len(left) // 2]
        cy = sorted(u["y"] for u in left)[len(left) // 2]
        here = [u for u in left if screen_of(obs, u["x"], u["y"])]
        if not here:
            if d.get("hunt"):
                # a hunt's camera on the unit nearest the median: an army in two groups has its median
                # point between them, where the camera found nobody until the order timed out, the army
                # standing for minutes by the last hatchery. (Not the army model's orders: carried out
                # in full there, they cost games, 25 won 7 lost against 28 and 4 on 120 games; those to
                # an army in two groups failing was a protection, see PROGRESS.md)
                u = min(left, key=lambda u: math.dist((u["x"], u["y"]), (cx, cy)))
                cx, cy = u["x"], u["y"]
            hi.camera_minimap(cx, cy)
            return
        box_around(hi, [self.g.hi.aim(obs, u) for u in here], 10)
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
                # a click, not a tiny box: on the current ladder maps the miners stand stacked (7 of 9 SCVs on
                # one spot on Odyssey-RE), a box caught several, the scout was given up and asked for again
                # every 6 frames, and the macro never got a turn again from 1:06 (L1, 3 games lost of 3)
                hi.click(*hi.aim(obs, w))
                t.stage = "picked"
                return
            sel = [u for u in mine if u["tag"] in obs["selection"] and u["type"] == g.race.worker]
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
        for p in d.get("then", ()):                       # a tour: the rest as shift+clicked waypoints
            hi.minimap_right_click(*p, queued=True)
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
            xs = [self.g.hi.aim(obs, u) for u in here]
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
