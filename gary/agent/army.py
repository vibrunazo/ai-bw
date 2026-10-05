"""The army (layer L3, docs/CONTROL.md): where groups go, and scouting.

  waves     every 16 army units attack-move to an enemy start (Gary v0.1-v0.2)
  learned   the army model (gary/policy/army.py, pro TvZ army orders; v0.3+): every 2 s it is asked
            about one of up to 3 clusters of the army; when it puts most of its belief on the
            cluster going 2+ grid cells away, says so twice, and its fight estimate doesn't expect
            going to cost army share, Gary drag-boxes the cluster and orders it on the minimap.
            Also the scout: one worker at 9 SCVs to the enemy start locations.

Units in a fight are the fight layer's; the arbiter leaves them out (and the posture can keep the
army home altogether).
"""

from __future__ import annotations

import math

import numpy as np

from gary import commands as C
from gary import terran as T
from gary.agent.gary import Gary, Task
from gary.interface import screen_of
from gary.policy.army import GRID, ArmyModel, ArmyTracker, is_building

WAVE = 16
HK_ARMY = [1, 2]
ASK_EVERY = 24 * 2                  # how often Gary asks the model about its army
MOVE_CELLS = 2                      # destinations closer than this are "stay"
MOVE_P = 0.5                        # move when the model puts more belief than this away from here
FIGHT_MIN = -0.02                   # don't go if the fight estimate says we'd lose 2+ points of army share
FIGHT_MARGIN = 0.01                 # ...or if going looks worse than staying by more than 1 point
REORDER_AFTER = 24 * 8              # don't repeat the same order to a group within 8 s
SCOUT_AT = 24 * 45                  # send a scouting worker from 0:45, at 9 SCVs (like pros)
SCOUT_WORKERS = 9
SCOUT_KEY = 0


def cell_dist(a: int, b: int) -> int:
    return max(abs(a // GRID - b // GRID), abs(a % GRID - b % GRID))


class WaveArmy:
    remembers = False

    def __init__(self, g: Gary):
        self.g = g
        self.attacks_sent = 0
        self.army_sent: set[int] = set()
        g.task_runners["attack"] = self._run_attack

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        """Attack-move with every 16 army units. True if Gary acted."""
        g = self.g
        army = [u for u in done if u["type"] in g.army_types and u["tag"] not in self.army_sent]
        if len(army) >= WAVE:
            g.task = Task("attack", started=g.hi.frame)
            return True
        return False

    def _run_attack(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Gather marines at the rally point with drag boxes, attack-move."""
        g, hi = self.g, self.g.hi
        if hi.frame - t.started > 24 * 40:
            g.task = None
            return
        waiting = [u for u in mine if u["type"] in g.army_types and u["completed"]
                   and u["tag"] not in self.army_sent]
        if not waiting:
            self.attacks_sent += 1
            g.task = None
            return
        # look where the waiting marines are (normally the rally point)
        cx = sorted(u["x"] for u in waiting)[len(waiting) // 2]
        cy = sorted(u["y"] for u in waiting)[len(waiting) // 2]
        if not screen_of(obs, cx, cy):
            hi.camera_minimap(cx, cy)
            return
        target = g.enemy_starts[self.attacks_sent % len(g.enemy_starts)]
        group = t.data.get("group", 0)
        if t.stage in ("start", "box"):
            here = [u for u in waiting if screen_of(obs, u["x"], u["y"])]
            if not here or group >= len(HK_ARMY) * 3:
                self.attacks_sent += 1
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
            self.army_sent.update(sent)
            t.data["group"] = group + 1
            t.stage = "box"


class LearnedArmy:
    remembers = True                # whatever Gary looks at, the tracker remembers

    def __init__(self, g: Gary, model: ArmyModel, pull=None):
        self.g = g
        self.model = model
        self.tracker = ArmyTracker(model.spec, g.slot, g.map.size, g.main.center)
        self.pull = pull            # a worker-pull rule run first (v0.3; later versions: the fight layer's)
        self.next_ask = 0
        self.ask_turn = 0
        self.proposed: dict[int, int] = {}            # cluster cell -> destination the model proposed
        self.ordered: dict[int, tuple[int, bool, int]] = {}   # cluster cell -> (cell, attack, frame)
        self.held: dict[int, int] = {}                # cluster cell -> destination Gary decided against
        self.scout_targets: list[tuple[int, int]] = []
        self.scout_done = False
        g.task_runners["scout"] = self._run_scout
        g.task_runners["army_order"] = self._run_army_order

    def remember(self, obs: dict) -> None:
        self.tracker.remember(obs)

    def _clusters(self, army: list[dict]) -> list[list[dict]]:
        """The army split by where it stands: up to 3 clusters (the main army, reinforcements on
        their way, ...), biggest first. Units within 2 grid cells of a cluster's center join it."""
        left, out = list(army), []
        while left and len(out) < 3:
            cells = [self.tracker.cell(u["x"], u["y"]) for u in left]
            center = max(set(cells), key=cells.count)
            near = [u for u, c in zip(left, cells) if cell_dist(c, center) <= 2]
            out.append(near)
            left = [u for u in left if u not in near]
        return out

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        g = self.g
        hi = g.hi
        if self.pull is not None and self.pull.step(obs, mine):
            return True
        if self._scouting(obs, mine):
            return True
        if hi.frame < self.next_ask:
            return False
        army = [u for u in done if u["type"] in g.army_types]
        clusters = self._clusters(army)
        if not clusters:
            return False
        self.next_ask = hi.frame + ASK_EVERY // len(clusters)       # one cluster per question
        group = clusters[self.ask_turn % len(clusters)]
        self.ask_turn += 1
        grid, glob = self.tracker.inputs(obs, group)
        where, kind = self.model.predict(grid, glob)
        cx = sum(u["x"] for u in group) / len(group)
        cy = sum(u["y"] for u in group) / len(group)
        here = self.tracker.cell(cx, cy)
        # move or stay: by how much of the model's belief lies away from here (spread over many
        # cells, "elsewhere" can be likelier than "here" even when "here" is the likeliest cell)
        away = np.array([cell_dist(c, here) >= MOVE_CELLS for c in range(GRID * GRID)])
        p_away = float(where[0, away].sum())
        if p_away < MOVE_P:
            self.proposed.pop(here, None)
            return False                                          # stay
        best = int(np.where(away, where[0], -1).argmax())          # where, if moving
        # move or attack-move as the model says, but never walk (rather than fight) into where
        # enemy buildings were seen. (Always attack-moving was tried: the model's plain moves are
        # often regrouping or pulling back, and stopping to fight there lost more games.)
        attack = int(kind[0].argmax()) == 1
        near_enemy = self.tracker.seen_buildings.reshape(GRID, GRID)[
            max(0, best // GRID - 2):best // GRID + 3, max(0, best % GRID - 2):best % GRID + 3].any()
        attack = attack or bool(near_enemy)
        proposed, self.proposed[here] = self.proposed.get(here), best
        if proposed is None or cell_dist(proposed, best) > 1:
            return False                                          # wait until it says so twice
        last = self.ordered.get(here)
        if last and cell_dist(last[0], best) <= 1 and last[1] == attack and hi.frame - last[2] < REORDER_AFTER:
            return False
        # will it go well? the model's fight estimate for going there vs staying put
        go = self.model.fight(grid, glob, best, attack)
        stay = self.model.fight(grid, glob, here, False)
        if go < max(FIGHT_MIN, stay - FIGHT_MARGIN):
            if self.held.get(here, -1) != best:
                g.say(f"army ({len(group)} units) holds: going to {self.tracker.center(best)} "
                      f"looks like {go:+.0%} army share, staying {stay:+.0%}")
            self.held[here] = best
            return False
        self.held.pop(here, None)
        x, y = self.tracker.center(best)
        self.ordered[here] = (best, attack, hi.frame)
        g.task = Task("army_order", data={"tags": [u["tag"] for u in group], "x": x, "y": y,
                                          "attack": attack, "p": p_away, "fight": go, "done": [], "boxes": 0},
                      started=hi.frame)
        return True

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
        pts = [screen_of(obs, u["x"], u["y"]) for u in here]
        hi.box(max(0, min(p[0] for p in pts) - 10), max(0, min(p[1] for p in pts) - 10),
               min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + 10),
               min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + 10))
        d["boxes"] += 1
        t.stage = "selected"

    # --- scouting (scripted for now) -----------------------------------------------------------

    def _scouting(self, obs: dict, mine: list[dict]) -> bool:
        """Like a pro, send a worker to find the enemy base early (the army model needs to know
        where the enemy is): the enemy start locations in turn, then home once it has seen an
        enemy building. True if Gary acted."""
        if self.scout_done:
            return False
        g = self.g
        hi = g.hi
        if g.scout is None:
            workers = sum(1 for u in mine if u["type"] == T.SCV)
            if hi.frame >= SCOUT_AT and workers >= SCOUT_WORKERS:
                self.scout_targets = sorted(g.enemy_starts, key=lambda p: math.dist(p, g.main.center))
                g.task = Task("scout", data={"to": self.scout_targets[0]}, started=hi.frame)
                return True
            return False
        scout = next((u for u in mine if u["tag"] == g.scout), None)
        if scout is None:                                  # the scout died
            self.scout_done = True
            return False
        if any(u["owner"] not in (g.slot, 11) and is_building(u["type"]) for u in obs["units"]):
            g.task = Task("scout", data={"to": g.main.minerals[0], "home": True}, started=hi.frame)
            return True                                    # found them: home again
        if self.scout_targets and math.dist((scout["x"], scout["y"]), self.scout_targets[0]) < 6 * 32:
            self.scout_targets.pop(0)                      # nobody here: try the next start
            if self.scout_targets:
                g.task = Task("scout", data={"to": self.scout_targets[0]}, started=hi.frame)
                return True
            self.scout_done = True
        return False

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
        if d.get("home"):
            self.scout_done = True
            g.say("scout found the enemy base, heading home")
        else:
            g.say(f"scouting {d['to']}")
        g.task = None
