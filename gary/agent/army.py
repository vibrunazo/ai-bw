"""The army (layer L3, docs/CONTROL.md): where groups go, and scouting.

  waves     every 16 army units attack-move to an enemy start (Gary v0.1-v0.2)
  learned   the army model (gary/policy/army.py, pro TvZ army orders; v0.3+): every 2 s it is asked
            about one of up to 3 clusters of the army; when it puts most of its belief on the
            cluster going 2+ grid cells away, says so twice, and its fight estimate doesn't expect
            going to cost army share, Gary drag-boxes the cluster and orders it on the minimap.
            Also the scout: one worker at 9 SCVs to the enemy start locations; where it finds the
            enemy, once around their main (7 tiles out: a Spawning Pool can stand out of sight
            of the hatchery), then home.

Units in a fight are the fight layer's; the arbiter leaves them out (and the posture can keep the
army home altogether). Orders are intents (ARMY_MOVE, ARMY_ATTACK, SCOUT, ATTACK_WAVE) that the
motor layer (motor.py) carries out.
"""

from __future__ import annotations

import math

import numpy as np

from gary import terran as T
from gary.agent.gary import Gary
from gary.agent.intents import Intent
from gary.policy.army import GRID, ArmyModel, ArmyTracker, is_building

WAVE = 16
ASK_EVERY = 24 * 2                  # how often Gary asks the model about its army
MOVE_CELLS = 2                      # destinations closer than this are "stay"
MOVE_P = 0.5                        # move when the model puts more belief than this away from here
FIGHT_MIN = -0.02                   # don't go if the fight estimate says we'd lose 2+ points of army share
FIGHT_MARGIN = 0.01                 # ...or if going looks worse than staying by more than 1 point
REORDER_AFTER = 24 * 8              # don't repeat the same order to a group within 8 s
SCOUT_AT = 24 * 45                  # send a scouting worker from 0:45, at 9 SCVs (like pros)
SCOUT_WORKERS = 9
CIRCLE = 7 * 32                     # the scout circles the enemy main this far from its start location


def cell_dist(a: int, b: int) -> int:
    return max(abs(a // GRID - b // GRID), abs(a % GRID - b % GRID))


class WaveArmy:
    remembers = False

    def __init__(self, g: Gary):
        self.g = g
        self.attacks_sent = 0
        self.army_sent: set[int] = set()

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        """Attack-move with every 16 army units. "army" if Gary acted."""
        g = self.g
        army = [u for u in done if u["type"] in g.army_types and u["tag"] not in self.army_sent]
        if len(army) >= WAVE:
            g.motor.start(Intent("ATTACK_WAVE", [], None, {}, "army"))
            return "army"
        return None



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
        self.circle: list | None = None                 # the scout's way around the enemy main, once found
        self.next_circle_order = 0

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

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        """Who acted ("rule:pull", "scout", "army"), or None."""
        g = self.g
        hi = g.hi
        if self.pull is not None and self.pull.step(obs, mine):
            return "rule:pull"
        if self._scouting(obs, mine):
            return "scout"
        if hi.frame < self.next_ask:
            return None
        army = [u for u in done if u["type"] in g.army_types]
        clusters = self._clusters(army)
        if not clusters:
            return None
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
            return None                                           # stay
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
            return None                                           # wait until it says so twice
        last = self.ordered.get(here)
        if last and cell_dist(last[0], best) <= 1 and last[1] == attack and hi.frame - last[2] < REORDER_AFTER:
            return None
        # will it go well? the model's fight estimate for going there vs staying put
        go = self.model.fight(grid, glob, best, attack)
        stay = self.model.fight(grid, glob, here, False)
        if go < max(FIGHT_MIN, stay - FIGHT_MARGIN):
            if self.held.get(here, -1) != best:
                g.say(f"army ({len(group)} units) holds: going to {self.tracker.center(best)} "
                      f"looks like {go:+.0%} army share, staying {stay:+.0%}")
            self.held[here] = best
            return None
        self.held.pop(here, None)
        x, y = self.tracker.center(best)
        self.ordered[here] = (best, attack, hi.frame)
        g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", [u["tag"] for u in group], (x, y),
                             {"tags": [u["tag"] for u in group], "x": x, "y": y, "attack": attack, "p": p_away,
                              "fight": go, "done": [], "boxes": 0}, "army"))
        return "army"

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
                g.motor.start(Intent("SCOUT", [], self.scout_targets[0], {"to": self.scout_targets[0]}, "scout"))
                return True
            return False
        scout = next((u for u in mine if u["tag"] == g.scout), None)
        if scout is None:                                  # the scout died
            self.scout_done = True
            return False
        if self.circle is None and any(u["owner"] not in (g.slot, 11) and is_building(u["type"]) for u in obs["units"]):
            # found them: around their main once (the pool can stand out of sight of the hatchery),
            # then home
            c = self.scout_targets[0] if self.scout_targets else (scout["x"], scout["y"])
            ring = [(c[0] + math.cos(k * math.pi / 3) * CIRCLE, c[1] + math.sin(k * math.pi / 3) * CIRCLE) for k in range(6)]
            k0 = min(range(6), key=lambda k: math.dist(ring[k], (scout["x"], scout["y"])))
            w, h = g.map.size
            self.circle = [(int(min(max(x, 32), w - 32)), int(min(max(y, 32), h - 32))) for x, y in ring[k0:] + ring[:k0]]
        if self.circle is not None:
            if self.circle and math.dist((scout["x"], scout["y"]), self.circle[0]) < 2 * 32:
                self.circle.pop(0)
            if self.circle and g.hi.frame >= self.next_circle_order:
                self.next_circle_order = g.hi.frame + 3 * 24          # (re-ordered while walking: it may be blocked)
                g.motor.start(Intent("SCOUT", [g.scout], self.circle[0], {"to": self.circle[0]}, "scout"))
                return True
            if not self.circle:
                self.circle = []
                g.motor.start(Intent("SCOUT", [g.scout], g.main.minerals[0], {"to": g.main.minerals[0], "home": True}, "scout"))
                self.scout_done = True
                return True
            return False
        if self.scout_targets and math.dist((scout["x"], scout["y"]), self.scout_targets[0]) < 6 * 32:
            self.scout_targets.pop(0)                      # nobody here: try the next start
            if self.scout_targets:
                g.motor.start(Intent("SCOUT", [g.scout], self.scout_targets[0], {"to": self.scout_targets[0]}, "scout"))
                return True
            self.scout_done = True
        return False
