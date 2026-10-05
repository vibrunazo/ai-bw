"""The attention arbiter (layer L6, docs/CONTROL.md): which layer gets Gary's hands next.

One task at a time on one pair of hands: when a task runs, the arbiter lets it go on; otherwise it
asks the layers in a fixed order and the first that wants the hands gets them (it sets the task).

  fixed_priority/v1  Gary v0.2-v0.9's order (each act):
                     1. the posture looks at the game
                     2. with a fight layer: production's turn after each fight command; then, every
                        half second, a fight turn (rules, then the model) that may interrupt cheap
                        chores (mining, gas, hotkeys)
                     3. the running task; finishing stalled buildings; idle workers to mining; the
                        idle-worker sweep; hotkeys for new buildings; gas
                     4. the army: the fight model first (units in a fight are its), then the army
                        layer with the rest (none of it if the posture keeps the army home)
                     5. macro
  drill/v1           micro drills (gary/drills.py): the fight layer, plus a chore and camera
                     interruptions so the fight doesn't get all of Gary's hands
"""

from __future__ import annotations

import math
import random

from gary import terran as T
from gary.agent.fight import INTERRUPTIBLE
from gary.agent.gary import Gary, Task


class FixedPriority:
    def __init__(self, g: Gary):
        self.g = g

    def act(self) -> None:
        g = self.g
        hi = g.hi
        g.posture.update()
        if g.fight is not None:
            if g.macro_turn and g.task is None and not hi.pending:
                g.macro_turn = False             # fights don't starve production: this turn goes
                obs = hi.observe()               # to buildings and production, not to chores
                mine = [u for u in obs["units"] if u["owner"] == g.slot]   # (refilling gas, mining)
                g.executor.release_reservations(mine)
                if not g.executor.resume_construction(mine):
                    g.macro.step(obs, mine, [u for u in mine if u["completed"]], obs["me"])
                return
            if not hi.pending and g.setup_done and hi.frame >= g.fight.next_fight and \
                    (g.task is None or g.task.kind in INTERRUPTIBLE):
                obs = hi.observe()
                g.army.remember(obs)
                mine = [u for u in obs["units"] if u["owner"] == g.slot]
                task = g.task
                g.task = None
                if g.fight.block(obs, mine):
                    return
                g.task = task
        if g.army.remembers and not hi.pending:  # whatever Gary looks at, it remembers
            g.army.remember(hi.observe())
        if hi.pending:
            return
        obs = hi.observe()
        if not g.setup_done:
            return g.setup(obs)
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == g.slot]
        done = [u for u in mine if u["completed"]]
        ex = g.executor
        ex.release_reservations(mine)
        if g.task:
            return g.run_task(obs, mine)
        if ex.resume_construction(mine) or ex.idle_mining(obs, done) or ex.idle_sweep(obs, done) or \
                ex.hotkey_new_buildings(done) or ex.refill_gas(obs, done):
            return
        if self._army(obs, mine, done):
            return
        g.macro.step(obs, mine, done, me)

    def _army(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        g = self.g
        if g.posture.army_home:          # the army model's moves leave the bunker empty: none now
            done = [u for u in done if u["type"] not in g.army_types]
        if g.fight is not None:
            if g.fight.decide(obs, mine):
                return True
            if g.fight.center is not None:   # the army layer leaves units in a fight alone
                cx, cy = g.fight.center
                done = [u for u in done if math.dist((u["x"], u["y"]), (cx, cy)) > 12 * 32]
        return g.army.step(obs, mine, done)


HK_CC = 4
AWAY_EVERY = (5 * 24, 12 * 24)       # frames between interruptions
AWAY_FOR = (24, 72)                  # frames each lasts


class DrillArbiter:
    """The fight layer alone, for micro drills: no base, no macro or army. So that a drill isn't a
    world where the fight gets all of Gary's hands (a real game shares them with macro), Gary also
    has:
      - a chore: when the Command Center is idle and minerals allow, Gary checks it (its hotkey,
        which replaces the selection) and trains an SCV if supply allows, in the turn production
        gets after each fight command (drills have no depots, so it's mostly the check)
      - interruptions: every 5-12 s the camera goes elsewhere for 1-3 s (as if macroing far away)
    """

    def __init__(self, g: Gary, seed: int = 0, interruptions: bool = True):
        self.g = g
        hi = g.hi
        self.attention = random.Random(seed * 7919 + 1)
        self.interruptions = interruptions
        self.next_away = hi.frame + self.attention.randint(*AWAY_EVERY)
        cc = next((u for u in hi.observe()["units"] if u["owner"] == g.slot and u["type"] == T.CC), None)
        self.cc = cc["tag"] if cc else None
        if self.cc:
            hi.hotkeys[HK_CC] = [self.cc]        # as a player has their Command Center on a key
        g.task_runners["away"] = self._run_away
        g.task_runners["chore"] = self._run_chore

    def act(self) -> None:
        g = self.g
        hi = g.hi
        if hi.pending:
            return
        obs = hi.observe()
        mine = [u for u in obs["units"] if u["owner"] == g.slot]
        if g.task:
            return g.run_task(obs, mine)
        if self.interruptions and hi.frame >= self.next_away:
            self.next_away = hi.frame + self.attention.randint(*AWAY_EVERY)
            w, h = hi.map_size
            g.task = Task("away", data={"until": hi.frame + self.attention.randint(*AWAY_FOR),
                                        "to": (self.attention.randint(64, w - 64), self.attention.randint(64, h - 64))},
                          started=hi.frame)
            return g.run_task(obs, mine)
        if g.macro_turn:
            g.macro_turn = False
            cc = next((u for u in mine if u["tag"] == self.cc), None)
            if cc and cc["completed"] and not cc.get("queue") and obs["me"]["minerals"] >= 50:
                g.task = Task("chore", started=hi.frame)
                return g.run_task(obs, mine)
        g.fight.decide(obs, mine)

    def _run_away(self, obs: dict, mine: list[dict], t: Task) -> None:
        """The camera elsewhere, as if macroing far away."""
        hi = self.g.hi
        if t.stage == "start":
            hi.camera_minimap(*t.data["to"])
            t.stage = "there"
        elif hi.frame >= t.data["until"]:
            self.g.task = None

    def _run_chore(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Hotkey the Command Center, train an SCV if supply allows."""
        g, hi = self.g, self.g.hi
        me = obs["me"]
        if hi.frame - t.started > 24 * 3:
            g.task = None
        elif obs["selection"] != [self.cc]:
            hi.hotkey_recall(HK_CC)
        else:
            if me["supply_used"] < me["supply_max"]:
                hi.train(T.SCV)
            g.task = None
