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
  fixed_priority/v2  v1, and a fight turn may also interrupt a build that is only waiting (its
                     worker walking to the base, or waiting for money with the worker selected):
                     the build is put aside and goes on when the fight leaves the hands free, the
                     same worker selected again. With v1 the fight waited for builds 43 times in
                     the 114 home defenses, up to 9.7 s
  drill/v1           micro drills (gary/drills.py): the fight layer, plus a chore and camera
                     interruptions so the fight doesn't get all of Gary's hands

The trace (Gary.trace, saved next to the POV log; python -m gary.agent.trace <file> sums it up):

  arbiter   a task started: who asked (granted_to: "macro", "army", "scout", "fight", "rule:pull",
            "executor:gas"...), the task, and the task a fight turn interrupted, if any
  task_end  a task ended: how long it held Gary's hands, whether it was interrupted
  wait      a layer wanted the hands while they were busy elsewhere, for 1 s or more: macro (a
            decision taken, not yet carried out) or the fight (one going on, the hands on a task
            the fight may not interrupt); with the tasks that held the hands meanwhile

Waits are read from what the layers already keep (the macro's goal, the fight's last snapshot):
the arbiter asks no layer more than it did before, so tracing changes nothing Gary does.
"""

from __future__ import annotations

import math
import random

from gary import terran as T
from gary.agent.fight import FIGHT_TASKS, INTERRUPTIBLE
from gary.agent.gary import Gary, Task


WAIT_MIN = 24                        # waits shorter than a second aren't traced


BUILD_WAITS = {"wait_arrival", "place"}   # v2: build stages a fight may interrupt (and resume)


class FixedPriority:
    def __init__(self, g: Gary, suspend_builds: bool = False):
        self.g = g
        self.suspend_builds = suspend_builds
        self.suspended = None            # v2: a build a fight turn put aside, to go on after
        self.interrupted = None          # the task the last fight turn dropped
        self.waits: dict[str, tuple[int, set[str]]] = {}   # layer -> (waiting since, tasks on the hands)
        self.wanted: dict[str, str | None] = {}            # layer -> what it was waiting to do

    def act(self) -> None:
        before = self.g.task
        self.interrupted = None
        who = self._act()
        self._trace(before, who)

    def _act(self) -> str | None:
        """One turn; who got the hands (or acted without a task), or None."""
        g = self.g
        hi = g.hi
        g.posture.update()
        if g.fight is not None:
            if g.macro_turn and g.task is None and not hi.pending:
                g.macro_turn = False             # fights don't starve production: this turn goes
                obs = hi.observe()               # to buildings and production, not to chores
                mine = [u for u in obs["units"] if u["owner"] == g.slot]   # (refilling gas, mining)
                g.executor.release_reservations(mine)
                if g.executor.resume_construction(mine):
                    return "executor:resume"
                if g.executor.resume_parked(obs, mine):
                    return "executor:parked"
                g.macro.step(obs, mine, [u for u in mine if u["completed"]], obs["me"])
                return "macro"
            if not hi.pending and g.setup_done and hi.frame >= g.fight.next_fight and self._interruptible(g.task):
                obs = hi.observe()
                g.army.remember(obs)
                mine = [u for u in obs["units"] if u["owner"] == g.slot]
                task = g.task
                g.task = None
                who = g.fight.block(obs, mine)
                if who:
                    self.interrupted = task
                    if task is not None and task.kind == "build":
                        task.data["resume"], task.stage = task.stage, "reselect"
                        self.suspended = task
                    return who
                g.task = task
        if g.army.remembers and not hi.pending:  # whatever Gary looks at, it remembers
            g.army.remember(hi.observe())
        if hi.pending:
            return None
        obs = hi.observe()
        if not g.setup_done:
            g.setup(obs)
            return "setup"
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == g.slot]
        done = [u for u in mine if u["completed"]]
        ex = g.executor
        ex.release_reservations(mine)
        if g.task is None and self.suspended is not None:     # v2: the build a fight put aside
            g.task, self.suspended = self.suspended, None
            g.run_task(obs, mine)
            return "macro:resumed"
        if ex.resume_parked(obs, mine):
            return "executor:parked"
        if g.task:
            g.run_task(obs, mine)
            return None
        for who, chore in (("executor:resume", lambda: ex.resume_construction(mine)),
                           ("executor:mine", lambda: ex.idle_mining(obs, done)),
                           ("executor:sweep", lambda: ex.idle_sweep(obs, done)),
                           ("executor:hotkey", lambda: ex.hotkey_new_buildings(done)),
                           ("executor:gas", lambda: ex.refill_gas(obs, done))):
            if chore():
                return who
        who = self._army(obs, mine, done)
        if who:
            return who
        g.macro.step(obs, mine, done, me)
        return "macro"

    def _interruptible(self, task) -> bool:
        return task is None or task.kind in INTERRUPTIBLE or \
            self.suspend_builds and task.kind == "build" and task.stage in BUILD_WAITS

    def _army(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        g = self.g
        if g.posture.army_home:          # the army model's moves leave the bunker empty: none now
            done = [u for u in done if u["type"] not in g.army_types]
        if g.fight is not None:
            who = g.fight.decide(obs, mine)
            if who:
                return who
            if g.fight.center is not None:   # the army layer leaves units in a fight alone
                cx, cy = g.fight.center
                done = [u for u in done if math.dist((u["x"], u["y"]), (cx, cy)) > 12 * 32]
        return g.army.step(obs, mine, done)

    # --- the trace -------------------------------------------------------------------------------

    def _trace(self, before, who: str | None) -> None:
        g = self.g
        now = g.hi.frame
        task = g.task
        if task is not before:
            if before is not None:
                g.event("task_end", task=before.kind, by=before.by, frames=now - before.started,
                        interrupted=before is self.interrupted)
            if task is not None:
                task.by = who or "?"
                g.event("arbiter", granted_to=task.by, task=task.kind,
                        interrupted=self.interrupted.kind if self.interrupted is not None else None)
        busy = task.kind if task is not None else None
        self._wait("macro", g.macro is not None and g.macro.goal is not None and busy is not None
                   and busy not in ("build", "command"), busy, now,
                   lambda: g.macro.goal["name"] if g.macro.goal else None)
        self._wait("fight", g.fight is not None and g.fight.center is not None and busy is not None
                   and busy not in FIGHT_TASKS and not self._interruptible(task), busy, now, lambda: None)

    def _wait(self, layer: str, waiting: bool, busy: str | None, now: int, what) -> None:
        if waiting:
            since, on = self.waits.get(layer, (now, set()))
            on.add(busy)
            if layer not in self.waits:
                self.waits[layer] = (since, on)
                self.wanted[layer] = what()
            return
        if layer in self.waits:
            since, on = self.waits.pop(layer)
            if now - since >= WAIT_MIN:
                self.g.event("wait", layer=layer, frames=now - since, hands_on=sorted(on),
                             wanted=self.wanted.get(layer))


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
