"""Gary: one choice of provider per layer (docs/CONTROL.md), and what the layers share.

The layers (posture, macro, executor, army, fight, arbiter) are objects that each get this Gary
and use the shared state through it: the human interface, the map, the one task Gary's hands are
on, the bookkeeping more than one layer reads (reservations, hotkeys, workers taken off mining).
What only one layer uses stays in that layer. A manifest (gary/agent/manifests.py) says which
provider fills each layer.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from gary.interface import HumanInterface, screen_of
from gary.mapinfo import Base, MapInfo

LOC_MAIN, LOC_NATURAL = 2, 3          # F2, F3: saved screens


@dataclass
class Task:
    """What Gary's hands are doing, over several actions (select, then order...). One at a time."""
    kind: str                          # "build", "fight_cmd", "scout"... (each layer registers its kinds)
    unit_type: int = -1
    where: str | Base = "main"         # "main", "natural" or any Base
    stage: str = "start"
    data: dict = field(default_factory=dict)
    started: int = 0
    by: str = ""                       # which layer (or rule) asked for it: the arbiter's trace
    intent: object = None              # the intent it carries out (intents.py), if a skill's


class Gary:
    """Built by gary.agent.manifests.build_gary; act() once per step, between hi.step() calls."""

    def __init__(self, hi: HumanInterface, mapinfo: MapInfo | None, version: str, verbose: bool = True):
        self.hi = hi
        self.map = mapinfo
        self.slot = hi.slot
        self.version = version
        self.verbose = verbose
        self.task: Task | None = None
        self.task_runners: dict[str, Callable] = {}      # task kind -> runner(obs, mine, task)
        self.trace: list[dict] = []                      # trace events (ARCHITECTURE.md 6.5): event()
        self.army_types: set[int] = set()
        self.production_types: tuple[int, ...] = ()
        self.announce: list[str] = []                    # said in the chat at the start of the game
        self.setup_done = False
        # bookkeeping more than one layer reads
        self.rally: tuple[int, int] | None = None        # where new army units gather
        self.hotkeyed: dict[int, int] = {}               # building tag -> hotkey (-1: none)
        self.ordered_at: dict[int, int] = {}             # building tag -> frame of its last order
        self.reserved: list[tuple[int, int, int]] = []   # (unit type, tile x, tile y) about to be built
        self.reserved_at: dict[tuple[int, int, int], int] = {}
        self.producing: dict[int, int] = {}              # building tag -> unit type it last started
        self.researched: set[int] = set()
        self.upgraded: Counter = Counter()
        self.gas_workers: dict[int, set[int]] = {}       # refinery tag -> workers on it
        self.ignore_idle_until: dict[int, int] = {}      # workers (or refineries) to leave alone until then
        self.pulled: set[int] = set()                    # workers taken off mining to fight
        self.scout: int | None = None                    # the scouting worker
        self.loaded: dict[int, set[int]] = {}            # bunker tag -> marines Gary sent into it
        self.macro_turn = False                          # production's turn, after a fight command
        # When zerg fighters were last near a Command Center. One clock shared by the worker pull
        # (calm 5 s: pulled workers go back) and v0.9's posture (calm 20 s: the rush build ends),
        # as in v0.3-v0.9, where both used one attribute: each one's sightings restart the
        # other's calm.
        self.calm_since = 0
        self.main: Base | None = None
        self.natural: Base | None = None
        self.enemy_starts: list[tuple[int, int]] = []
        self.flip = (False, False)                       # mirror so the main is top-left (fight models)
        if mapinfo is not None:
            obs = hi.observe()
            cc = next(u for u in obs["units"] if u["owner"] == self.slot and u["type"] == 106)
            self.main = mapinfo.base_near((cc["x"], cc["y"]))
            self.natural = mapinfo.natural_of(self.main)
            self.enemy_starts = mapinfo.enemy_starts(self.main) or mapinfo.starts
            self.rally = self.toward_center(self.natural.center, 6 * 32)
            self.flip = (self.main.center[0] > mapinfo.size[0] / 2, self.main.center[1] > mapinfo.size[1] / 2)
        # the layers (set by build_gary)
        self.motor = None
        self.leases = None
        self.executor = None
        self.posture = None
        self.macro = None
        self.army = None
        self.fight = None
        self.arbiter = None

    def act(self) -> None:
        self.arbiter.act()

    def adopt(self) -> None:
        """For a game Gary takes over midway (eval/scenarios.py), as if it had played it so far: the
        hotkeys its production buildings would have, and the early scouting done (else a worker
        leaves for the enemy start at the takeover, minutes late, into the zerglings)."""
        self.executor.adopt_hotkeys()
        if hasattr(self.army, "scout_done"):
            self.army.scout_done = True

    # --- the trace --------------------------------------------------------------------------

    def event(self, kind: str, **payload) -> None:
        self.trace.append({"frame": self.hi.frame, "kind": kind, **payload})

    def save_trace(self, path) -> None:
        """The trace as JSON lines, next to the POV log (python -m gary.agent.trace <file> sums it up)."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"schema": "trace/v1", "gary": self.version, "slot": self.slot}) + "\n")
            for e in self.trace:
                f.write(json.dumps(e) + "\n")

    # --- tasks ------------------------------------------------------------------------------

    def run_task(self, obs: dict, mine: list[dict]) -> None:
        self.task_runners[self.task.kind](obs, mine, self.task)

    # --- what every layer uses --------------------------------------------------------------

    def say(self, text: str) -> None:
        if self.verbose:
            s = self.hi.frame // 24
            print(f"  {s // 60}:{s % 60:02d} Gary {text}", flush=True)

    def toward_center(self, p: tuple[int, int], d: int) -> tuple[int, int]:
        cx, cy = self.map.size[0] / 2, self.map.size[1] / 2
        n = math.dist(p, (cx, cy)) or 1
        return (round(p[0] + (cx - p[0]) * d / n), round(p[1] + (cy - p[1]) * d / n))

    def base(self, where: str | Base) -> Base:
        return self.main if where == "main" else self.natural if where == "natural" else where

    def look_at(self, obs: dict, where: str | Base) -> bool:
        """Make sure the base ("main", "natural" or any Base) is on screen. True if it already is
        (no action taken)."""
        base = self.base(where)
        if screen_of(obs, *base.center):
            return True
        loc = {"main": LOC_MAIN, "natural": LOC_NATURAL}.get(where) if isinstance(where, str) else None
        if loc in self.hi.camera_locations:
            self.hi.camera_location(loc)
        else:
            self.hi.camera_minimap(*base.center)
        return False

    def camera_to(self, x: int, y: int) -> None:
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

    def setup(self, obs: dict) -> None:
        """Say which Gary this is in the chat, then remember the main as F2."""
        if self.announce:
            self.hi.chat(self.announce.pop(0))
            return
        if not self.look_at(obs, "main"):
            return
        self.hi.camera_location_set(LOC_MAIN)
        self.setup_done = True
