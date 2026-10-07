"""A test fixture, not a player: a Zerg whose army appears when a real pro Zerg made it.

Scripting a Zerg that produces pro pressure (gary/bots/zerg_v2.py) costs more than it teaches: its
drones, larvae and build orders through a human interface lag a pro's by minutes. This one skips
production. Each seed picks a pro Zerg's build order from a TvZ game
(data/interim/build_orders.jsonl), and the env creates (Game.create_unit, OpenBW) what that
Zerg ordered, when it would have been done: zerglings in pairs, hydralisks and mutalisks after
their egg time at the main hatchery, zergling speed when its research would end; the Spawning
Pool, Extractor, Hydralisk Den and Spire near the main, hatcheries at the natural and then the
nearest free bases, all when ordered (finished at once: what Gary's scout sees is close enough). Its army goes as the dummy's does
(gary/bots/zerg_rush.py): attack-moving at Gary's start as it hatches, the first 6 together.

So Gary meets the spread of pro openings in their real proportions (pool first 30%, the rest
hatch first) at their real timings. It has no economy: killing its drones doesn't slow it, and it
never pays for anything. Use it to test whether Gary holds pressure, never to judge Gary's
economic choices, and only in the headless env (it creates units).

    python -m eval.vs_rush --versions v012 --opponent replayed --seeds 1 2 3 4
"""

from __future__ import annotations

import json
import math
import random

from gary import commands as C
from gary.bots.zerg_rush import HATCH, ZergRush
from gary.interface import screen_of

UNITS = {"Zergling": (37, 2), "Hydralisk": (38, 1), "Mutalisk": (43, 1)}
BUILDINGS = {"Spawning Pool": 142, "Extractor": 149, "Hydralisk Den": 135, "Spire": 141, "Hatchery": HATCH}
ARMY = {37, 38, 43}
SPEED, SPEED_FRAMES = 27, 1500    # Metabolic Boost: its upgrade id and research time
UNTIL = 12 * 60 * 24              # what the pro made after this isn't spawned


def pro_build_orders():
    """The pro Zergs of TvZ games of 5+ minutes (one record per game)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ingest"))
    from inventory import data_root
    out = []
    for line in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["matchup"] == "TvZ" and r["race"] == "Z" and r.get("duration_s", 0) >= 300:
            out.append(r)
    return out


class ZergReplayed(ZergRush):
    def __init__(self, hi, mapinfo, seed: int):
        super().__init__(hi, mapinfo)
        orders = pro_build_orders()
        self.source = random.Random(seed).choice(orders)
        game = hi.game
        frames = lambda t: game.unit_type_info(t)["build_frames"]
        self.schedule = []                                   # (frame, unit type, count, kind)
        for it in self.source["items"]:
            f = int(it["t"] * 24)
            if f > UNTIL or it.get("conf") in ("cancelled?", "not built"):
                continue
            if it["name"] in UNITS:                          # (lurkers stay hydralisks: no burrow micro here)
                t, n = UNITS[it["name"]]
                self.schedule.append((f + frames(t), t, n, "unit"))
            elif it["name"] in BUILDINGS:
                self.schedule.append((f, BUILDINGS[it["name"]], 1, "building"))
            elif it["name"].startswith("Metabolic Boost"):
                self.schedule.append((f + SPEED_FRAMES, SPEED, 1, "upgrade"))
        self.schedule.sort()
        first = {}
        for f, t, n, b in self.schedule:
            first.setdefault(t, f)
        pool, hatch = first.get(142, 10 ** 9), first.get(HATCH, 10 ** 9)
        self.opening = "pool first" if pool < hatch else "hatch first"
        print(f"Zerg replays {self.source['player']} ({self.source['rel_path'].split('/')[-1]}): {self.opening}, "
              f"pool at {pool // 24 // 60}:{pool // 24 % 60:02d}", flush=True)
        main = mapinfo.base_near(self.home)
        natural = mapinfo.natural_of(main)
        self.bases = [natural] + sorted((b for b in mapinfo.bases if b not in (main, natural)),
                                        key=lambda b: math.dist(b.center, main.center))
        d = math.dist(self.home, self.rally) or 1               # units hatch on the open side, off the minerals
        self.spawn_at = (round(self.home[0] + (self.rally[0] - self.home[0]) * 80 / d),
                         round(self.home[1] + (self.rally[1] - self.home[1]) * 80 / d))

    def act(self) -> None:
        hi = self.hi
        while self.schedule and self.schedule[0][0] <= hi.frame:
            _, t, n, kind = self.schedule.pop(0)
            if kind == "upgrade":
                hi.game.set_upgrade(hi.slot, t)
                continue
            for _ in range(n):
                self._spawn(t, kind == "building")
        if hi.pending:
            return
        obs = hi.observe()
        mine = [u for u in obs["units"] if u["owner"] == hi.slot]
        if not screen_of(obs, *self.home) and self.stage != "attack":
            hi.camera_minimap(*self.home)
            return
        self.army(obs, mine)

    def army(self, obs: dict, mine: list[dict]) -> bool:
        """The dummy's waves, with every army unit (not only zerglings)."""
        from gary.bots.zerg_rush import FIRST_WAVE, LATER_WAVES
        hi = self.hi
        army = [u for u in mine if u["type"] in ARMY and u["completed"] and u["tag"] not in self.sent]
        if len(army) >= (FIRST_WAVE if self.waves == 0 else LATER_WAVES) and hi.frame >= self.attack_after:
            here = [u for u in army if screen_of(obs, u["x"], u["y"])]
            if self.stage == "attack":
                hi.minimap_command(C.ORDER_ATTACK_MOVE, *self.target)
                self.sent.update(obs["selection"])
                self.waves += 1
                self.stage = ""
                return True
            if here:
                self._box(obs, here)
                self.stage = "attack"
                return True
        return False

    def _spawn(self, unit_type: int, building: bool) -> None:
        game, slot = self.hi.game, self.hi.slot
        if not building:
            game.create_unit(slot, unit_type, *self.spawn_at)
            return
        if unit_type == HATCH:                               # an expansion: the nearest free base
            taken = {(u["x"], u["y"]) for u in game.observe()["units"] if u["type"] in (106, 131, 132, 133)}
            for b in self.bases:
                if not any(math.dist(b.center, p) < 4 * 32 for p in taken):
                    if game.create_unit(slot, HATCH, *b.center):
                        return
            return
        info = game.unit_type_info(unit_type)
        hx, hy = self.home[0] // 32, self.home[1] // 32
        for r in range(3, 12):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r and game.can_place(slot, unit_type, hx + dx, hy + dy, 0):
                        x, y = (hx + dx) * 32 + info["w"] // 2, (hy + dy) * 32 + info["h"] // 2
                        if game.create_unit(slot, unit_type, x, y):
                            return
