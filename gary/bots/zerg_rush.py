"""A scripted Zerg sparring dummy: 9 pool, then zerglings sent as they hatch (the first 6 together).

Its timings follow pro early pools (data/interim/build_orders.jsonl, TvZ, pool before the
hatchery: pool ordered at 1:17, zerglings at 2:02-2:10): the drone sets out at 150 minerals so the
pool goes down at 200, and zerglings go as they hatch (it used to wait for 12 and arrive about a
minute later than a real 9-pool's).

It exists to test Gary (does it defend? does it hold its ramp, pull back, counter?). Like Gary it
plays through the human interface (fog of war, mouse and camera, APM), so neither side cheats.

Its timing should depend on the seed, not on how Gary plays: once the pool is down it rallies the
hatchery away from the larvae, and it picks a larva with nothing else around it. (New zerglings
used to stand on the larvae, so the tiny box around a larva caught a zergling too, and the dummy
kept reselecting: up to 50 s without a zergling, depending on where larvae happened to spawn.)

    python -m gary.bots.zerg_rush --vs v03 --map path/to/map.scx --minutes 10
"""

from __future__ import annotations

import argparse
import math

from gary import commands as C
from gary.agent.manifests import manifest_name
from gary.env import LIVE_COMMAND_DELAY, Game
from gary.interface import PROFILES, HumanInterface, screen_of
from gary.mapinfo import MapInfo

POOL, LING, OVERLORD, DRONE, HATCH = 142, 37, 42, 41, 131
MINING = 87                     # order: mining minerals (the drone stands at the field)
DRONES = 9
WAVE = 12                       # (Zerg v1's groups)
FIRST_WAVE, LATER_WAVES = 6, 2   # the dummy: the first 6 zerglings together, then each pair as it hatches
POOL_WALK_AT = 150              # minerals when the pool's drone sets out (the pool goes down at 200)
RALLY = 5 * 32                  # the hatchery's rally: this far from it, toward the map's middle


class ZergRush:
    def __init__(self, hi: HumanInterface, mapinfo: MapInfo):
        self.hi = hi
        self.map = mapinfo
        obs = hi.observe()
        hatch = next(u for u in obs["units"] if u["owner"] == hi.slot and u["type"] == HATCH)
        self.home = (hatch["x"], hatch["y"])
        self.hatch = hatch["tag"]
        self.target = min((s for s in mapinfo.starts if math.dist(s, self.home) > 10 * 32),
                          key=lambda s: math.dist(s, self.home))
        self.mining: set[int] = set()
        self.pool_ordered = 0
        self.sent: set[int] = set()
        self.waves = 0
        self.stage = ""
        self.overlord_at = -10**9
        self.attack_after = 0                 # frame of the first allowed attack
        self.rallied = False
        w, h = mapinfo.size
        d = math.dist(self.home, (w / 2, h / 2)) or 1
        self.rally = (round(self.home[0] + (w / 2 - self.home[0]) / d * RALLY),
                      round(self.home[1] + (h / 2 - self.home[1]) / d * RALLY))

    def act(self) -> None:
        hi = self.hi
        if hi.pending:
            return
        obs = hi.observe()
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == hi.slot]
        count = lambda t: sum(1 for u in mine if u["type"] == t)
        larva = [u for u in mine if u["type"] == C.LARVA and screen_of(obs, u["x"], u["y"])]
        pool = [u for u in mine if u["type"] == POOL]
        if not screen_of(obs, *self.home) and self.stage != "attack":
            hi.camera_minimap(*self.home)
            return

        # new drones mine (a small box: clicking a drone on a mineral field can pick the field)
        idle = [u for u in mine if u["type"] == DRONE and u["completed"] and u["tag"] not in self.mining
                and screen_of(obs, u["x"], u["y"])]
        if idle:
            u = idle[0]
            idle_tags = {s["tag"] for s in idle}
            sel = [s for s in mine if s["tag"] in obs["selection"] and s["tag"] in idle_tags]
            if not sel:                             # (larvae caught in the box just stay put)
                x, y = screen_of(obs, u["x"], u["y"])
                hi.box(max(0, x - 12), max(0, y - 12), x + 12, y + 12)   # idle drones clump: take them all
                if self.stage == "box_drone":
                    self.mining.add(u["tag"])       # couldn't select it: leave it
                self.stage = "box_drone"
                return
            self.stage = ""
            fields = [m for b in self.map.bases for m in b.minerals if screen_of(obs, *m)]
            if fields:
                m = min(fields, key=lambda m: math.dist(m, (u["x"], u["y"])))
                hi.right_click(*screen_of(obs, *m))
            self.mining.update(s["tag"] for s in sel)
            return

        # the pool, with a drone, at 9 supply
        if not pool and me["minerals"] >= POOL_WALK_AT and count(DRONE) >= DRONES and hi.frame - self.pool_ordered > 24 * 15:
            drones = [u for u in mine if u["type"] == DRONE and u["completed"] and screen_of(obs, u["x"], u["y"])]
            sel = [u for u in mine if u["tag"] in obs["selection"]]
            one_drone = len(sel) == 1 and sel[0]["type"] == DRONE
            spot = self._pool_spot(sel[0]["tag"] if one_drone else 0)
            if drones and spot:
                if len(sel) != 1 or sel[0]["type"] != DRONE:     # any single drone will do
                    # one that stands still, mining (what Gary sees is a moment old: a walking
                    # drone isn't where it was)
                    still = [u for u in drones if u["order"] == MINING] or drones
                    d = still[(hi.frame // 48) % len(still)]
                    x, y = screen_of(obs, d["x"], d["y"])
                    hi.box(max(0, x - 2), max(0, y - 2), x + 2, y + 2)
                    return
                d = sel[0]
                p = screen_of(obs, spot[0] * 32 + 16, spot[1] * 32 + 16)
                if not p:
                    return
                if self.stage != "pool_walk":
                    # a drone busy mining ignores a build order: walk it toward the spot first
                    hi.right_click(*p)
                    self.stage = "pool_walk"
                    return
                if me["minerals"] < 200:                  # at the spot early, the pool as soon as there's money
                    return
                self.stage = ""
                if hi.build(POOL, *p, order=C.ORDER_DRONE_START_BUILD).accepted:
                    self.pool_ordered = hi.frame      # (the drone stays off the idle list: it becomes the pool)
            return

        if self.army(obs, mine):
            return

        # once the pool is down: the hatchery's rally away from the larvae (click the hatchery's top,
        # above the larvae, then right-click the rally spot)
        if pool and not self.rallied:
            p = screen_of(obs, *self.rally)
            if obs["selection"] == [self.hatch] and p:
                hi.right_click(*p)
                self.rallied = True
                return
            top = screen_of(obs, self.home[0], self.home[1] - 24)
            if top:
                hi.click(*top)
                return

        # larvae: overlords when supply runs short, drones to 9, then zerglings
        if not larva or me["minerals"] < 50:
            return
        want = None
        if me["supply_max"] - me["supply_used"] <= 1 and me["supply_max"] < 200 and me["minerals"] >= 100 \
                and hi.frame - self.overlord_at > 24 * 25:      # one overlord at a time (25 s to hatch)
            want = OVERLORD
        elif count(DRONE) < DRONES:
            want = DRONE
        elif pool and pool[0]["completed"]:
            want = LING
        if want is None or me["supply_used"] + (0 if want == OVERLORD else 1) > me["supply_max"]:
            return
        # a click on a larva usually picks the hatchery behind it: drag a tiny box around it instead
        sel = [u for u in mine if u["tag"] in obs["selection"]]
        if not sel or any(u["type"] != C.LARVA for u in sel):
            clear = [lv for lv in larva if not any(u["type"] != C.LARVA and abs(u["x"] - lv["x"]) < 12
                                                   and abs(u["y"] - lv["y"]) < 12 for u in mine)]
            pick = clear or larva
            lv = pick[(hi.frame // 48) % len(pick)]
            x, y = screen_of(obs, lv["x"], lv["y"])
            hi.box(max(0, x - 3), max(0, y - 3), x + 3, y + 3)
            return
        if hi.morph(want).accepted and want == OVERLORD:
            self.overlord_at = hi.frame

    def army(self, obs: dict, mine: list[dict]) -> bool:
        """Waves of zerglings to the enemy start (attack-moved there, 12 at a time). True if it acted."""
        hi = self.hi
        lings = [u for u in mine if u["type"] == LING and u["completed"] and u["tag"] not in self.sent]
        if len(lings) >= (FIRST_WAVE if self.waves == 0 else LATER_WAVES) and hi.frame >= self.attack_after:
            here = [u for u in lings if screen_of(obs, u["x"], u["y"])]
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

    def _box(self, obs: dict, units: list[dict]) -> None:
        pts = [screen_of(obs, u["x"], u["y"]) for u in units]
        self.hi.box(max(0, min(p[0] for p in pts) - 8), max(0, min(p[1] for p in pts) - 8),
                    min(self.hi.p.viewport[0] - 1, max(p[0] for p in pts) + 8),
                    min(self.hi.p.viewport[1] - 1, max(p[1] for p in pts) + 8))

    def _pool_spot(self, builder: int = 0) -> tuple[int, int] | None:
        hx, hy = self.home[0] // 32, self.home[1] // 32
        for r in range(3, 9):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    if self.hi.game.can_place(self.hi.slot, POOL, hx + dx, hy + dy, builder):
                        return hx + dx, hy + dy
        return None


def play_match(map_path: str, minutes: float, seed: int | None, save: str, make_terran, name: str,
               attack_after_s: float = 0, opponent: str = "dummy") -> None:
    """Gary (Terran, made by make_terran(hi, mapinfo)) against ZergRush on a headless game."""
    with Game.new(map_path, ["T", "Z"], [name, "Zerg rush"], seed=seed, command_delay=LIVE_COMMAND_DELAY) as game:
        mapinfo = MapInfo.from_game(game)
        obs = game.observe()
        slot_of = {}
        for u in obs["units"]:
            if u["type"] == C.COMMAND_CENTER:
                slot_of["T"] = u["owner"]
            elif u["type"] == HATCH:
                slot_of["Z"] = u["owner"]
        # Gary with its pro-speed hands; the dummy keeps b_rank hands (the benchmark stays the same)
        his = {r: HumanInterface(game, s, PROFILES["pro" if r == "T" else "b_rank"], seed=s) for r, s in slot_of.items()}
        gary = make_terran(his["T"], mapinfo)
        if opponent == "replayed":
            from gary.bots.zerg_replayed import ZergReplayed
            zerg = ZergReplayed(his["Z"], mapinfo, seed or 0)
        elif opponent in ("v1", "v1a", "v1d", "v2"):
            from gary.bots import zerg_v1 as Z
            from gary.bots.zerg_v2 import ZergV2
            zerg = {"v1": Z.ZergV1, "v1a": Z.ZergV1Aggressive, "v1d": Z.ZergV1Direct, "v2": ZergV2}[opponent](his["Z"], mapinfo)
        else:
            zerg = ZergRush(his["Z"], mapinfo)
        # the dummy knows where Gary is (no scouting): its waves go straight to Gary's main
        zerg.attack_after = int(attack_after_s * 24)
        zerg.target = next((u["x"], u["y"]) for u in obs["units"] if u["type"] == C.COMMAND_CENTER)
        end = int(minutes * 60 * 24)
        result = "time"
        while his["T"].frame < end:
            players = {p["slot"] for p in game.observe()["players"]}
            if not set(slot_of.values()) <= players:          # someone was eliminated
                result = "Gary won" if slot_of["T"] in players else "Gary lost"
                break
            gary.act()
            zerg.act()
            for i, hi in enumerate(his.values()):
                hi.step(2, advance_game=(i == 0))
            if his["T"].frame % (24 * 30) < 2:
                o = game.observe()
                alive = {r: sum(1 for u in o["units"] if u["owner"] == s and 106 <= u["type"] <= 175)
                         for r, s in slot_of.items()}
                army = {r: sum(1 for u in o["units"] if u["owner"] == s and u["type"] in (0, 32, 34, LING, 38, 43))
                        for r, s in slot_of.items()}
                t = his["T"].frame // 24
                print(f"{t // 60}:{t % 60:02d}  buildings T {alive['T']} Z {alive['Z']}  "
                      f"army T {army['T']} Z {army['Z']}  waves sent {zerg.waves}", flush=True)
                if not alive["T"] or not alive["Z"]:
                    result = "Gary lost" if not alive["T"] else "Gary won"
                    break
        game.save_replay(save)
        pov = save[:-4] + ".pov.jsonl" if save.endswith(".rep") else save + ".pov.jsonl"
        his["T"].save_pov(pov)
        if hasattr(gary, "save_trace") and pov.endswith(".pov.jsonl"):
            gary.save_trace(pov[:-len(".pov.jsonl")] + ".trace.jsonl")
        print(f"{result}; saved {save} and {pov}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vs", default="v03", type=manifest_name, help="which Gary plays Terran")
    ap.add_argument("--map", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", default="gary_vs_rush.rep")
    ap.add_argument("--attack-after", type=float, default=0, help="no attack before this many seconds")
    ap.add_argument("--style", type=int, help="build style (taxonomy cluster) to steer Gary's macro toward")
    ap.add_argument("--opponent", default="dummy", choices=["dummy", "v1", "v1a", "v1d", "v2", "replayed"],
                    help="the Zerg: this dummy, or gary/bots/zerg_v1.py (gathers, judges, surrounds, retreats)")
    args = ap.parse_args()
    from gary.agent.manifests import build_gary
    make = lambda hi, m: build_gary(hi, m, args.vs, args.style)
    play_match(args.map, args.minutes, args.seed, args.save, make, f"Gary {args.vs}", args.attack_after, args.opponent)


if __name__ == "__main__":
    main()
