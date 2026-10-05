"""A scripted Zerg sparring dummy: 9 pool, then zerglings, attacking in waves of 12.

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
from gary.env import LIVE_COMMAND_DELAY, Game
from gary.interface import PROFILES, HumanInterface, screen_of
from gary.mapinfo import MapInfo

POOL, LING, OVERLORD, DRONE, HATCH = 142, 37, 42, 41, 131
MINING = 87                     # order: mining minerals (the drone stands at the field)
DRONES = 9
WAVE = 12
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
        if not pool and me["minerals"] >= 200 and count(DRONE) >= DRONES and hi.frame - self.pool_ordered > 24 * 15:
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
                self.stage = ""
                if hi.build(POOL, *p, order=C.ORDER_DRONE_START_BUILD).accepted:
                    self.pool_ordered = hi.frame      # (the drone stays off the idle list: it becomes the pool)
            return

        # waves of zerglings to the enemy start
        lings = [u for u in mine if u["type"] == LING and u["completed"] and u["tag"] not in self.sent]
        if len(lings) >= WAVE and hi.frame >= self.attack_after:
            here = [u for u in lings if screen_of(obs, u["x"], u["y"])]
            if self.stage == "attack":
                hi.minimap_command(C.ORDER_ATTACK_MOVE, *self.target)
                self.sent.update(obs["selection"])
                self.waves += 1
                self.stage = ""
                return
            if here:
                pts = [screen_of(obs, u["x"], u["y"]) for u in here]
                hi.box(max(0, min(p[0] for p in pts) - 8), max(0, min(p[1] for p in pts) - 8),
                       min(hi.p.viewport[0] - 1, max(p[0] for p in pts) + 8),
                       min(hi.p.viewport[1] - 1, max(p[1] for p in pts) + 8))
                self.stage = "attack"
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
               attack_after_s: float = 0) -> None:
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
                army = {r: sum(1 for u in o["units"] if u["owner"] == s and u["type"] in (0, 32, 34, LING))
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
        print(f"{result}; saved {save} and {pov}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vs", default="v03", choices=["v01", "v02", "v03", "v04", "v05", "v06", "v07", "v08", "v09"], help="which Gary plays Terran")
    ap.add_argument("--map", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", default="gary_vs_rush.rep")
    ap.add_argument("--attack-after", type=float, default=0, help="no attack before this many seconds")
    ap.add_argument("--style", type=int, help="build style (taxonomy cluster) to steer Gary's macro toward")
    args = ap.parse_args()
    if args.vs == "v01":
        from gary.bots.terran_v01 import TerranGary as make
    elif args.vs == "v02":
        from gary.bots.terran_v02 import TerranGaryV2, latest_model
        from gary.policy.macro import MacroModel
        macro = MacroModel.load(latest_model())
        make = lambda hi, m: TerranGaryV2(hi, m, macro, args.style)
    elif args.vs == "v04":
        from gary.bots.terran_v02 import latest_model
        from gary.bots.terran_v03 import latest_army_model
        from gary.bots.terran_v04 import TerranGaryV4, latest_fight_model
        from gary.policy.army import ArmyModel
        from gary.policy.fight import FightModel
        from gary.policy.macro import MacroModel
        macro, army = MacroModel.load(latest_model()), ArmyModel.load(latest_army_model())
        fight = FightModel.load(latest_fight_model())
        make = lambda hi, m: TerranGaryV4(hi, m, macro, army, fight, args.style)
    elif args.vs in ("v08", "v09"):
        from gary.bots.terran_v02 import latest_model
        from gary.bots.terran_v03 import latest_army_model
        from gary.bots.terran_v08 import TerranGaryV8, latest_rl_fight_model
        from gary.bots.terran_v09 import TerranGaryV9
        from gary.policy.army import ArmyModel
        from gary.policy.fight_cmd import CommandModel
        from gary.policy.macro import MacroModel
        macro, army = MacroModel.load(latest_model()), ArmyModel.load(latest_army_model())
        command = CommandModel.load(latest_rl_fight_model())
        cls = TerranGaryV9 if args.vs == "v09" else TerranGaryV8
        make = lambda hi, m: cls(hi, m, macro, army, command, args.style)
    elif args.vs in ("v05", "v06", "v07"):
        from gary.bots.terran_v02 import latest_model
        from gary.bots.terran_v03 import latest_army_model
        from gary.bots.terran_v05 import TerranGaryV5, latest_command_model
        from gary.bots.terran_v06 import TerranGaryV6
        from gary.bots.terran_v07 import TerranGaryV7
        from gary.policy.army import ArmyModel
        from gary.policy.fight_cmd import CommandModel
        from gary.policy.macro import MacroModel
        macro, army = MacroModel.load(latest_model()), ArmyModel.load(latest_army_model())
        command = CommandModel.load(latest_command_model(memory=args.vs != "v05"))
        cls = {"v05": TerranGaryV5, "v06": TerranGaryV6, "v07": TerranGaryV7}[args.vs]
        make = lambda hi, m: cls(hi, m, macro, army, command, args.style)
    else:
        from gary.bots.terran_v02 import latest_model
        from gary.bots.terran_v03 import TerranGaryV3, latest_army_model
        from gary.policy.army import ArmyModel
        from gary.policy.macro import MacroModel
        macro, army = MacroModel.load(latest_model()), ArmyModel.load(latest_army_model())
        make = lambda hi, m: TerranGaryV3(hi, m, macro, army, args.style)
    play_match(args.map, args.minutes, args.seed, args.save, make, f"Gary {args.vs}", args.attack_after)


if __name__ == "__main__":
    main()
