"""A mirror game: Gary against Gary in the same race, headless (docs/RACES.md step 1).

Zerg today: two Gary-Z (gary/bots/gary_z.py) with the ZvZ models (runs/macro/ZvZ_Z_*, runs/army/ZvZ_Z_*,
train it with --matchup ZvZ --race Z). Both sides' POVs and the shared decision records are saved
next to the replay (eval/why.py reads them). Prints who is ahead every 30 s and the result.

Each side is Gary-Z ("garyz") or the one Gary from a manifest (e.g. "gz1", docs/REFACTOR.md): --sides A B.

Usage:
  python -m eval.mirror --map tests/fixtures/replays/stardata_tvz_standard_ozp3w.rep --seed 5001 --minutes 20
  python -m eval.mirror --map <map> --seed 5001 --minutes 4 --sides gz1 garyz
  python -m eval.mirror --map "<your StarCraft folder>/Maps/(4)Fighting Spirit.scx" --seed 5001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gary.bots.zerg_rush import HATCH  # noqa: E402
from gary.decision_log import LOG  # noqa: E402
from gary.env import LIVE_COMMAND_DELAY, Game  # noqa: E402
from gary.interface import PROFILES, HumanInterface  # noqa: E402
from gary.mapinfo import MapInfo  # noqa: E402

FIGHTERS = {37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 50, 62, 103}   # zerg fighting units (no drones, overlords, larvae)


def play(map_path: str, minutes: float, seed: int, save: str, matchup: str = "ZvZ",
         sides: tuple[str, str] = ("garyz", "garyz")) -> str:
    race = matchup[0]
    with Game.new(map_path, [race, race], ["Gary A", "Gary B"], seed=seed, command_delay=LIVE_COMMAND_DELAY) as game:
        mapinfo = MapInfo.from_game(game)
        obs = game.observe()
        halls = [u for u in obs["units"] if u["type"] == HATCH]
        slots = sorted(u["owner"] for u in halls)
        his = {s: HumanInterface(game, s, PROFILES["pro"], seed=s) for s in slots}
        LOG.clear_records()
        bots = {}
        from gary.bots.gary_z import GaryZ
        for s, side in zip(slots, sides):
            his[s].side = race
            if side == "garyz":
                bot = GaryZ(his[s], mapinfo, seed, matchup=matchup)
                other = next(u for u in halls if u["owner"] != s)
                bot.target = (other["x"], other["y"])         # (where the scripted waves would go: the enemy main)
                bot.attack_after = 0
            else:
                from gary.agent.manifests import build_gary
                LOG.slot = s
                bot = build_gary(his[s], mapinfo, side, seed=seed)
                LOG.slot = None
            bots[s] = bot
        end = int(minutes * 60 * 24)
        result = "time"
        first = slots[0]
        while his[first].frame < end:
            players = {p["slot"] for p in game.observe()["players"]}
            if not set(slots) <= players:
                alive = [s for s in slots if s in players]
                result = f"{'A' if alive == [slots[0]] else 'B'} won"
                break
            for s in slots:
                LOG.slot = s                          # (the records say whose they are: eval/why.py --slot)
                bots[s].act()
            LOG.slot = None
            for i, (s, hi) in enumerate(his.items()):
                LOG.slot = s                          # (the orders the hands carry out are recorded here too)
                hi.step(2, advance_game=(i == 0))
            LOG.slot = None
            if his[first].frame % (24 * 30) < 2:
                o = game.observe()
                t = his[first].frame // 24
                parts = []
                for name, s in zip("AB", slots):
                    mine = [u for u in o["units"] if u["owner"] == s]
                    buildings = sum(1 for u in mine if 106 <= u["type"] <= 175)
                    workers = sum(1 for u in mine if u["type"] == 41)
                    army = sum(1 for u in mine if u["type"] in FIGHTERS)
                    bank = next(p for p in o["players"] if p["slot"] == s)
                    parts.append(f"{name}: buildings {buildings} drones {workers} army {army} bank {bank['minerals']}/{bank['gas']}")
                print(f"{t // 60}:{t % 60:02d}  " + "   ".join(parts), flush=True)
                dead = [name for name, s in zip("AB", slots) if not any(u["owner"] == s and 106 <= u["type"] <= 175 for u in o["units"])]
                if dead:
                    result = f"{'B' if dead == ['A'] else 'A'} won"
                    break
        game.save_replay(save)
        base = save[:-4] if save.endswith(".rep") else save
        for name, s in zip("ab", slots):
            his[s].save_pov(f"{base}.{name}pov.jsonl")
        LOG.save_records(base + ".records.jsonl")
        for name, s in zip("ab", slots):
            if hasattr(bots[s], "save_trace"):
                bots[s].save_trace(f"{base}.{name}trace.jsonl")
        print(f"{result}; saved {save} (POVs and records beside it)")
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--seed", type=int, default=5001)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--matchup", default="ZvZ", choices=["ZvZ"])
    ap.add_argument("--save", default="mirror.rep")
    ap.add_argument("--sides", nargs=2, default=["garyz", "garyz"], metavar=("A", "B"),
                    help="garyz, or a manifest name (gary/agent/manifests.py) for the one Gary")
    args = ap.parse_args()
    play(args.map, args.minutes, args.seed, args.save, args.matchup, tuple(args.sides))


if __name__ == "__main__":
    main()
