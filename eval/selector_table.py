"""The strategy selector's table (gary/agent/selector.py) from eval/vs_rush.py's games.

Each --add names a folder of games (vs_rush --out, or the temporary folder it prints), the style
Gary played them with ("none": unsteered) and the map: every game's result (won, lost, or going at
the time limit) is counted for that style on that map. --reset starts the table over.

    python -m eval.selector_table --add runs/x 1 tests/fixtures/replays/stardata_tvz_python_gg13676.rep
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path


def result_of(log: str) -> str:
    txt = Path(log).read_text(encoding="utf-8", errors="replace")
    return "won" if "Gary won" in txt else "lost" if "Gary lost" in txt else "time"


def main() -> None:
    from gary.agent.selector import StrategySelector, map_key
    from gary.env import Game
    from gary.mapinfo import MapInfo
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", nargs=3, action="append", metavar=("FOLDER", "STYLE", "MAP"), default=[])
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--versions", nargs="+", default=["v018", "v017+posture5"],
                    help="only games of these versions (a folder can hold several; v017+posture5 plays as v018)")
    args = ap.parse_args()
    sel = StrategySelector({} if args.reset else None)
    keys = {}
    for folder, style, mp in args.add:
        if mp not in keys:
            g = Game.new(mp, ["T", "Z"], seed=0)
            keys[mp] = map_key(MapInfo.from_game(g))
            g.close()
        option = None if style == "none" else int(style)
        logs = [f for v in args.versions for f in glob.glob(str(Path(folder) / f"{glob.escape(v)}_seed*.log"))]
        for f in logs:
            sel.add(keys[mp], option, result_of(f))
        print(f"{folder}: {len(logs)} games, style {style}, map {Path(mp).stem}")
    sel.save()
    for key, row in sel.table.items():
        print(key, {o: f"{c.get('won', 0)}/{c.get('lost', 0)}/{c.get('time', 0)}" for o, c in row.items()},
              "-> plays", sel.choose(key))


if __name__ == "__main__":
    main()
