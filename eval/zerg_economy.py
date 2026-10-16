"""The Zerg's economy in a batch of games (Gary-Z's, Zerg v3's): drones, hatcheries, army supply and
the bank at fixed minutes, per game and the median, from the replays (re-simulated in OpenBW).

    python -m eval.zerg_economy "runs/garyz_seeds/*.rep"
"""

from __future__ import annotations

import argparse
import glob
import statistics
from pathlib import Path

from gary import zerg as Z
from gary.env import Game

AT = (4, 6, 8, 10, 12)


def economy(rep: str, at=AT) -> dict[int, dict]:
    out: dict[int, dict] = {}
    g = Game.scenario(rep)
    try:
        slot = None
        for m in at:
            try:
                g.step(m * 1440 - g.frame)
            except Exception:
                break
            o = g.observe()
            if slot is None:
                slot = next((u["owner"] for u in o["units"] if u["type"] in Z.TOWN_HALLS), None)
            me = next((p for p in o["players"] if p["slot"] == slot), None)
            if me is None:
                break
            mine = [u for u in o["units"] if u["owner"] == slot]
            out[m] = {"drones": sum(1 for u in mine if u["type"] == Z.DRONE),
                      "hatcheries": sum(1 for u in mine if u["type"] in Z.TOWN_HALLS),
                      "army": sum(Z.SUPPLY.get(u["type"], 0) / (2 if u["type"] in Z.PAIRS else 1)
                                  for u in mine if u["type"] in Z.ARMY),
                      "minerals": me["minerals"], "gas": me["gas"]}
    finally:
        g.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replays", help="a glob of replays")
    args = ap.parse_args()
    rows = {Path(r).stem: economy(r) for r in sorted(glob.glob(args.replays))}
    print(f"{'game':24s}" + "".join(f"{m:>2d}:00 drones/hatch/army/bank      " for m in AT))
    for name, e in rows.items():
        print(f"{name:24s}" + "".join(
            f"{e[m]['drones']:>9d}/{e[m]['hatcheries']}/{e[m]['army']:4.0f}/{e[m]['minerals']:>4d}/{e[m]['gas']:<5d}  "
            if m in e else f"{'(gone)':>30s}  " for m in AT))
    for key in ("drones", "hatcheries", "army", "minerals", "gas"):
        meds = [statistics.median([e[m][key] for e in rows.values() if m in e]) if any(m in e for e in rows.values())
                else float("nan") for m in AT]
        print(f"median {key:10s}" + "".join(f"{v:>10.0f}" for v in meds))


if __name__ == "__main__":
    main()
