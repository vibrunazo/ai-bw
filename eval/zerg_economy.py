"""The Zerg's economy in a batch of games (Gary-Z's, Zerg v3's): drones, hatcheries, army supply and
the bank at fixed minutes, per game and the median, from the replays (re-simulated in OpenBW); and
how long its hatcheries sat at the larva cap, next to the pros'.

  capped   the share of hatchery-time (sampled every 2 s, each larva counted at its nearest
           hatchery) at 3 larvae, the cap: a hatchery there makes no more until one is used. Pros
           keep it low with few hatcheries by using larvae as they come. High with money in the
           bank: the hands are slow (execution). Low with money in the bank and no larvae: too few
           hatcheries for the spending (a macro hatchery: a decision, learned)

    python -m eval.zerg_economy "runs/garyz_seeds/*.rep"
    python -m eval.zerg_economy "runs/garyz_seeds/*.rep" --pros 100     # next to 100 pro Zergs (cached)
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from gary import zerg as Z
from gary.env import Game

AT = (4, 6, 8, 10, 12)
SAMPLE = 48                      # frames between larva counts (2 s)
NEAR = 6 * 32                    # a larva this near a hatchery is that hatchery's
CAP = 3
KEYS = ("drones", "hatcheries", "army", "minerals", "gas", "capped")


def economy(rep: str, at=AT) -> dict[int, dict]:
    """{minute: drones, hatcheries, army, minerals, gas, and capped: the share of hatchery-time at
    the larva cap since the previous minute in `at`}."""
    out: dict[int, dict] = {}
    g = Game.scenario(rep)
    try:
        slot, prev = None, 0
        for m in at:
            halls = capped = 0
            try:
                while g.frame + SAMPLE <= m * 1440:
                    g.step(SAMPLE)
                    o = g.observe()
                    if slot is None:
                        slot = next((u["owner"] for u in o["units"] if u["type"] in Z.TOWN_HALLS), None)
                    if g.frame <= prev * 1440:
                        continue
                    mine = [u for u in o["units"] if u["owner"] == slot]
                    hs = [u for u in mine if u["type"] in Z.TOWN_HALLS and u["completed"]]
                    larvae = [0] * len(hs)
                    for u in mine:
                        if u["type"] == Z.LARVA and hs:
                            k, d = min(((k, (h["x"] - u["x"]) ** 2 + (h["y"] - u["y"]) ** 2) for k, h in enumerate(hs)),
                                       key=lambda kd: kd[1])
                            if d <= NEAR ** 2:
                                larvae[k] += 1
                    halls += len(hs)
                    capped += sum(1 for n in larvae if n >= CAP)
                if m * 1440 > g.frame:
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
                      "minerals": me["minerals"], "gas": me["gas"],
                      "capped": capped / halls if halls else 0.0}
            prev = m
    finally:
        g.close()
    return out


def _pro_game(path: str) -> dict | None:
    try:
        e = economy(path)
        return e if max(e, default=0) == AT[-1] else None
    except Exception:
        return None


def pro_reference(games: int) -> dict:
    """{minute: {key: median}} over pro TvZ Zergs in games of 12:30 or more that play the same in
    OpenBW (cached)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    from inventory import data_root
    cache = data_root() / "interim" / f"pro_zerg_economy_{games}.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    recs = [json.loads(line) for line in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
    from eval.humanlike import clean_replays
    clean = clean_replays()
    recs = [r for r in recs if r["matchup"] == "TvZ" and r["race"] == "Z" and r.get("duration_s", 0) >= 750
            and r["rel_path"] in clean]
    random.Random(0).shuffle(recs)
    with ProcessPoolExecutor(4) as pool:
        found = [g for g in pool.map(_pro_game, [str(data_root() / "raw" / r["rel_path"]) for r in recs[:games]]) if g]
    ref = {m: {k: statistics.median(g[m][k] for g in found) for k in KEYS} | {"games": len(found)} for m in AT}
    cache.write_text(json.dumps(ref))
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replays", help="a glob of replays")
    ap.add_argument("--pros", type=int, default=0, help="also the medians of this many pro Zergs (TvZ)")
    args = ap.parse_args()
    rows = {Path(r).stem: economy(r) for r in sorted(glob.glob(args.replays))}
    print(f"{'game':24s}" + "".join(f"{m:>2d}:00 drones/hatch/army/bank/capped  " for m in AT))
    for name, e in rows.items():
        print(f"{name:24s}" + "".join(
            f"{e[m]['drones']:>7d}/{e[m]['hatcheries']}/{e[m]['army']:3.0f}/{e[m]['minerals']:>4d}/{e[m]['gas']:<4d}/{e[m]['capped']:4.0%}  "
            if m in e else f"{'(gone)':>34s}  " for m in AT))
    ref = pro_reference(args.pros) if args.pros else None
    for key in KEYS:
        meds = [statistics.median([e[m][key] for e in rows.values() if m in e]) if any(m in e for e in rows.values())
                else float("nan") for m in AT]
        fmt = (lambda v: f"{v:>10.0%}") if key == "capped" else (lambda v: f"{v:>10.0f}")
        print(f"median {key:10s}" + "".join(fmt(v) for v in meds))
        if ref:
            print(f"  pros {key:10s}" + "".join(fmt(ref[m][key]) for m in AT))
    if ref:
        print(f"(pros: {ref[AT[0]]['games']} Zergs in TvZ games of 12:30 or more)")


if __name__ == "__main__":
    main()
