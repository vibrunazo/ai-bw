"""How far along a Terran's economy and army are, Gary's next to the pros'.

Losses alone reward turtling (a Gary that never expands loses fewest rushes), so eval/vs_rush.py
also reports, at 8:00 and 12:00, the Terran's SCVs, Command Centers and army supply (supply of
its units other than SCVs), next to the same numbers in pro TvZ games (their replays played in
OpenBW: games of 12:30 or more, so both players are still there; cached in
data/interim/pro_economy.json).

    python -m eval.economy            # the pros' numbers (built once, then read from the cache)
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

AT = (8 * 60, 12 * 60)
GAMES = 300
CC, SCV = 106, 7


def terran_economy(units: list[dict], slot: int) -> dict:
    """SCVs, Command Centers and army supply of one player, from a game observation's units."""
    from gary.policy.army import is_building, supply_x2
    mine = [u for u in units if u["owner"] == slot]
    return {"scvs": sum(1 for u in mine if u["type"] == SCV),
            "ccs": sum(1 for u in mine if u["type"] == CC),
            "army": sum(supply_x2(u["type"]) for u in mine if u["type"] != SCV and not is_building(u["type"])
                        and u["completed"]) / 2}


def _pro_game(path: str) -> dict | None:
    try:
        from gary.env import Game
        game = Game.scenario(path)
        slot = next(u["owner"] for u in game.observe()["units"] if u["type"] == CC)
        out = {}
        for t in AT:
            game.step(t * 24 - game.observe()["frame"])
            out[t] = terran_economy(game.observe()["units"], slot)
        game.close()
        return out
    except Exception:
        return None


def pro_reference() -> dict:
    """{seconds: {"scvs", "ccs", "army": median}} over pro TvZ Terrans."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    from inventory import data_root
    cache = data_root() / "interim" / "pro_economy.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    recs = [json.loads(line) for line in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["matchup"] == "TvZ" and r["race"] == "T" and r.get("duration_s", 0) >= 750]
    random.Random(0).shuffle(recs)
    with ProcessPoolExecutor(16) as pool:
        games = [g for g in pool.map(_pro_game, [str(data_root() / "raw" / r["rel_path"]) for r in recs[:GAMES]]) if g]
    ref = {t: {k: st.median(g[t][k] for g in games) for k in ("scvs", "ccs", "army")} | {"games": len(games)} for t in AT}
    cache.write_text(json.dumps(ref))
    return ref


def line(name: str, at: dict) -> str:
    """One summary line: {seconds: {"scvs", "ccs", "army"}} medians."""
    return f"{name}: " + "; ".join(f"{t // 60}:{t % 60:02d} SCVs {v['scvs']:.0f}, CCs {v['ccs']:.0f}, army {v['army']:.0f}"
                                   for t, v in sorted(at.items()))


if __name__ == "__main__":
    print(line("pros", pro_reference()))
