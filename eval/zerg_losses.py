"""Why the Zerg loses (Gary-Z against Gary-T, and pro Zergs as the yardstick): what eval/losses.py did for
Gary-T. Each game re-simulated in OpenBW (the full state: this explains, it isn't an input), every
second, the Zerg's side:

  deaths     every Zerg fighter that died: where (at a Zerg base, in the field, at a Terran base), when,
             the local odds (Zerg and Terran fighter supply within NEAR of it), and the share of the whole
             Zerg army that was there (small: it trickled in, a few at a time)
  trades     the value the Zerg's fighters lost and the Terran value they killed, by place
  army       its army supply and composition at fixed minutes
  mutas      mutalisks alive, and SCVs that died with Zerg mutalisks near (harass that worked)
  attacks    seconds with a quarter or more of the Zerg army at a Terran base, and Terran buildings lost

    python -m eval.zerg_losses --gary "runs/league/c1/*/*.rep" --pros 60
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority

REPO = Path(__file__).resolve().parents[1]
NEAR = 8 * 32
BASE = 15 * 32
AT = (6, 8, 10, 12, 15)
MAPS = ("python", "fighting spirit", "heartbreak")


def _value(t: int) -> float:
    from gary import terran as T
    from gary import zerg as Z
    m, g = T.COST.get(t) or Z.COST.get(t) or (50, 0)
    return (m + g) / (2 if t in Z.PAIRS else 1)


def game(job: tuple) -> dict | None:
    rep, end = job
    from gary import terran as T
    from gary import zerg as Z
    from gary.env import Game
    from gary.policy.army import is_building, is_worker, supply_x2
    fighter = lambda t: supply_x2(t) > 0 and not is_worker(t) and not is_building(t)
    out = {"deaths": [], "killed": {"home": 0.0, "field": 0.0, "enemy": 0.0}, "lost": {"home": 0.0, "field": 0.0, "enemy": 0.0},
           "army": {}, "mutas": [], "scv_by_mutas": 0, "attack_s": 0, "terran_buildings_lost": 0, "seconds": 0}
    try:
        g = Game.scenario(rep)
    except Exception:
        return None
    try:
        zslot = tslot = None
        prev: dict = {}
        prev_zf, prev_total = [], 1
        last = min(end or 20 * 1440, 20 * 1440)
        while g.frame + 24 <= last:
            try:
                g.step(24)
            except Exception:
                break
            o = g.observe()
            units = [u for u in o["units"] if u["owner"] != 11]
            if zslot is None:
                zslot = next((u["owner"] for u in units if u["type"] in Z.TOWN_HALLS), None)
                tslot = next((u["owner"] for u in units if u["type"] == T.CC), None)
                if zslot is None or tslot is None:
                    continue
            zh = [(u["x"], u["y"]) for u in units if u["owner"] == zslot and u["type"] in Z.TOWN_HALLS]
            th = [(u["x"], u["y"]) for u in units if u["owner"] == tslot and u["type"] == T.CC]
            if not zh or not th:
                break
            out["seconds"] += 1
            now = {u["tag"]: u for u in units}
            zf = [u for u in units if u["owner"] == zslot and fighter(u["type"])]
            tf = [u for u in units if u["owner"] == tslot and fighter(u["type"])]
            total = sum(supply_x2(u["type"]) for u in zf) or 1

            def place(x, y):
                if min((x - a) ** 2 + (y - b) ** 2 for a, b in zh) < BASE ** 2:
                    return "home"
                if min((x - a) ** 2 + (y - b) ** 2 for a, b in th) < BASE ** 2:
                    return "enemy"
                return "field"
            for tag, u in prev.items():
                if tag in now:
                    continue
                if u["owner"] == zslot and fighter(u["type"]):
                    x, y = u["x"], u["y"]
                    near = lambda us: sum(supply_x2(v["type"]) for v in us if (v["x"] - x) ** 2 + (v["y"] - y) ** 2 < NEAR ** 2)
                    pz, pt = near(prev_zf) + supply_x2(u["type"]), near(tf)
                    p = place(x, y)
                    out["deaths"].append({"t": g.frame // 24, "place": p, "zerg": pz / 2, "terran": pt / 2,
                                          "share": pz / max(1, prev_total)})
                    out["lost"][p] += _value(u["type"])
                elif u["owner"] == tslot:
                    p = place(u["x"], u["y"])
                    if fighter(u["type"]) or is_worker(u["type"]):
                        out["killed"][p] += _value(u["type"])
                    if is_building(u["type"]):
                        out["terran_buildings_lost"] += 1
                    if is_worker(u["type"]) and any(v["type"] == Z.MUTALISK and (v["x"] - u["x"]) ** 2 + (v["y"] - u["y"]) ** 2 < NEAR ** 2
                                                    for v in prev_zf):
                        out["scv_by_mutas"] += 1
            at_enemy = sum(supply_x2(u["type"]) for u in zf if place(u["x"], u["y"]) == "enemy")
            out["attack_s"] += at_enemy >= 0.25 * total and total >= 8
            m = g.frame // 1440
            if g.frame % 1440 < 24 and m in AT:
                out["army"][m] = {"supply": total / 2, "mutas": sum(1 for u in zf if u["type"] == Z.MUTALISK),
                                  "lings": sum(1 for u in zf if u["type"] == Z.ZERGLING),
                                  "hydras": sum(1 for u in zf if u["type"] == Z.HYDRALISK),
                                  "lurkers": sum(1 for u in zf if u["type"] == Z.LURKER),
                                  "terran_supply": sum(supply_x2(u["type"]) for u in tf) / 2}
            prev = now
            prev_zf, prev_total = zf, total
    finally:
        g.close()
    return out


def summary(games: list[dict]) -> dict:
    d = [x for gm in games for x in gm["deaths"]]
    med = lambda v: round(float(st.median(v)), 2) if v else None
    out = {"games": len(games), "deaths": len(d)}
    for p in ("home", "field", "enemy"):
        dp = [x for x in d if x["place"] == p]
        lost = sum(gm["lost"][p] for gm in games)
        killed = sum(gm["killed"][p] for gm in games)
        out[p] = {"deaths_share": round(len(dp) / max(1, len(d)), 3),
                  "odds_median": med([x["zerg"] / max(0.5, x["terran"]) for x in dp]),
                  "army_share_median": med([x["share"] for x in dp]),
                  "outnumbered_share": round(float(np.mean([x["terran"] > 1.5 * x["zerg"] for x in dp])), 3) if dp else None,
                  "killed_per_lost": round(killed / lost, 2) if lost else None}
    out["army"] = {m: {k: med([gm["army"][m][k] for gm in games if m in gm["army"]]) for k in
                       ("supply", "lings", "hydras", "mutas", "lurkers", "terran_supply")} for m in AT}
    out["scv_by_mutas_per_game"] = round(float(np.mean([gm["scv_by_mutas"] for gm in games])), 2)
    out["attack_minutes_per_game"] = round(float(np.mean([gm["attack_s"] for gm in games])) / 60, 2)
    out["terran_buildings_lost_per_game"] = round(float(np.mean([gm["terran_buildings_lost"] for gm in games])), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gary", required=True, help="a glob of Gary-Z's league replays")
    ap.add_argument("--pros", type=int, default=60, help="pro TvZ games on the same maps (the Zerg's side)")
    ap.add_argument("--workers", type=int, default=4)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from eval.humanlike import clean_replays, gary_end
    from ingest.inventory import data_root
    reps = sorted(glob.glob(args.gary))
    jobs = [(r, gary_end(r)) for r in reps]
    clean = clean_replays()
    recs = [json.loads(line) for line in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["matchup"] == "TvZ" and r["race"] == "Z" and r["rel_path"] in clean
            and r.get("duration_s", 0) >= 600 and any(m in re.sub(r"[^a-z ]", "", r.get("map", "").lower()) for m in MAPS)]
    random.Random(0).shuffle(recs)
    pro_jobs = [(str(data_root() / "raw" / r["rel_path"]), clean[r["rel_path"]] or None) for r in recs[:args.pros]]
    with ProcessPoolExecutor(args.workers) as pool:
        gary = [x for x in monitor.track(pool.map(game, jobs), len(jobs), "Gary-Z games") if x]
        pros = [x for x in monitor.track(pool.map(game, pro_jobs), len(pro_jobs), "pro games") if x]
    report = {"gary": summary(gary), "pros": summary(pros)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
