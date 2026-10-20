"""How a Terran's attacks are carried out, Gary's against modern pros' (TvZ), from the replays re-simulated
(the full state, so what each side really had, not what it saw). Human game 4's first item: "the attack carried
out as an attack-move from home: no medics, into sunkens, no regroup in front of the natural, no retreat when
hit, not called off when outnumbered".

An attack begins when ATTACK_N or more of the Terran's fighters are within NEAR of a Zerg building, after QUIET
with none there, and lasts while any are (MAX_S at most). Each attack, measured around its point (the fighters'
middle at the start):

  share     of the Terran's fighter supply on the map, the most within THERE of the point over its first
            ARRIVE_S: the army that went (pros gather it first)
  late      fighters arriving within THERE after ARRIVE_S, per fighter there by then: the trickle; and of those,
            the share made after the attack began (production walking to it) rather than stragglers
  medics    medics per marine within THERE at the share's moment, over the same ratio across its army
            (1: they came along; under 1: left behind)
  sunkens   sunken colonies within NEAR of the point at the start: attacked into static defense
  lost      the Terran's fighter supply that died in the attack (within THERE of the point)
  killed    the Zerg's fighter supply, drones and buildings (a building counted as BUILDING_SUPPLY) that
            died within THERE of the point
  stayed    of the attacks where the Terran lost more than it killed by its middle, the share of its fighters
            still within THERE at the end: not called off

    python -m eval.attacks --gary "runs/league/G13/*/*.rep" --pros 40
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))

from eval import monitor, priority  # noqa: E402
from gary import terran as T  # noqa: E402
from gary import zerg as Z  # noqa: E402

STEP = 12                         # frames between looks (half a second)
MINUTES = 30
NEAR = 10 * 32
THERE = 14 * 32
ATTACK_N = 3
QUIET = 20 * 24
ARRIVE_S = 15
MAX_S = 120
BUILDING_SUPPLY = 4.0
SIEGED = 30
FIGHTERS = {T.MARINE, T.MEDIC, T.FIREBAT, T.GHOST, T.VULTURE, T.TANK, SIEGED, T.GOLIATH, T.WRAITH, T.VESSEL,
            T.VALKYRIE, T.BC}
MEDIC = T.MEDIC
Z_FIGHTERS = {Z.ZERGLING, Z.HYDRALISK, Z.MUTALISK, Z.SCOURGE, Z.LURKER, Z.ULTRALISK, Z.DEFILER, Z.GUARDIAN,
              Z.DEVOURER, Z.QUEEN}


def _supply(t: int) -> float:
    return T.SUPPLY.get(T.TANK if t == SIEGED else t, 0)


def _zsupply(t: int) -> float:
    return Z.SUPPLY.get(t, 0) / (2 if t in Z.PAIRS else 1)


def _is_zbuilding(t: int) -> bool:
    return 131 <= t <= 149                        # Hatchery to Extractor


def attacks(path: str) -> list[dict] | None:
    """The Terran's attacks in one replay."""
    from gary.env import Game
    try:
        g = Game.scenario(str(path))
    except Exception:
        return None
    ts = zs = None
    out: list[dict] = []
    cur = None
    last_near = last_big = -10 ** 9           # (any fighter near a Zerg building; ATTACK_N or more)
    prev: dict[int, dict] = {}
    born: dict[int, int] = {}                 # each Terran fighter's first frame seen
    end = MINUTES * 60 * 24
    while g.frame < end:
        try:
            g.step(STEP)
        except Exception:
            break
        o = g.observe()
        if ts is None:
            ts = next((p["slot"] for p in o["players"] if p["race"] == 1), None)
            zs = next((p["slot"] for p in o["players"] if p["race"] == 0), None)
            if ts is None or zs is None:
                return None
        f = o["frame"]
        units = {u["tag"]: u for u in o["units"]}
        mine = [u for u in o["units"] if u["owner"] == ts and u["type"] in FIGHTERS and u["completed"]]
        for u in mine:
            born.setdefault(u["tag"], f)
        zb = [(u["x"], u["y"]) for u in o["units"] if u["owner"] == zs and _is_zbuilding(u["type"])]
        near = [u for u in mine if zb and min(math.dist((u["x"], u["y"]), b) for b in zb) < NEAR]
        if near:
            if cur is None and len(near) >= ATTACK_N and f - last_big >= QUIET:
                px = sum(u["x"] for u in near) / len(near)
                py = sum(u["y"] for u in near) / len(near)
                cur = {"start": f, "at": (px, py), "share": 0.0, "medics": None, "there": set(), "late": 0,
                       "late_new": 0, "sunkens": sum(1 for u in o["units"] if u["owner"] == zs and u["type"] == Z.SUNKEN
                                      and math.dist((u["x"], u["y"]), (px, py)) < NEAR),
                       "lost": 0.0, "killed": 0.0, "lost_mid": None, "killed_mid": None}
            last_near = f
            if len(near) >= ATTACK_N:
                last_big = f
        if cur is not None:
            at = cur["at"]
            here = [u for u in mine if math.dist((u["x"], u["y"]), at) < THERE]
            total = sum(_supply(u["type"]) for u in mine) or 1.0
            s = (f - cur["start"]) / 24
            if s <= ARRIVE_S:
                share = sum(_supply(u["type"]) for u in here) / total
                if share >= cur["share"]:
                    cur["share"] = share
                    m_here = sum(1 for u in here if u["type"] == T.MARINE)
                    d_here = sum(1 for u in here if u["type"] == MEDIC)
                    m_all = sum(1 for u in mine if u["type"] == T.MARINE)
                    d_all = sum(1 for u in mine if u["type"] == MEDIC)
                    cur["medics"] = (d_here / m_here) / (d_all / m_all) if m_here and m_all and d_all else None
                cur["there"] |= {u["tag"] for u in here}
            else:
                fresh = [u for u in here if u["tag"] not in cur["there"]]
                cur["late"] += len(fresh)
                cur["late_new"] += sum(1 for u in fresh if born.get(u["tag"], 0) > cur["start"])
                cur["there"] |= {u["tag"] for u in here}
            for t, u in prev.items():                     # (deaths since the last look, near the point)
                if t in units or math.dist((u["x"], u["y"]), at) >= THERE:
                    continue
                if u["owner"] == ts and u["type"] in FIGHTERS:
                    cur["lost"] += _supply(u["type"])
                elif u["owner"] == zs:
                    cur["killed"] += BUILDING_SUPPLY if _is_zbuilding(u["type"]) else _zsupply(u["type"]) if u["type"] in Z_FIGHTERS else 1.0 if u["type"] == Z.DRONE else 0.0
            if s >= MAX_S / 2 and cur["lost_mid"] is None:
                cur["lost_mid"], cur["killed_mid"] = cur["lost"], cur["killed"]
            if f - last_near >= 10 * 24 or s >= MAX_S:
                n0 = len(cur["there"]) or 1
                still = sum(1 for u in here) / n0
                lm = cur["lost"] if cur["lost_mid"] is None else cur["lost_mid"]
                km = cur["killed"] if cur["killed_mid"] is None else cur["killed_mid"]
                out.append({"minute": cur["start"] / 1440, "share": cur["share"], "late": cur["late"] / n0,
                            "late_n": cur["late"], "late_new": cur["late_new"],
                            "medics": cur["medics"], "sunkens": cur["sunkens"], "lost": cur["lost"],
                            "killed": cur["killed"], "losing": lm > km, "stayed": still})
                cur = None
        prev = {u["tag"]: u for u in o["units"] if u["owner"] in (ts, zs)}
    return out


def summary(name: str, games: list[list[dict]]) -> None:
    rows = [a for g in games for a in g]
    if not rows:
        print(name, "no attacks")
        return
    med = lambda v: round(st.median(v), 2) if v else None
    mean = lambda v: round(sum(v) / len(v), 2) if v else None
    print(f"{name}: {len(games)} games, {len(rows)} attacks ({len(rows) / len(games):.1f} a game)")
    print(f"  share of the army there (median) {med([a['share'] for a in rows])}; late arrivals per one there "
          f"{mean([a['late'] for a in rows])}, of them made after the attack began "
          f"{sum(a['late_new'] for a in rows) / max(1, sum(a['late_n'] for a in rows)):.0%}")
    print(f"  medics along (1 = as in its army) {med([a['medics'] for a in rows if a['medics'] is not None])}; "
          f"into sunkens {mean([a['sunkens'] > 0 for a in rows])} of attacks")
    print(f"  lost / killed supply a game {mean([a['lost'] for a in rows]) * len(rows) / len(games):.1f} / "
          f"{mean([a['killed'] for a in rows]) * len(rows) / len(games):.1f}; per attack median {med([a['lost'] for a in rows])} / "
          f"{med([a['killed'] for a in rows])}")
    losing = [a for a in rows if a["losing"]]
    print(f"  attacks losing by their middle {len(losing) / len(rows):.0%}; of those, fighters still there at the end "
          f"{mean([a['stayed'] for a in losing])}")
    by = {}
    for a in rows:
        by.setdefault(min(int(a["minute"] // 4) * 4, 28), []).append(a)
    print("  by minute:", {f"{k}-{k + 4}": (len(v), med([a['share'] for a in v])) for k, v in sorted(by.items())})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gary", nargs="*", default=[], help="globs of Gary's replays (the Terran's side)")
    ap.add_argument("--pros", type=int, default=0, help="this many modern pro TvZ games (clean, 10+ minutes)")
    ap.add_argument("--workers", type=int, default=8)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    sets = []
    for pattern in args.gary:
        sets.append((pattern, sorted(glob.glob(pattern))))
    if args.pros:
        from inventory import data_root
        from eval.humanlike import clean_replays
        clean = clean_replays()
        recs = []
        with open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("matchup") == "TvZ" and r.get("race") == "T" and r.get("source") == "cwal" \
                        and r["rel_path"] in clean and r.get("duration_s", 0) >= 600:
                    recs.append(r)
        random.Random(0).shuffle(recs)
        sets.append(("pros", [str(data_root() / "raw" / r["rel_path"]) for r in recs[:args.pros]]))
    with ProcessPoolExecutor(args.workers) as pool:
        for name, paths in sets:
            games = [g for g in monitor.track(pool.map(attacks, paths), len(paths), f"replays: {name}") if g is not None]
            summary(name, games)


if __name__ == "__main__":
    main()
