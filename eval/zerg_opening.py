"""The one Gary playing Zerg against Gary-Z, their openings side by side (docs/REFACTOR.md Phase 1's gate:
the Zerg executor builds as Gary-Z does in the first four minutes).

Mirror games (eval/mirror.py) on the ladder maps, each seed played twice with the sides swapped, re-simulated
from the replays (Game.scenario): per player, at the limit, drones, hatcheries (town halls) and army supply;
when the first Spawning Pool, Extractor and second hatchery were started; the seconds supply-blocked (supply
used at its cap, under 200); and the larvae standing unused (summed over the seconds, as larva-seconds).

    python -m eval.zerg_opening --manifest gz1 --seeds 5001 5002 5003 5004 --minutes 4
"""

from __future__ import annotations

import argparse
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval import monitor, priority  # noqa: E402
from gary import zerg as Z  # noqa: E402

OUT = REPO / "runs" / "zerg_opening"


def play(job: tuple) -> Path:
    map_path, seed, sides, minutes = job
    rep = OUT / f"{Path(map_path).stem.replace(' ', '_')}_{seed}_{sides[0]}_{sides[1]}.rep"
    cmd = [sys.executable, "-m", "eval.mirror", "--map", str(map_path), "--seed", str(seed), "--minutes", str(minutes),
           "--sides", *sides, "--save", str(rep)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return rep


def opening(rep: Path, minutes: float) -> dict[int, dict]:
    """Per slot: the opening's numbers (above)."""
    from gary.env import Game
    from gary.policy.army import supply_x2
    g = Game.scenario(str(rep))
    out: dict[int, dict] = {}
    for sec in range(1, int(minutes * 60) + 1):
        try:
            while g.frame < sec * 24:
                g.step(24 - g.frame % 24 if g.frame % 24 else 24)
        except Exception:
            break
        o = g.observe()
        for p in o["players"]:
            s = p["slot"]
            mine = [u for u in o["units"] if u["owner"] == s]
            if not any(u["type"] in Z.TOWN_HALLS for u in mine):
                continue
            d = out.setdefault(s, {"pool": None, "extractor": None, "hatch2": None, "blocked": 0, "larva_s": 0})
            if d["pool"] is None and any(u["type"] == Z.POOL for u in mine):
                d["pool"] = sec
            if d["extractor"] is None and any(u["type"] == Z.EXTRACTOR for u in mine):
                d["extractor"] = sec
            if d["hatch2"] is None and sum(1 for u in mine if u["type"] in Z.TOWN_HALLS) >= 2:
                d["hatch2"] = sec
            if p["supply_max"] < 200 and p["supply_used"] >= p["supply_max"]:
                d["blocked"] += 1
            d["larva_s"] += sum(1 for u in mine if u["type"] == Z.LARVA)
            d["drones"] = sum(1 for u in mine if u["type"] == Z.DRONE)
            d["halls"] = sum(1 for u in mine if u["type"] in Z.TOWN_HALLS)
            d["army"] = sum(supply_x2(u["type"]) for u in mine if u["type"] in Z.ARMY) / 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="gz1", help="the one Gary's manifest")
    ap.add_argument("--seeds", nargs="+", type=int, default=[5001, 5002, 5003, 5004])
    ap.add_argument("--maps", nargs="+", default=["KnockOut 1.4", "Fighting Spirit 1.4"])
    ap.add_argument("--minutes", type=float, default=4)
    ap.add_argument("--parallel", type=int, default=8)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from eval.league import ladder_maps
    maps = ladder_maps(args.maps)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(path, seed, sides, args.minutes) for path in maps.values() for seed in args.seeds
            for sides in ((args.manifest, "garyz"), ("garyz", args.manifest))]
    with ThreadPoolExecutor(args.parallel) as pool:
        reps = list(monitor.track(pool.map(play, jobs), len(jobs), "mirror openings played"))
    rows: dict[str, list[dict]] = {args.manifest: [], "garyz": []}
    for rep, (_, _, sides, _) in monitor.track(list(zip(reps, jobs)), len(jobs), "openings re-simulated"):
        per = opening(rep, args.minutes)
        for s, side in zip(sorted(per), sides):
            rows[side].append(per[s])
    keys = ("drones", "halls", "army", "pool", "extractor", "hatch2", "blocked", "larva_s")
    print(f"{len(jobs)} games, {args.minutes:g} minutes; medians (none: never by then)")
    print("side      " + "".join(f"{k:>10s}" for k in keys))
    for side, rs in rows.items():
        cells = []
        for k in keys:
            vals = [r.get(k) for r in rs if r.get(k) is not None]
            missing = sum(1 for r in rs if r.get(k) is None)
            cells.append(f"{st.median(vals):>8g}" + (f"/{missing}" if missing else "  ") if vals else f"{'none':>10s}")
        print(f"{side:10s}" + "".join(cells))


if __name__ == "__main__":
    main()
