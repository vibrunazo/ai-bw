"""The league (docs/SELF_PLAY.md, "The loop"): Gary-T against Gary-Z in parallel headless games, each
game's outcome, both sides' humanlikeness and the learned layers' decisions kept for the RL that
trains on them.

A round: the Terran version against Gary-Z on the three pool maps, seeds from --first. Per game, in
runs/league/<round>/<map>/: the replay, Gary's log, and Gary-Z's decision log (gary/bots/
decision_log.py: both sides' learned layers' drawn decisions, each with inputs and log-probability). Then the round's table, round.json:

  results      Gary-T won / lost / undecided, by map
  gap          each side's gap score against the pros (eval/humanlike.py: mean standardized gap per
               minute range; 0 is the pros), the anchor the league isn't trained on
  decisions    how many each learned layer logged (what RL has to learn from)

The fight episodes train the fight roles (zerg_roles.py); the army layer (gary_z.py _army_goal)
learns from whole games and segments, through the outcomes logged here.

    python -m eval.league --round r1 --seeds 8 --minutes 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority

REPO = Path(__file__).resolve().parents[1]
MAPS = {m: REPO / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}


def play(job: tuple) -> dict:
    """One game in its own process (Gary-Z's decision log written as the process ends)."""
    version, name, path, seed, minutes, out, pairs, incumbent, candidate, layer = job
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    rep = folder / f"{version}_seed{seed}.rep"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "GARY_DECISION_LOG": str(rep.with_suffix(".decisions.npz"))}
    # Gary-Z's fight roles: the incumbent, or on odd seeds the candidate (its own games: the gap score
    # judges it too); paired episodes against the other one
    main, other = (candidate, incumbent) if candidate and seed % 2 else (incumbent, candidate)
    if main:
        env["GARY_TARMY_MODEL" if layer == "army_t" else "GARY_ZROLES_MODEL"] = str(main)
    if pairs:
        env["GARY_PAIRS"] = str(rep.with_suffix(".pairs.npz"))
        env["GARY_PAIRS_LAYER"] = layer
        if other:
            env["GARY_PAIRS_CANDIDATE"] = str(other)
    cmd = [sys.executable, "-m", "gary.bots.zerg_rush", "--vs", version, "--opponent", "garyz", "--map", str(path),
           "--minutes", str(minutes), "--seed", str(seed), "--save", str(rep)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    log = p.stdout + p.stderr
    rep.with_suffix(".log").write_text(log, encoding="utf-8")
    result = "error"
    for line in reversed(log.splitlines()):
        if line.startswith(("Gary won", "Gary lost", "time")):
            result = line.split(";")[0].replace("Gary ", "")
            break
    return {"map": name, "seed": seed, "result": result, "replay": str(rep), "zroles": str(main) if main else None}


def decisions(rep: str) -> dict:
    p = Path(rep).with_suffix(".decisions.npz")
    if not p.exists():
        return {}
    d = np.load(p)
    return {k.split("/")[0]: int(d[k].shape[0]) for k in d.files if k.endswith("/frame")}


def gaps(replays: list[str], race: str, workers: int) -> list:
    """The side's gap score per minute range (eval/humanlike.py), against the cached pros."""
    from eval.humanlike import _job, _names, gap_score, gary_end, matrix, pro_rows
    names = _names(race)
    with ProcessPoolExecutor(workers) as pool:
        got = dict(pool.map(_job, [(r, race, gary_end(r)) for r in replays]))
    gary = {p: rows for p, rows in got.items() if rows}
    pros = pro_rows(race, 300, workers)
    Xg, _, _ = matrix(gary, names)
    Xp, _, _ = matrix(pros, names)
    return gap_score(Xg, Xp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, help="the round's name (its folder in runs/league)")
    ap.add_argument("--vs", default="v029", help="the Terran Gary")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per map")
    ap.add_argument("--first", type=int, default=101)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--pairs", action="store_true", help="paired fight episodes in every game (eval/pairs.py)")
    ap.add_argument("--incumbent", help="Gary-Z's fight-role model (default: its newest)")
    ap.add_argument("--candidate", help="a candidate model of the layer: odd seeds play it, pairs A/B it")
    ap.add_argument("--layer", default="roles_z", choices=("roles_z", "army_t"),
                    help="what the pairs and the candidate are: Gary-Z's fight roles or Gary-T's army layer")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    out = REPO / "runs" / "league" / args.round
    jobs = [(args.vs, name, path, s, args.minutes, out, args.pairs, args.incumbent, args.candidate, args.layer)
            for name, path in MAPS.items() for s in range(args.first, args.first + args.seeds)]
    with ThreadPoolExecutor(args.parallel) as pool:
        games = list(monitor.track(pool.map(play, jobs), len(jobs), "games"))
    for g in games:
        g["decisions"] = decisions(g["replay"])
        monitor.tally("results", g["result"])
    done = [g["replay"] for g in games if g["result"] != "error"]
    table = {"round": args.round, "terran": args.vs, "minutes": args.minutes, "games": games,
             "results": dict(Counter(g["result"] for g in games)),
             "by_map": {m: dict(Counter(g["result"] for g in games if g["map"] == m)) for m in MAPS},
             "decisions": {k: sum(g["decisions"].get(k, 0) for g in games) for k in ("army_t", "roles_t", "army_z", "roles_z")},
             "gap": {"T": gaps(done, "T", args.parallel), "Z": gaps(done, "Z", args.parallel)},
             "decided": sum(1 for g in games if g["result"] in ("won", "lost")) / max(1, len(games))}
    if args.candidate:                    # (the layer's side's gap, its candidate's games against the incumbent's)
        side = "T" if args.layer == "army_t" else "Z"
        for who, model in (("candidate", args.candidate), ("incumbent", args.incumbent)):
            mine = [g["replay"] for g in games if g["result"] != "error" and g["zroles"] == str(model)]
            table["gap"][f"{side}_{who}"] = gaps(mine, side, args.parallel) if mine else None
    if args.pairs:
        from eval.pairs import summarize
        files = sorted(glob.glob(str(out / "*" / "*.pairs.npz")))
        table["pairs"] = summarize(files, args.incumbent, args.candidate)
    (out / "round.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"round {args.round}: {len(games)} games, Gary-T {table['results']}; by map {table['by_map']}")
    print(f"decisions logged: {table['decisions']}")
    for side, gap in table["gap"].items():
        if gap:
            print(f"gap score {side} (2-4/5-7/8-10/11-14/15-20 min, all): "
                  + " ".join("-" if v is None else f"{v[0]:.2f}" for v in gap))
    if "pairs" in table:
        print(f"pairs: {json.dumps(table['pairs'])}")
    print(f"saved {out / 'round.json'}")


if __name__ == "__main__":
    main()
