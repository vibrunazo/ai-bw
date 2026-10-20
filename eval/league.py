"""The league (docs/SELF_PLAY.md, "The loop"): a Terran Gary against a Zerg Gary (gz1 by default) in parallel
headless games (eval/match.py), each game's outcome, both sides' humanlikeness and the learned layers'
decisions kept for the RL that trains on them.

A round: the Terran version against the Zerg on the maps, seeds from --first. Per game, in
runs/league/<round>/<map>/: the replay, the game's log, both sides' screens, traces and records, and the
learned layers' drawn decisions (gary/decision_log.py: each with inputs and log-probability). Then the
round's table, round.json:

  results      Gary-T won / lost / undecided, by map
  gap          each side's gap score against the pros (eval/humanlike.py: mean standardized gap per
               minute range; 0 is the pros), the anchor the league isn't trained on
  decisions    how many each learned layer logged (what RL has to learn from)

The fight episodes train the Zerg's fight roles (gary/agent/zerg_roles.py); the army layers learn from
whole games and segments, through the outcomes logged here.

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
    """One game in its own process (the learned layers' decision log written as the process ends)."""
    version, name, path, seed, minutes, out, pairs, incumbent, candidate, layer, zerg = job
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    rep = folder / f"{version}_seed{seed}.rep"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "GARY_DECISION_LOG": str(rep.with_suffix(".decisions.npz"))}
    # the Zerg's fight roles: the incumbent, or on odd seeds the candidate (its own games: the gap score
    # judges it too); paired episodes against the other one
    main, other = (candidate, incumbent) if candidate and seed % 2 else (incumbent, candidate)
    if main:
        env[{"army_t": "GARY_TARMY_MODEL", "engage_t": "GARY_TENGAGE_MODEL", "commit_t": "GARY_TCOMMIT_MODEL"}.get(
            layer, "GARY_ZROLES_MODEL")] = str(main)
    if pairs:
        env["GARY_PAIRS"] = str(rep.with_suffix(".pairs.npz"))
        env["GARY_PAIRS_LAYER"] = layer
        if other:
            env["GARY_PAIRS_CANDIDATE"] = str(other)
    cmd = [sys.executable, "-m", "eval.match", "--a", version, "--b", zerg, "--map", str(path),
           "--minutes", str(minutes), "--seed", str(seed), "--save", str(rep)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    log = p.stdout + p.stderr
    rep.with_suffix(".log").write_text(log, encoding="utf-8")
    result = "error"
    for line in reversed(log.splitlines()):
        if line.startswith(("A won", "B won", "time")):
            result = {"A won": "won", "B won": "lost"}.get(line.split(";")[0], "time")
            break
    return {"map": name, "seed": seed, "result": result, "replay": str(rep), "zroles": str(main) if main else None}


def decisions(rep: str) -> dict:
    p = Path(rep).with_suffix(".decisions.npz")
    if not p.exists():
        return {}
    d = np.load(p)
    return {k.split("/")[0]: int(d[k].shape[0]) for k in d.files if k.endswith("/frame")}


def side_rows(replays: list[str], race: str, workers: int) -> dict[str, list[dict]]:
    """Each replay's per-minute feature rows for one side (eval/humanlike.py features), from the re-simulation."""
    from eval.humanlike import _job, gary_end
    with ProcessPoolExecutor(workers) as pool:
        got = dict(pool.map(_job, [(r, race, gary_end(r)) for r in replays]))
    return {p: rows for p, rows in got.items() if rows}


def gaps(rows: dict[str, list[dict]], race: str, workers: int, era: str = "old") -> list:
    """The side's gap score per minute range (eval/humanlike.py), against the cached pros (era "modern":
    the pros on the current ladder maps, for rounds there)."""
    from eval.humanlike import _names, gap_score, matrix, pro_rows
    names = _names(race)
    pros = pro_rows(race, 300, workers, era)
    Xg, _, _ = matrix(rows, names)
    Xp, _, _ = matrix(pros, names)
    return gap_score(Xg, Xp)


AHEAD = 1.4          # a game at time: the side whose supply used (workers and army) in the last minute is this many
                     # times the other's is ahead; else even


def grade(games: list[dict], rows_t: dict, rows_z: dict) -> None:
    """Games at time graded by who is ahead at the end (g["graded"]): the critic's job for the league (docs/
    SELF_PLAY.md) until a Zerg one exists; the Terran value network reads the lead already taken well (0.92 AUC
    on decided games), and that is all this needs. Supply used is race-neutral: drones and army against SCVs
    and army. 15 of 24 games a round ended at time with both sides looking alike in the results."""
    for g in games:
        g["graded"] = g["result"]
        if g["result"] != "time":
            continue
        t, z = rows_t.get(g["replay"]), rows_z.get(g["replay"])
        if not t or not z:
            continue
        st, sz = t[-1].get("supply_used") or 0, z[-1].get("supply_used") or 0
        g["graded"] = "time, T ahead" if st >= AHEAD * sz else "time, Z ahead" if sz >= AHEAD * st else "time, even"


def ladder_maps(names: list[str] | None = None) -> dict[str, Path]:
    """The current ladder maps (data/maps/ladder/<name>.rep, ingest/ladder_maps.py), all or these."""
    sys.path.insert(0, str(REPO / "ingest"))
    from inventory import data_root
    found = {p.stem: p for p in sorted((data_root() / "maps" / "ladder").glob("*.rep"))}
    if not found:
        sys.exit("no ladder maps: python ingest/ladder_maps.py")
    return found if names is None else {n: found[n] for n in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, help="the round's name (its folder in runs/league)")
    ap.add_argument("--vs", default="v029", help="the Terran Gary")
    ap.add_argument("--zerg", default="gz1", help="the Zerg Gary's manifest")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per map")
    ap.add_argument("--maps", default="fixtures", help="fixtures (the three 1.16 maps), ladder (every current ladder map "
                    "written by ingest/ladder_maps.py) or ladder names, comma-separated (e.g. KnockOut,Attitude SE)")
    ap.add_argument("--first", type=int, default=101)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--pairs", action="store_true", help="paired fight episodes in every game (eval/pairs.py)")
    ap.add_argument("--incumbent", help="the Zerg's fight-role model (default: its newest)")
    ap.add_argument("--candidate", help="a candidate model of the layer: odd seeds play it, pairs A/B it")
    ap.add_argument("--layer", default="roles_z", choices=("roles_z", "army_t", "engage_t", "commit_t"),
                    help="what the pairs and the candidate are: the Zerg's fight roles or the Terran's army layer")
    priority.add_argument(ap)
    ap.add_argument("--regrade", action="store_true", help="an existing round: grade its games at time by who is ahead (no games played)")
    args = ap.parse_args()
    maps = MAPS if args.maps == "fixtures" else ladder_maps(None if args.maps == "ladder" else args.maps.split(","))
    era = "old" if args.maps == "fixtures" else "modern"     # (the pros the gap score compares with: the same maps')
    priority.apply(args.priority)
    out = REPO / "runs" / "league" / args.round
    if args.regrade:
        table = json.loads((out / "round.json").read_text(encoding="utf-8"))
        games = table["games"]
        done = [g["replay"] for g in games if g["result"] != "error"]
        rows_t, rows_z = side_rows(done, "T", args.parallel), side_rows(done, "Z", args.parallel)
        grade(games, rows_t, rows_z)
        table["graded"] = dict(Counter(g["graded"] for g in games))
        table["graded_by_map"] = {m: dict(Counter(g["graded"] for g in games if g["map"] == m)) for m in maps}
        (out / "round.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
        print(f"round {args.round}: {table['results']}")
        print(f"graded (games at time by who is ahead at the end, supply used {AHEAD:g}x): {table['graded']}")
        for m, c in table["graded_by_map"].items():
            print(f"  {m}: {c}")
        return
    jobs = [(args.vs, name, path, s, args.minutes, out, args.pairs, args.incumbent, args.candidate, args.layer, args.zerg)
            for name, path in maps.items() for s in range(args.first, args.first + args.seeds)]
    with ThreadPoolExecutor(args.parallel) as pool:
        games = list(monitor.track(pool.map(play, jobs), len(jobs), "games"))
    for g in games:
        g["decisions"] = decisions(g["replay"])
        monitor.tally("results", g["result"])
    done = [g["replay"] for g in games if g["result"] != "error"]
    rows_t, rows_z = side_rows(done, "T", args.parallel), side_rows(done, "Z", args.parallel)
    grade(games, rows_t, rows_z)
    table = {"round": args.round, "terran": args.vs, "minutes": args.minutes, "games": games,
             "results": dict(Counter(g["result"] for g in games)),
             "graded": dict(Counter(g["graded"] for g in games)),
             "by_map": {m: dict(Counter(g["result"] for g in games if g["map"] == m)) for m in maps},
             "decisions": {k: sum(g["decisions"].get(k, 0) for g in games) for k in ("army_t", "roles_t", "army_z", "roles_z")},
             "gap": {"T": gaps(rows_t, "T", args.parallel, era), "Z": gaps(rows_z, "Z", args.parallel, era)},
             "decided": sum(1 for g in games if g["result"] in ("won", "lost")) / max(1, len(games))}
    if args.candidate:                    # (the layer's side's gap, its candidate's games against the incumbent's)
        side = "Z" if args.layer == "roles_z" else "T"
        for who, model in (("candidate", args.candidate), ("incumbent", args.incumbent)):
            mine = [g["replay"] for g in games if g["result"] != "error" and g["zroles"] == str(model)]
            rows = rows_z if side == "Z" else rows_t
            table["gap"][f"{side}_{who}"] = gaps({r: rows[r] for r in mine if r in rows}, side, args.parallel, era) if mine else None
    if args.pairs:
        from eval.pairs import summarize
        files = sorted(glob.glob(str(out / "*" / "*.pairs.npz")))
        table["pairs"] = summarize(files, args.incumbent, args.candidate)
    (out / "round.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"round {args.round}: {len(games)} games, Gary-T {table['results']}; by map {table['by_map']}")
    print(f"graded (games at time by who is ahead at the end, supply used {AHEAD:g}x): {table['graded']}")
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
