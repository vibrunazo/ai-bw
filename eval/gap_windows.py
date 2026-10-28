"""Two arms' gap to the pros per time window, game against game: is a difference between two versions real, and
where in the game is it? (eval/humanlike.py's gap score, per WINDOW of minutes, with a bootstrap over games.)

An arm is one or more league rounds (runs/league/<round>); games pair across arms by map and seed (the same
game start under both versions). Each round's per-minute feature rows come from its rows.json.gz (eval/league.py
keeps them), or the replays are re-simulated once and the rows kept there. For each window: each arm's gap score
on the paired games, the difference (B - A: positive, B further from the pros), and its 90% interval from
resampling the paired games (BOOT times).

    python -m eval.gap_windows --side T --a B30 B30b --b R30T R30Tb
    python -m eval.gap_windows --side T --a D50a --b R50a --group fight      # the fight features only

--group: all features, the macro ones (economy, unit counts, buildings) or the fight ones (where the army stands,
its orders, threats met, fighting, what it does at a sighting, lone units).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval import monitor, priority  # noqa: E402

WINDOW = 5                      # minutes
FIGHT = ("army_standing", "orders_per_fighter", "army_home", "army_field", "army_enemy", "army_alone", "army_spread",
         "army_gone", "threatened", "threat_met", "response_s", "fighting", "hit_standing", "moving_between_shots",
         "harass_response_s", "harass_hp_lost", "sight_reorder", "sight_gather", "sight_stim", "air_sight_gather",
         "air_sight_stim", "lone_away", "lone_attack", "fight_attended", "fight_reached")


def group_names(names: list[str], group: str) -> list[str]:
    """The minute and a group's features (eval/humanlike.py _names)."""
    if group == "all":
        return names
    fight = [n for n in names if n in FIGHT]
    return ["minute"] + (fight if group == "fight" else [n for n in names[1:] if n not in FIGHT])
BOOT = 200


def round_rows(name: str, side: str, workers: int) -> dict[tuple, list[dict]]:
    """{(map, seed): rows} of one round's games for one side."""
    out = REPO / "runs" / "league" / name
    table = json.loads((out / "round.json").read_text(encoding="utf-8"))
    cache = out / "rows.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            saved = json.load(f)
    else:
        from eval.league import save_rows, side_rows
        done = [g["replay"] for g in table["games"] if g["result"] != "error"]
        rows_t, rows_z = side_rows(done, "T", workers), side_rows(done, "Z", workers)
        save_rows(out, rows_t, rows_z)
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            saved = json.load(f)
    by_rel = saved[side]
    got = {}
    for g in table["games"]:
        if g["result"] == "error":
            continue
        rel = str(Path(g["replay"]).resolve().relative_to(out.resolve())).replace("\\", "/")
        if rel in by_rel:
            got[(g["map"], g["seed"])] = by_rel[rel]
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=("T", "Z", "P"), required=True)
    ap.add_argument("--matchup", default="TvZ", help="the pros it is judged against (PvT, PvZ for the Protoss)")
    ap.add_argument("--a", nargs="+", required=True, help="arm A's rounds")
    ap.add_argument("--b", nargs="+", required=True, help="arm B's rounds")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--group", choices=("all", "macro", "fight"), default="all")
    ap.add_argument("--workers", type=int, default=8)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from eval.humanlike import _names, gap_score, matrix, pro_rows
    arm = {}
    for key, rounds in (("a", args.a), ("b", args.b)):
        arm[key] = {}
        for r in rounds:
            arm[key].update(round_rows(r, args.side, args.workers))
    keys = sorted(set(arm["a"]) & set(arm["b"]))
    names = group_names(_names(args.side), args.group)
    has = lambda key, n: any(r.get(n) == r.get(n) and r.get(n) is not None for k in keys for r in arm[key][k])
    names = [n for n in names if n == "minute" or (has("a", n) and has("b", n))]   # (a feature both arms have)
    Xp, _, _ = matrix(pro_rows(args.side, 300, args.workers, "modern", args.matchup), names)
    windows = [(a, a + WINDOW - 1) for a in range(0, args.minutes, WINDOW)]

    def score(key: str, ks: list[tuple]) -> list:
        X, _, _ = matrix({f"{k[0]}|{k[1]}|{i}": arm[key][k] for i, k in enumerate(ks)}, names)
        return [None if v is None else v[0] for v in gap_score(X, Xp, buckets=windows)]

    base_a, base_b = score("a", keys), score("b", keys)
    rng = random.Random(0)
    diffs = [[] for _ in range(len(windows) + 1)]
    for _ in monitor.track(range(BOOT), BOOT, "bootstrap"):
        ks = [rng.choice(keys) for _ in keys]
        sa, sb = score("a", ks), score("b", ks)
        for i, (x, y) in enumerate(zip(sa, sb)):
            if x is not None and y is not None:
                diffs[i].append(y - x)
    print(f"{args.side}: {len(keys)} paired games; A {' '.join(args.a)}, B {' '.join(args.b)}; gap (lower: nearer the pros)"
          + ("" if args.group == "all" else f", {args.group} features ({len(names) - 1})"))
    print(f"{'window':>10} {'A':>6} {'B':>6} {'B-A':>7}   90% interval")
    for i, label in enumerate([f"{a}-{b + 1} min" for a, b in windows] + ["all"]):
        x, y, d = base_a[i], base_b[i], diffs[i]
        if x is None or y is None:
            print(f"{label:>10}      -      -       -")
            continue
        lo, hi = (np.percentile(d, [5, 95]) if d else (float("nan"), float("nan")))
        mark = "  (B further)" if lo > 0 else "  (B nearer)" if hi < 0 else ""
        print(f"{label:>10} {x:6.3f} {y:6.3f} {y - x:+7.3f}   [{lo:+.3f}, {hi:+.3f}]{mark}")


if __name__ == "__main__":
    main()
