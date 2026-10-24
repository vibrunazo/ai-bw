"""The pipe check for the army's commit decision (docs/SELF_PLAY.md, "Learning beyond imitation", rule 1): does an
order the commit model sends move its units where it said? Before concluding a model doesn't learn, the decisions it
makes have to reach the game.

For each of the Terran's army orders from the commit model in a round's games (the records' "army order" lines: the
units, the target, what sent it), the replay is re-simulated and the order's units are looked at WINDOW_S later:

  progress    how much of the way to the target they covered (their mean distance's fall over the start's)
  reached     the share of the order's living units within REACHED of the target
  died        the share dead by then
  taken       the share given another army order before then (by the commit model again, the hunt, a rule...)

Engage and move orders are reported apart; a working pipe moves both, and a pushed decision (an option shifting
the model's odds) shows in which orders go out, not in whether they arrive.

    python -m eval.commit_pipe DC100a EB120a --workers 4
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval import monitor, priority  # noqa: E402

WINDOW_S = 30
REACHED = 6 * 32                # px: a unit this near the target has got there
STEP = 12                       # frames between looks
MIN_DIST = 8 * 32               # orders whose units start this near the target are left out (nothing to cover)


def _orders(rec: Path) -> list[dict]:
    out = []
    with open(rec, encoding="utf-8") as f:
        for line in f:
            if '"army order"' not in line:
                continue
            r = json.loads(line)
            if r.get("side") != "T":
                continue
            out.append({"f": r["f"], "by": r.get("by", "?"), "tags": list(r["in"]["tags"]),
                        "to": tuple(c * 32 + 16 for c in r["in"]["to"])})
    return sorted(out, key=lambda o: o["f"])


def game(path: str) -> list[dict] | None:
    """One replay's commit orders, each with its units' progress, reach, deaths and takeovers."""
    from gary.env import Game
    rec = Path(path.replace(".rep", ".records.jsonl"))
    if not rec.exists():
        return None
    orders = _orders(rec)
    commit = [o for o in orders if str(o["by"]).startswith("commit")]
    if not commit:
        return []
    try:
        g = Game.scenario(path)
    except Exception:
        return None
    later: dict[int, list[int]] = {}                 # tag -> frames of every army order it got
    for o in orders:
        for t in o["tags"]:
            later.setdefault(t, []).append(o["f"])
    looks = sorted({o["f"] for o in commit} | {o["f"] + WINDOW_S * 24 for o in commit})
    seen: dict[int, dict[int, tuple[int, int]]] = {}  # look frame -> tag -> position
    i = 0
    end = looks[-1] + STEP
    while g.frame < end and i < len(looks):
        try:
            g.step(STEP)
        except Exception:
            break
        if g.frame >= looks[i]:
            pos = {u["tag"]: (u["x"], u["y"]) for u in g.observe()["units"]}
            while i < len(looks) and g.frame >= looks[i]:
                seen[looks[i]] = pos
                i += 1
    g.close()
    out = []
    for o in commit:
        a, b = seen.get(o["f"]), seen.get(o["f"] + WINDOW_S * 24)
        if a is None or b is None:
            continue
        units = [t for t in o["tags"] if t in a]
        if not units:
            continue
        d0 = [math.dist(a[t], o["to"]) for t in units]
        if sum(d0) / len(d0) < MIN_DIST:
            continue
        alive = [t for t in units if t in b]
        d1 = [math.dist(b[t], o["to"]) for t in alive]
        taken = [t for t in units if any(o["f"] < f2 < o["f"] + WINDOW_S * 24 for f2 in later.get(t, []))]
        out.append({"by": o["by"], "n": len(units),
                    "progress": (sum(d0) / len(d0) - sum(d1) / len(d1)) / (sum(d0) / len(d0)) if alive else None,
                    "reached": sum(1 for d in d1 if d < REACHED) / len(alive) if alive else None,
                    "died": 1 - len(alive) / len(units), "taken": len(taken) / len(units)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rounds", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    for r in args.rounds:
        reps = sorted(glob.glob(str(REPO / "runs" / "league" / r / "*" / "*.rep")))
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [x for res in monitor.track(pool.map(game, reps), len(reps), f"{r} replays") if res for x in res]
        print(f"{r}: {len(reps)} games, {len(rows)} commit orders with {WINDOW_S} s after them")
        for kind in sorted({x["by"] for x in rows}):
            k = [x for x in rows if x["by"] == kind]
            print(f"  {kind:>14}: {len(k) / len(reps):5.1f} a game, units {mean([x['n'] for x in k]):.1f}; progress "
                  f"{mean([x['progress'] for x in k if x['progress'] is not None]):.2f}, reached "
                  f"{mean([x['reached'] for x in k if x['reached'] is not None]):.2f}, died {mean([x['died'] for x in k]):.2f}, "
                  f"taken by another order {mean([x['taken'] for x in k]):.2f}")


if __name__ == "__main__":
    main()
