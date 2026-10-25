"""The pipe check for the macro (docs/SELF_PLAY.md, rule 1; issue #47): what the macro model decides, does it get
built, and how soon?

For a round's games: each of the Terran's macro decisions (the records' macro decisions: what, who decided it, at
what odds) is matched to the first unit or building of that type that appears for the Terran after it (the replay
re-simulated; a building appears when placed, a unit when trained), within WINDOW_S. Per decision kind:

  decided       decisions a game
  carried out   the share matched to one that appeared
  delay         the median seconds from the decision to its appearing (a unit's includes its training time)
  lost to       the orders' fates the records give for the rest: dropped (and why), refused by the game, retried

    python -m eval.macro_pipe DC100a --workers 4
    python -m eval.macro_pipe ZP141a --side Z           # the Zerg's (a unit counts once it hatches from its egg)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval import monitor, priority  # noqa: E402

WINDOW_S = 120                  # a decision not carried out within this is lost
STEP = 24                       # frames between looks


RACE_ID = {"Z": 0, "T": 1, "P": 2}             # the game's race ids


def game(job: tuple) -> dict | None:
    path, side = job
    from gary.env import Game
    from gary.race import Race
    rec = Path(path.replace(".rep", ".records.jsonl"))
    if not rec.exists():
        return None
    names = {v: k for k, v in Race.of(side).names.items()}   # name -> type id
    decisions, losses = [], Counter()
    with open(rec, encoding="utf-8") as f:
        for line in f:
            if '"macro"' not in line and '"order"' not in line:
                continue
            r = json.loads(line)
            if r.get("side") != side:
                continue
            if r.get("kind") == "decision" and r.get("layer") == "macro" and r.get("choice") in names:
                decisions.append((r["f"], names[r["choice"]], str(r.get("by", ""))))
            elif r.get("kind") == "order" and r.get("fate") in ("dropped", "refused", "retried"):
                losses[(r["fate"], str(r.get("why", ""))[:48])] += 1
    try:
        g = Game.scenario(path)
    except Exception:
        return None
    slot = None
    born: dict[int, list[int]] = defaultdict(list)   # type -> frames a new one appeared
    seen = set()                                     # (tag, type): a Zerg unit keeps its larva's tag, a building its drone's
    end = (max((d[0] for d in decisions), default=0) + WINDOW_S * 24) if decisions else 0
    while g.frame < end:
        try:
            g.step(STEP)
        except Exception:
            break
        o = g.observe()
        if slot is None:
            slot = next((p["slot"] for p in o["players"] if p["race"] == RACE_ID[side]), None)
        for u in o["units"]:
            if u["owner"] == slot and (u["tag"], u["type"]) not in seen:
                seen.add((u["tag"], u["type"]))
                if g.frame > STEP:                     # (the start's units aren't anyone's decision)
                    born[u["type"]].append(g.frame)
    g.close()
    used: dict[int, int] = defaultdict(int)
    rows = []
    for f, t, by in sorted(decisions):
        frames = born.get(t, [])
        k = used[t]
        while k < len(frames) and frames[k] < f:
            k += 1
        if k < len(frames) and frames[k] - f <= WINDOW_S * 24:
            rows.append((t, by, (frames[k] - f) / 24))
            used[t] = k + 1
        else:
            rows.append((t, by, None))
            used[t] = k
    return {"rows": rows, "losses": losses}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rounds", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--side", default="T", choices=["T", "Z"], help="whose macro (the league plays T against Z)")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from gary.race import Race
    name = Race.of(args.side).names
    for r in args.rounds:
        reps = sorted(glob.glob(str(REPO / "runs" / "league" / r / "*" / "*.rep")))
        with ProcessPoolExecutor(args.workers) as pool:
            res = [x for x in monitor.track(pool.map(game, [(p, args.side) for p in reps]), len(reps), f"{r} replays") if x]
        rows = [row for x in res for row in x["rows"]]
        losses = Counter()
        for x in res:
            losses.update(x["losses"])
        n = len(res)
        done = [row for row in rows if row[2] is not None]
        print(f"{r}: {n} games, {len(rows) / n:.0f} macro decisions a game, carried out within {WINDOW_S} s "
              f"{len(done) / max(1, len(rows)):.2f}")
        by_type = defaultdict(list)
        for t, by, d in rows:
            by_type[t].append(d)
        print(f"  {'decision':>22} {'a game':>7} {'carried out':>12} {'median delay s':>15}")
        for t, ds in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:16]:
            got = sorted(d for d in ds if d is not None)
            med = got[len(got) // 2] if got else float("nan")
            print(f"  {name.get(t, t):>22} {len(ds) / n:7.1f} {len(got) / len(ds):12.2f} {med:15.0f}")
        by_who = defaultdict(list)
        for t, by, d in rows:
            by_who[by.split(":")[0][:20]].append(d is not None)
        print("  by who decided: " + ", ".join(f"{k} {len(v) / n:.0f} a game ({sum(v) / len(v):.2f} carried out)"
                                               for k, v in sorted(by_who.items(), key=lambda kv: -len(kv[1]))))
        print("  the orders' losses a game: " + "; ".join(f"{f} {w!r} {c / n:.1f}" for (f, w), c in losses.most_common(8)))


if __name__ == "__main__":
    main()
