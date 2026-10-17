"""The league's nights (docs/SELF_PLAY.md "The loop"): rounds of Gary-T against Gary-Z with paired fight
episodes, each round training a candidate for Gary-Z's fight roles and judging the last one. One RL
target: the fight roles (eval/pairs.py, train/league_rl.py train_pairs); the army layers stay at
imitation (their decisions logged for later) until the critic predicts more than who is ahead.

A round (fresh seeds every round: never judged on the seeds it trained on):
  play     eval/league.py --pairs: the incumbent fight-role model plays even seeds, the candidate odd
           seeds; at fights, paired arms (incumbent drawing twice: training; incumbent against the
           candidate, the same dice: A/B)
  gate     the candidate is kept only if the A/B pairs favor it (mean above zero, AB_MIN pairs or more)
           and Gary-Z's gap score in its own games is no worse than in the incumbent's (GAP_TOLERANCE);
           else the incumbent stays (the candidate's checkpoint kept, not used)
  train    a new candidate from this round's training pairs, from the incumbent, KL-leashed to the
           imitation model
  record   state.json (resumable) with every round: results, decided share, both sides' gap score, the
           A/B evidence, the arms' retreat share (field fights and fights with stakes) and moves, the KL
           to the imitation model; the monitor shows each round's summary line

    python -m eval.league_loop --name n1 --rounds 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from eval import monitor, priority

REPO = Path(__file__).resolve().parents[1]
IMITATION = REPO / "runs" / "roles" / "20261016-212927" / "model.pt"    # the Zerg role model from the pros
AB_MIN = 20
GAP_TOLERANCE = 0.02


def band_mean(gap) -> float | None:
    """The gap score's mean over 5-20 minutes (the ranges with fights)."""
    vals = [v[0] for v in (gap or [])[1:5] if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the run's name (runs/league_loop/<name>)")
    ap.add_argument("--rounds", type=int, default=4, help="rounds to play now (the run resumes where it was)")
    ap.add_argument("--vs", default="v029+draw")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per map per round")
    ap.add_argument("--base", type=int, default=1001, help="the first round's first seed")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--parallel", type=int, default=4)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from train.league_rl import train_pairs
    folder = REPO / "runs" / "league_loop" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"round": 0, "incumbent": str(IMITATION), "candidate": None, "imitation": str(IMITATION), "history": []}
    for _ in monitor.track(range(args.rounds), args.rounds, "rounds"):
        k = state["round"]
        rnd = f"{args.name}_r{k}"
        first = args.base + k * args.seeds
        cmd = [sys.executable, "-m", "eval.league", "--round", rnd, "--vs", args.vs, "--seeds", str(args.seeds),
               "--first", str(first), "--minutes", str(args.minutes), "--parallel", str(args.parallel), "--pairs",
               "--incumbent", state["incumbent"]] + (["--candidate", state["candidate"]] if state["candidate"] else [])
        t0 = time.time()
        subprocess.run(cmd, cwd=REPO, check=True)
        table = json.loads((REPO / "runs" / "league" / rnd / "round.json").read_text())
        pairs = table.get("pairs", {})
        entry = {"round": k, "seeds": [first, first + args.seeds - 1], "incumbent": state["incumbent"],
                 "candidate": state["candidate"], "results": table["results"], "decided": table["decided"],
                 "gap": table["gap"], "pairs": pairs, "minutes_wall": round((time.time() - t0) / 60, 1)}
        # the gate
        if state["candidate"]:
            ab_ok = (pairs.get("pairs_ab", 0) >= AB_MIN and (pairs.get("ab_mean") or 0) > 0)
            gc, gi = band_mean(table["gap"].get("Z_candidate")), band_mean(table["gap"].get("Z_incumbent"))
            gap_ok = gc is not None and gi is not None and gc <= gi + GAP_TOLERANCE
            entry["gate"] = {"ab_ok": ab_ok, "gap_ok": gap_ok, "gap_candidate": gc, "gap_incumbent": gi,
                             "kept": ab_ok and gap_ok}
            if ab_ok and gap_ok:
                state["incumbent"] = state["candidate"]
        # the next candidate
        report = train_pairs([rnd], Path(state["incumbent"]), IMITATION, folder / f"r{k}_candidate")
        entry["train"] = report
        state["candidate"] = str(folder / f"r{k}_candidate" / "model.pt") if report.get("decisions") else None
        state["history"].append(entry)
        state["round"] = k + 1
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        g = entry.get("gate", {})
        arms = pairs.get("arms", {})
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"round {k}: Gary-T {table['results']}, decided {table['decided']:.0%}; gap T {fmt(band_mean(table['gap']['T']))} "
              f"Z {fmt(band_mean(table['gap']['Z']))}; pairs training {pairs.get('pairs_training')} A/B {pairs.get('pairs_ab')} "
              f"(candidate - incumbent {pairs.get('ab_mean')}, se {pairs.get('ab_se')}); gate {g or '-'}; "
              f"retreat in field fights {({m: a.get('field_away') for m, a in arms.items()})}; "
              f"new candidate from {report.get('decisions', 0)} decisions, KL to imitation {report.get('kl_to_imitation')}",
              flush=True)


if __name__ == "__main__":
    main()
