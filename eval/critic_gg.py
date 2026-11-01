"""Would Gary's gg have been right? The critic's win odds on its held-out league games (issue #62).

The gg rule (#62): when the critic's win odds stay at or under a threshold for a minute, type gg and leave. Before it
plays, how often a game read that low was won anyway. For each critic (the newest per matchup and race, or the runs
named), on the games it held out (train/value.py: every fifth seed), the moments read every GARY_EVERY_S seconds:

  calibration   the odds read against how the decided games ended, by bins of odds
  would gg      the games where the odds stayed at or under each threshold for HOLD_S: how they ended (won, lost,
                at time), and when the first such minute ended

    python -m eval.critic_gg                          # every matchup's newest critic
    python -m eval.critic_gg --runs PvZ_P_20261101-015437 --thresholds 0.1 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HOLD_S = 60
BINS = (0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)


def newest_runs() -> list[Path]:
    runs: dict[str, Path] = {}
    for p in sorted((REPO / "runs" / "value").glob("*_*_*")):
        if (p / "model.pt").exists() and (p / "gary.jsonl").exists():
            runs["_".join(p.name.split("_")[:2])] = p
    return list(runs.values())


def held_out(run: Path) -> list[dict]:
    with open(run / "gary.jsonl", encoding="utf-8") as f:
        games = [json.loads(line) for line in f]
    return [g for g in games if len(g["x"]) and int(g["game"].split("/")[-1]) % 5 == 0]


def report(run: Path, thresholds: list[float]) -> None:
    from gary.policy.value import ValueModel
    from train.value import GARY_EVERY_S
    model = ValueModel.load(run / "model.pt")
    games = held_out(run)
    hold = HOLD_S // GARY_EVERY_S
    odds, ended = [], []
    for g in games:
        p = 1 / (1 + np.exp(-model.predict(np.array(g["x"], np.float32))["win"]))
        g["p"] = p
        if g["result"] != "time":
            odds.append(p)
            ended.append(np.full(len(p), g["result"] == "won"))
    print(f"{run.name}: {len(games)} held-out games ({sum(g['result'] == 'won' for g in games)} won, "
          f"{sum(g['result'] == 'lost' for g in games)} lost, {sum(g['result'] == 'time' for g in games)} at time)")
    if odds:
        p, y = np.concatenate(odds), np.concatenate(ended)
        cells = []
        for lo, hi in zip(BINS, BINS[1:]):
            m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if m.sum() >= 20:
                cells.append(f"{lo:.2f}-{hi:.2f} won {y[m].mean():.2f} ({m.sum()})")
        print("  calibration (decided games' moments): " + "; ".join(cells))
    for t in thresholds:
        out = {"won": 0, "lost": 0, "time": 0}
        when = []
        for g in games:
            low = g["p"] <= t
            run_len = np.convolve(low.astype(int), np.ones(hold, int), "valid") if len(low) >= hold else []
            hit = np.flatnonzero(np.asarray(run_len) == hold)
            if len(hit):
                out[g["result"]] += 1
                when.append(g["frames"][hit[0] + hold - 1] / 1440)
        n = sum(out.values())
        print(f"  would gg at <= {t:.0%} for {HOLD_S} s: {n} games, won {out['won']}, lost {out['lost']}, at time "
              f"{out['time']}" + (f"; first at {np.median(when):.1f} min (median)" if when else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="runs/value folders (default: the newest per matchup and race)")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.1, 0.2])
    args = ap.parse_args()
    for run in [REPO / "runs" / "value" / r for r in args.runs] if args.runs else newest_runs():
        report(run, args.thresholds)


if __name__ == "__main__":
    main()
