"""The belief model on league games (Fable's check): its error in the enemy's army supply on Gary's games,
against its error on held-out pro games (train/belief.py's report). Trained on pro games and frozen: a
much larger error on league games means Gary's games aren't the pros' (the opponent's army moves
differently), and its estimates there can't be trusted as an input.

The inputs are rebuilt from each league replay as train/belief.py builds them from the macro training
set: every second, the enemy units in the player's sight (its own fog), its own units, its supply.

    python -m eval.belief_league --race Z --games "runs/league/n1_r*/*/*.rep"
"""

from __future__ import annotations

import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority

REPO = Path(__file__).resolve().parents[1]
HALLS = {"T": {106}, "Z": {131, 132, 133}}


def series(job: tuple):
    """(features (T, F), the enemy's real army supply x2 (T,), frames) for one game, the race's side."""
    rep, race, model_path = job
    from gary.env import Game
    from gary.policy.belief import ARMY_SUPPLY, BeliefModel, features_series
    model = BeliefModel.load(model_path)
    enemy = model.enemy
    g = Game.scenario(rep)
    seen, own, eco, enemy_counts, frames = [], [], [], [], []
    try:
        slot = None
        while True:
            try:
                g.step(24)
            except Exception:
                break
            o = g.observe()
            if slot is None:
                slot = next((u["owner"] for u in o["units"] if u["type"] in HALLS[race]), None)
                if slot is None:
                    continue
            me = next((p for p in o["players"] if p["slot"] == slot), None)
            if me is None or not any(u["owner"] == slot and u["type"] in HALLS[race] for u in o["units"]):
                break
            s, m, e = np.zeros(228), np.zeros(228), np.zeros(228)
            for u in o["units"]:
                if not 0 <= u["type"] < 228 or u["owner"] == 11:
                    continue
                if u["owner"] == slot:
                    m[u["type"]] += 1
                else:
                    e[u["type"]] += 1
                    if (u.get("visible_to", 0) >> slot) & 1:
                        s[u["type"]] += 1
            seen.append(s)
            own.append(m)
            enemy_counts.append(e)
            eco.append([me["minerals"], me["gas"], round(me["supply_used"] * 2), round(me["supply_max"] * 2)])
            frames.append(g.frame)
    finally:
        g.close()
    if len(frames) < 300:
        return None
    X = features_series(np.array(seen), np.array(own), np.array(eco), np.array(frames), enemy, race)
    sup = {t: v for t, v in ARMY_SUPPLY[enemy].items()}
    truth = np.array([sum(c[t] * v for t, v in sup.items()) for c in enemy_counts], np.float32)
    pred = np.array([model.predict(x)["supply"] for x in X], np.float32)
    return pred, truth, np.array(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", default="Z", choices=("T", "Z"))
    ap.add_argument("--games", required=True, help="a glob of league replays")
    ap.add_argument("--model", help="default: the newest belief model of the race")
    ap.add_argument("--workers", type=int, default=2)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    model = args.model or str(sorted((REPO / "runs" / "belief").glob(f"TvZ_{args.race}_*/model.pt"))[-1])
    reps = sorted(glob.glob(args.games))
    with ProcessPoolExecutor(args.workers) as pool:
        got = [r for r in monitor.track(pool.map(series, [(r, args.race, model) for r in reps]), len(reps), "games") if r]
    err = np.concatenate([np.abs(p - t) for p, t, _ in got]) / 2      # (supply, not x2)
    report = json.loads((Path(model).parent / "report.json").read_text()) if (Path(model).parent / "report.json").exists() else {}
    print(f"{len(got)} league games, {len(err)} seconds: mean error in the enemy's army supply {err.mean():.1f} "
          f"(median {np.median(err):.1f})")
    print(f"held-out pro games (train/belief.py): {json.dumps(report.get('model', report))[:400]}")


if __name__ == "__main__":
    main()
