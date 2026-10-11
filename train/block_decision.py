"""The decision to close the main's ramp with bodies (gary/agent/block.py), learned from branches.

eval/block_branch.py played moments of Gary's games on twice, without and with the block: the gain
of blocking is the difference of their scores over the 90 s after (value killed minus lost). A
small model (gradient-boosted trees) predicts it from what Gary saw then (block.FEATURES); in play
Gary blocks where it predicts a gain (fight.py BlockRule, option decide).

Judged as it would be used, on games it wasn't trained on (folds by game): the gain of blocking
where it says to, against never blocking (0) and always blocking.

    python -m train.block_decision                  # -> runs/block_decision/<stamp>/model.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics as st
import time
from pathlib import Path

import numpy as np

from eval.block_branch import out_path
from gary.agent.block import FEATURES

REPO_ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    X = np.array([[r["features"][k] for k in FEATURES] for r in rows], np.float32)
    y = np.array([r["block"]["net"] - r["base"]["net"] for r in rows], np.float32)
    games = np.array([hash((r["opponent"], r["map"], r["seed"])) for r in rows])
    return rows, X, y, games


def model():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     min_samples_leaf=10, random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(out_path()))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    from sklearn.model_selection import GroupKFold
    rows, X, y, games = load(Path(args.data))
    print(f"{len(rows)} decision points from {len(set(games))} games; blocking better in {np.mean(y > 0):.0%}, "
          f"worse in {np.mean(y < 0):.0%}; mean gain {y.mean():+.0f} (median {np.median(y):+.0f})")
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(args.folds).split(X, y, games):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    chosen = pred > 0
    print(f"held out: blocks at {chosen.mean():.0%} of the points; gain where it blocks {y[chosen].mean() if chosen.any() else 0:+.0f}, "
          f"where it doesn't {y[~chosen].mean() if (~chosen).any() else 0:+.0f}")
    print(f"  per point: never 0, always {y.mean():+.0f}, the model {np.where(chosen, y, 0).mean():+.0f}, "
          f"perfect {np.maximum(y, 0).mean():+.0f}")
    for key in ("opponent", "map"):
        for v in sorted({r[key] for r in rows}):
            k = np.array([r[key] == v for r in rows])
            print(f"    {v:18s} points {k.sum():4d}  always {y[k].mean():+6.0f}  model {np.where(chosen, y, 0)[k].mean():+6.0f}  "
                  f"blocks at {chosen[k].mean():.0%}")
    final = model().fit(X, y)
    imp = sorted(zip(FEATURES, final.feature_importances_), key=lambda t: -t[1])
    print("  features by importance:", ", ".join(f"{k} {v:.2f}" for k, v in imp))
    out = REPO_ROOT / "runs" / "block_decision" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump({"model": final, "features": FEATURES}, f)
    print(f"-> {out / 'model.pkl'}")


if __name__ == "__main__":
    main()
