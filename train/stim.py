"""Train the stim model (gary/policy/stim.py) on ingest/stim_dataset.py's samples.

Gradient-boosted trees on each unit's view of the fight -> P(the pro stims it in the next half
second). Split by game (a game's samples all train or all test). The threshold: Gary stims a unit
when P passes it, chosen so Gary stims as often as the pros did in the test games' fights.

    python -m train.stim
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="default: data/interim/stim/stim_v1.npz")
    ap.add_argument("--out", default="runs/stim")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    from inventory import data_root
    d = np.load(args.data or data_root() / "interim" / "stim" / "stim_v2.npz")
    X, y, game, names = d["X"], d["y"], d["game"], list(d["names"])
    test = game % 5 == 0
    print(f"{len(y)} samples from {len(set(game))} games, stimmed {y.mean():.2%}; test games {len(set(game[test]))}")
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                         early_stopping=False, random_state=0)
    t0 = time.time()
    clf.fit(X[~test], y[~test])
    p = clf.predict_proba(X[test])[:, 1]
    rate = y[test].mean()
    threshold = float(np.quantile(p, 1 - rate))           # as many stims as the pros
    hit = (p >= threshold)
    precision = (y[test][hit] == 1).mean() if hit.any() else 0.0
    recall = hit[y[test] == 1].mean() if (y[test] == 1).any() else 0.0
    print(f"trained in {time.time() - t0:.0f} s; test AUC {roc_auc_score(y[test], p):.3f}, "
          f"average precision {average_precision_score(y[test], p):.3f} (base rate {rate:.3f})")
    print(f"threshold {threshold:.3f} (stims as often as the pros: {hit.mean():.2%}): precision {precision:.2f}, "
          f"recall {recall:.2f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"stim_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": clf, "threshold": threshold, "names": names, "test_auc": float(roc_auc_score(y[test], p))}, f)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
