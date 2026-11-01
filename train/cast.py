"""Train a spell's cast models (gary/policy/cast.py) on ingest/cast_dataset.py's samples.

Gradient-boosted trees, split by game (every fifth game tests), as train/stim.py:
  when   a caster's view of the fight -> P(the pro casts with it in the next second); Gary draws at these odds (the
         mean odds on the test games against their cast rate: calibrated)
  where  each candidate spot's view -> a score, a softmax over the cast's spots trained on the one the pro cast at
         (it beat trees scoring each spot alone: top-1 0.25 against 0.18); Gary casts at the best. Judged on the test
         casts' top-1 against the plain ranking (enemy supply hit less own) and a random spot, and on what the spots
         chosen hit against the pros'

    python -m train.cast --spell storm
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def train_where(S: np.ndarray, start: np.ndarray, label: np.ndarray, train: np.ndarray, epochs: int = 300):
    """The where model: a small network scoring each spot, a softmax over a cast's spots (the pro's the target)."""
    import torch
    from gary.policy.cast import SpotScorer
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Z = torch.tensor((S - mu) / sd, dtype=torch.float32)
    grp = torch.tensor(np.repeat(np.arange(len(label)), np.diff(start)))
    tgt = torch.tensor(start[:-1] + label)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(S.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    cast = torch.tensor(np.flatnonzero(train))
    for _ in range(epochs):
        opt.zero_grad()
        s = net(Z).squeeze(1)
        top = torch.full((len(label),), -1e9).scatter_reduce(0, grp, s, "amax")
        den = torch.zeros(len(label)).index_add_(0, grp, torch.exp(s - top[grp]))
        loss = -(s[tgt] - top - torch.log(den))[cast].mean()
        loss.backward()
        opt.step()
    layers = [m for m in net if isinstance(m, torch.nn.Linear)]
    return SpotScorer(mu, sd, [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in layers])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spell", default="storm")
    ap.add_argument("--data", help="default: data/interim/cast/<spell>_v1.npz")
    ap.add_argument("--out", default="runs/cast")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    from gary.policy.cast import spot_value
    from inventory import data_root
    d = np.load(args.data or data_root() / "interim" / "cast" / f"{args.spell}_v1.npz")
    from gary.policy.cast import CAST_NAMES, SPOT_NAMES
    stored = list(d["names"])                     # (the inputs as the module has them now: columns dropped since)
    X, y, game = d["X"][:, [stored.index(n) for n in CAST_NAMES]], d["y"], d["game"]
    test = game % 5 == 0
    print(f"when: {len(y)} samples from {len(set(game))} games, cast {y.mean():.2%}; test games {len(set(game[test]))}")
    when = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, early_stopping=False,
                                          random_state=0)
    t0 = time.time()
    when.fit(X[~test], y[~test])
    p = when.predict_proba(X[test])[:, 1]
    rate = y[test].mean()
    auc = float(roc_auc_score(y[test], p))
    print(f"  trained in {time.time() - t0:.0f} s; test AUC {auc:.3f}, average precision "
          f"{average_precision_score(y[test], p):.3f} (base rate {rate:.3f}); mean odds {p.mean():.3f} (Gary draws at "
          f"them: as many casts as the pros where the odds are calibrated)")

    stored = list(d["spot_names"])
    S, start, label, s_game = d["S"][:, [stored.index(n) for n in SPOT_NAMES]], d["s_start"], d["s_label"], d["s_game"]
    is_test = s_game % 5 == 0
    print(f"where: {len(label)} casts, {len(S)} spots ({len(S) / max(1, len(label)):.1f} a cast); test casts "
          f"{is_test.sum()}")
    where = train_where(S, start, label, ~is_test)
    q = where.scores(S)
    top = plain = rand = 0.0
    chosen, pros = [], []
    for i in np.flatnonzero(is_test):
        a, b = start[i], start[i + 1]
        j = int(np.argmax(q[a:b]))
        top += j == label[i]
        plain += np.argmax(spot_value(S[a:b])) == label[i]
        rand += 1 / (b - a)
        chosen.append(S[a + j])
        pros.append(S[a + label[i]])
    n = max(1, int(is_test.sum()))
    print(f"  test top-1: model {top / n:.2f}, plain ranking {plain / n:.2f}, random {rand / n:.2f}")
    print("  the spot chosen, test casts' means (model / pros): " + "; ".join(
        f"{k} {c:.2f} / {pr:.2f}" for k, c, pr in zip(SPOT_NAMES, np.mean(chosen, 0), np.mean(pros, 0))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.spell}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"spell": args.spell, "when": when, "where": where,
                     "names": CAST_NAMES, "spot_names": SPOT_NAMES, "test_auc": auc,
                     "test_top1": top / n}, f)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
