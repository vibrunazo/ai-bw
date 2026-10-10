"""The hand rules' role decisions as a dataset, to distill them into the role model (docs/CONTROL.md).

Reinforcement learning needs a model to start from; the best role decisions so far are v0.10's
hand rules (gary/agent/roles.py), better than the model learned from the pros. So Gary v0.10 plays
the home-defense scenarios (eval/scenarios.py, the train split) with its decision log open, and
every turn becomes a training example: the role model's inputs (the snapshot as Gary saw it, the
units' motion) and, for each of Gary's units the roles could command, the rules' decision (the
role of the group they commanded, else keep). Same format as ingest/role_dataset.py, labels in
gary.policy.roles.DECISIONS.

    python -m ingest.role_distill --parallel 16
    python -m train.roles --data data/interim/roles/distill_train.npz
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gary.policy.roles import DECISIONS, EXTRA, MAX_UNITS


def _job(args: tuple) -> dict:
    from eval.scenarios import run_one
    sc, version, seconds, hands = args
    row = run_one(sc, "gary", seconds, version, 1, sample_seed=0, hands_seed=1 + hands)
    return {"sha1": sc["sha1"], "error": row.get("error"), "log": row.get("log", [])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v010", help="the Gary whose role decisions are distilled")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=45, help="how long each scenario plays")
    ap.add_argument("--draws", type=int, default=1, help="plays of each, Gary's hands seeded apart (more varied data)")
    ap.add_argument("--out", help="default: data/interim/roles/distill_<split>.npz")
    args = ap.parse_args()
    from eval.scenarios import scenarios_path
    from ingest.inventory import data_root
    scs = [json.loads(line) for line in scenarios_path(args.split).read_text(encoding="utf-8").splitlines()]
    with ProcessPoolExecutor(args.parallel) as pool:
        results = list(pool.map(_job, [(sc, args.version, args.seconds, h) for sc in scs for h in range(args.draws)],
                                chunksize=args.draws))
    turns = [(gi, d) for gi, r in enumerate(results) for d in r["log"] if len(d["rows"]) and d["units"]]
    n = len(turns)
    rows = np.zeros((n, MAX_UNITS, 11), np.int16)
    extra = np.zeros((n, MAX_UNITS, EXTRA), np.float32)
    labels = np.full((n, MAX_UNITS), -1, np.int8)
    count = np.zeros(n, np.int16)
    time_s = np.zeros(n, np.float32)
    game = np.zeros(n, np.int32)
    for i, (gi, d) in enumerate(turns):
        k = min(len(d["rows"]), MAX_UNITS)
        rows[i, :k], extra[i, :k], count[i], time_s[i], game[i] = d["rows"][:k], d["extra"][:k], k, d["time"], gi
        for j, y in zip(d["units"], d["roles"]):
            if j < MAX_UNITS:
                labels[i, j] = y
    out = Path(args.out) if args.out else data_root() / "interim" / "roles" / f"distill_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=rows, extra=extra, labels=labels, count=count, time=time_s, game=game,
                        sha1=np.array([r["sha1"] for r in results]), names=np.array(DECISIONS))
    y = labels[labels >= 0]
    print(f"{len(results)} scenarios ({sum(1 for r in results if r['error'])} errors), {n} turns, {len(y)} unit decisions -> {out}")
    print("  " + ", ".join(f"{DECISIONS[k]} {np.mean(y == k):.1%}" for k in range(len(DECISIONS))))


if __name__ == "__main__":
    main()
