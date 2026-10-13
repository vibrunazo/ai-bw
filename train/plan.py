"""The plan policy as it starts (gary/policy/plan.py; docs/CONTROL.md step 7): the posture's rules,
with a learned correction of zero, for whole-game training to improve.

Gary plays its usual opponents with the rules deciding the mode (gary/agent/posture.py
LearnedPosture with no model: v6's rules), recording every second the facts the policy sees, the
mode it was in and the rules' choice: the scale of each input (the correction's normalization), and
how often the rules change mode. The policy's checkpoint: the rules' vote and a zero correction.

    python -m train.plan --seeds 16                 # -> runs/plan/TvZ_T_<stamp>/model.pt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from eval import monitor, priority
from gary.policy.plan import MODES, PlanModel

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
OPPONENTS = ("dummy", "v2", "replayed")
VERSION = "v018+watch+flood"
MINUTES = 10


def collect(args: tuple) -> dict:
    """One game with the rules deciding: the policy's input every second (its last part: the rules' mode)."""
    from eval.walls_bench import Match
    from gary.agent.posture import LearnedPosture
    opponent, mp, seed = args
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 0, VERSION, MAPS[mp], opponent)
    rec: list = []
    m.gary.posture = LearnedPosture(m.gary, record=rec)
    end = MINUTES * 60 * 24
    try:
        while m.game.frame < end and not m.done:
            if m.game.frame % 240 < 2 and m.over():
                break
            m.step()
    finally:
        m.game.close()
    return {"game": f"{opponent}/{mp}/{seed}", "x": [x.tolist() for x, _ in rec]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--data", help="the games collected before (JSON), instead of playing them")
    ap.add_argument("--parallel", type=int, default=14)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    data = Path(args.data) if args.data else REPO_ROOT / "runs" / "plan" / "rules_games.json"
    if args.data and data.exists():
        games = json.loads(data.read_text(encoding="utf-8"))
    else:
        jobs = [(o, mp, s) for o in OPPONENTS for mp in MAPS for s in range(1, args.seeds + 1)]
        with ProcessPoolExecutor(args.parallel) as pool:
            games = list(monitor.track(pool.map(collect, jobs), len(jobs), "games"))
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(json.dumps(games), encoding="utf-8")
    X = np.array([x for g in games for x in g["x"]], np.float32)
    n = len(MODES)
    mean, std = X.mean(0), X.std(0) + 1e-3
    mean[-2 * n:], std[-2 * n:] = 0.0, 1.0           # (the one-hots as they are)
    rules, prev = X[:, -n:].argmax(1), X[:, -2 * n:-n].argmax(1)
    model = PlanModel.from_rules(mean, std)
    p = model.probs(X)
    agree = np.mean(p.argmax(1) == rules)
    print(f"{len(games)} games, {len(X)} seconds; the rules' modes {dict(zip(MODES, np.bincount(rules, minlength=n).tolist()))}, "
          f"{int((rules != prev).sum())} changes; the policy as it starts picks the rules' mode {agree:.1%} of seconds "
          f"(p {p[np.arange(len(p)), rules].mean():.3f})")
    out = REPO_ROOT / "runs" / "plan" / f"TvZ_T_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({**model.ckpt, "modes": MODES, "source": f"the rules of {VERSION}, zero correction", "games": len(games)},
               out / "model.pt")
    print(f"-> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
