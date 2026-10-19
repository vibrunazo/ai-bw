"""The macro model trained on what its decisions led to (docs/CONTROL.md step 7), from branches.

Gary (--version) plays its usual opponents with the macro choosing its favorite as always, every
decision logged with the decisions it could have taken (gary/agent/macro.py log). At sampled
decision moments (POINTS a game, between FROM and UNTIL) the same game is played again (games are
deterministic per seed: identical up to the moment) with another of those decisions taken there
(drawn from the model's own odds, macro.py force), everything else as before. Each pair is scored
over the HORIZON seconds after the moment (eval/score.py: value killed minus lost): the gain of the
other decision over the favorite. The model is then trained on the pairs: its preference between
the two moved toward the better one (a logistic loss on the difference of their logits, weighted by
how much better, clipped), and kept near where it started (a KL penalty on the same states).

Games are chaotic (half a second of Gary's hands moves the next 90 s by +-425 net value,
eval/block_branch.py), so one pair says little; thousands do. Judged on held-out games: how often
the trained model prefers the better decision of a pair (chance: 50%).

    python -m train.macro_rl --seeds 16 --points 6        # -> runs/macro_rl/<stamp>/model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval import monitor, priority
from eval.block_branch import MAPS, Watch, _match, _step

REPO_ROOT = Path(__file__).resolve().parent.parent
OPPONENTS = ("dummy", "v2", "replayed")
VERSION = "v018"
FROM, UNTIL = 2 * 60 * 24, 9 * 60 * 24
HORIZON = 180 * 24
GAIN_CLIP = 600.0                    # net value: a pair's weight grows with its gain up to this
KL_WEIGHT = 1.0


def base_game(args: tuple) -> list[dict]:
    """The game as Gary plays it: its macro decisions with their alternatives, and the score's windows."""
    opponent, mp, seed, points, version = args
    rng = random.Random(hash((opponent, mp, seed)) & 0xFFFFFFFF)
    m = _match(opponent, mp, seed, version)
    m.gary.macro.log = True
    w = Watch(m)
    w.look()
    end = UNTIL + HORIZON
    while m.game.frame < end and _step(m, w):
        pass
    final = m.game.frame
    model = m.gary.macro.model
    found = [d for d in m.gary.macro.draws if FROM <= d["frame"] <= UNTIL and d.get("k") is not None
             and len(d["cands"]) >= 2 and d["frame"] + HORIZON <= final]
    rng.shuffle(found)
    out = []
    for d in found[:points]:
        p, _ = model.probs(d["x"][None])
        others = [c for c in d["cands"] if c != d["k"]]
        q = np.array([p[0, c] for c in others], np.float64)
        alt = others[rng.choices(range(len(others)), weights=(q / q.sum()).tolist())[0]]
        out.append({"opponent": opponent, "map": mp, "seed": seed, "version": version, "frame": d["frame"],
                    "x": d["x"].tolist(), "cands": d["cands"], "k": d["k"], "alt": int(alt),
                    "base": w.window(d["frame"], d["frame"] + HORIZON)})
    m.game.close()
    return out


def branch(point: dict) -> dict:
    """The same game to the moment, the other decision there, then as before."""
    m = _match(point["opponent"], point["map"], point["seed"], point["version"])
    m.gary.macro.log = True                  # (as the base game: the same work, the same game)
    w = Watch(m)
    w.look()
    m.gary.macro.force = (point["frame"], point["alt"])
    end = point["frame"] + HORIZON
    while m.game.frame < end and _step(m, w):
        pass
    forced = m.gary.macro.force is None
    out = {**point, "branch": w.window(point["frame"], min(m.game.frame, end)), "forced": forced}
    m.game.close()
    return out


def train(rows: list[dict], model_path: str, out: Path, epochs: int = 20) -> None:
    from gary.policy.macro import MacroModel
    model = MacroModel.load(model_path)
    ref = MacroModel.load(model_path)
    games = sorted({(r["opponent"], r["map"], r["seed"]) for r in rows})
    held = set(games[::5])
    tr = [r for r in rows if (r["opponent"], r["map"], r["seed"]) not in held]
    te = [r for r in rows if (r["opponent"], r["map"], r["seed"]) in held]

    def tensors(rs):
        X = torch.tensor(np.array([r["x"] for r in rs], np.float32))
        a = torch.tensor([r["alt"] for r in rs])
        k = torch.tensor([r["k"] for r in rs])
        gain = torch.tensor([r["branch"]["net"] - r["base"]["net"] for r in rs], dtype=torch.float32)
        return X, a, k, gain

    def prefers(net, X, a, k):
        with torch.no_grad():
            logits, _ = net(X)
        return (logits.gather(1, a[:, None]) - logits.gather(1, k[:, None]))[:, 0]

    Xte, ate, kte, gte = tensors(te)
    decided = gte != 0
    before = prefers(ref.net, Xte, ate, kte)
    net = model.net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    X, a, k, gain = tensors(tr)
    sign = torch.sign(gain)
    weight = gain.abs().clamp(max=GAIN_CLIP) / GAIN_CLIP
    with torch.no_grad():
        ref_logp = F.log_softmax(ref.net(X)[0], -1)
    for _ in range(epochs):
        for i in torch.randperm(len(X)).split(256):
            logits, _ = net(X[i])
            diff = logits.gather(1, a[i, None])[:, 0] - logits.gather(1, k[i, None])[:, 0]
            pair = (F.softplus(-sign[i] * diff) * weight[i]).sum() / max(1e-6, float(weight[i].sum()))
            kl = F.kl_div(F.log_softmax(logits, -1), ref_logp[i], log_target=True, reduction="batchmean")
            loss = pair + KL_WEIGHT * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    after = prefers(net, Xte, ate, kte)
    right = lambda d: float(((d > 0) == (gte > 0))[decided].float().mean()) if decided.any() else float("nan")
    print(f"pairs: {len(tr)} training, {len(te)} held out ({int(decided.sum())} with a difference); held out, "
          f"prefers the better decision: before {right(before):.1%}, after {right(after):.1%} (chance 50%)")
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.pt")
    caps = Path(model_path).with_name("caps.json")
    if caps.exists():
        shutil.copy(caps, out / "caps.json")
    print(f"-> {out / 'model.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--points", type=int, default=6, help="decision moments branched per game")
    ap.add_argument("--version", default=VERSION)
    ap.add_argument("--data", help="pairs played before (JSON lines): train on them, adding none")
    ap.add_argument("--parallel", type=int, default=14)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    run = REPO_ROOT / "runs" / "macro_rl" / time.strftime("%Y%m%d_%H%M%S")
    if args.data:
        rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    else:
        jobs = [(o, mp, s, args.points, args.version) for o in OPPONENTS for mp in MAPS for s in range(1, args.seeds + 1)]
        with ProcessPoolExecutor(args.parallel) as pool:
            points = [p for ps in monitor.track(pool.map(base_game, jobs), len(jobs), "games to find moments") for p in ps]
            print(f"{len(points)} decision moments in {len(jobs)} games", flush=True)
            rows = [r for r in monitor.track(pool.map(branch, points), len(points), "branches") if r["forced"]]
        run.mkdir(parents=True, exist_ok=True)
        with open(run / "pairs.jsonl", "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    gains = np.array([r["branch"]["net"] - r["base"]["net"] for r in rows])
    print(f"{len(rows)} pairs: the other decision better {np.mean(gains > 0):.0%}, worse {np.mean(gains < 0):.0%}, "
          f"mean {gains.mean():+.0f}")
    from gary.agent.manifests import model_for
    train(rows, str(model_for("macro", "TvZ", "T")), run)


if __name__ == "__main__":
    main()
