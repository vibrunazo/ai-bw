"""Reinforcement learning for the fight model from pro scenarios (#2, #13).

Starts from the imitation-trained fight model (train/fight.py) and improves it on scenarios from
training games (eval/scenarios.py --split train): each round, Gary plays a batch of scenarios
several times, sampling its fight decisions from the model; draws that ended better than the
scenario's average make the decisions taken in them more likely, worse draws less likely
(group-relative advantages, PPO-style clipped update). Each decision is judged by what happened
in the 20 s after it (net value: zerg lost minus terran lost), against the other draws over the
same stretch, so a draw's early losses don't blame its later decisions. A KL penalty keeps the model near the imitation model, so it stays pro-like where the
scenarios say nothing. Every few rounds it's scored on the held-out scenarios (the test split).

Only decisions Gary carried out are learned from (a group ordered by human hands); the
executor's filters (units already doing it, one group per turn) are part of the environment.

    python -m train.fight_rl --rounds 30 --parallel 16
    python -m eval.scenarios --run 45 --controllers gary --samples 4 --fight-model runs/fight_rl/<run>/model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval import scenarios as S
from gary.agent.fight import DEST_NOISE
from gary.agent.manifests import latest_fight_model
from gary.policy.fight import ACTIONS, DEST_SCALE, FightModel, tensors

REPO_ROOT = Path(__file__).resolve().parent.parent
POINTER = [ACTIONS.index(a) for a in ("attack_unit", "gather", "own_unit")]
MOVES = [ACTIONS.index(a) for a in ("move", "attack_move")]
ADV_FLOOR = 50                  # value: draws closer than this to each other count as ties
HORIZON_S = 20                  # a decision is credited with what happens in the next 20 s


def rollouts(pool, scs: list[dict], draws: int, model_path: str, args, seed0: int) -> list[dict]:
    jobs = [(sc, "gary", args.seconds, "v04", args.style, None, False, model_path, seed0 + k)
            for sc in scs for k in range(draws)]
    return list(pool.map(S._job, jobs, chunksize=draws))   # a scenario's plays on one worker (reuses its game)


def net_value(r: dict) -> float | None:
    return None if "error" in r else r["Z_lost"] - r["T_lost"]


def to_go(deaths: list, t: float, horizon: float) -> float:
    """Net value of the deaths in (t, t + horizon] seconds."""
    return sum(v for f, v in deaths if t < f * 42 / 1000 <= t + horizon)


def decisions(rows: list[dict], horizon: float) -> list[tuple]:
    """(unit rows, time, unit, action, target, dest, advantage) for every carried-out decision.
    The advantage is reward-to-go: what this draw gained in the horizon after the decision,
    against the other draws of the same scenario over the same stretch of time."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        if net_value(r) is not None:
            by.setdefault(r["sha1"], []).append(r)
    out = []
    for rs in by.values():
        if len(rs) < 2:
            continue
        for k, r in enumerate(rs):
            for d in r.get("log", []):
                g = np.array([to_go(o["deaths"], d["time"], horizon) for o in rs], np.float32)
                a = (g[k] - g.mean()) / max(float(g.std()), ADV_FLOOR)
                if a == 0:
                    continue
                for u, act, tgt, dest in zip(d["unit"], d["act"], d["target"], d["dest"]):
                    out.append((d["rows"], d["time"], u, act, tgt, dest, float(a)))
    return out


def batch(dec: list[tuple], device: str):
    n = max(len(d[0]) for d in dec)
    units = np.zeros((len(dec), n, 11), np.int16)
    pad = np.ones((len(dec), n), bool)
    for b, d in enumerate(dec):
        units[b, :len(d[0])] = d[0]
        pad[b, :len(d[0])] = False
    t = lambda k, dt: torch.tensor([d[k] for d in dec], dtype=dt, device=device)
    return (tensors(units, pad, np.array([d[1] for d in dec], np.float32), device),
            t(2, torch.long), t(3, torch.long), t(4, torch.long), t(5, torch.float32) / DEST_SCALE, t(6, torch.float32))


def log_probs(net, inputs, unit, act, tgt, dest):
    """Log-probability of each decision (action, then its target or destination), and the action
    log-probabilities of the decided units (for the KL penalty)."""
    a_l, t_l, d_o = net(*inputs)
    b = torch.arange(len(unit), device=unit.device)
    a_lp = F.log_softmax(a_l[b, unit], -1)
    lp = a_lp[b, act]
    is_ptr = torch.isin(act, torch.tensor(POINTER, device=act.device))
    t_lp = F.log_softmax(t_l[b, unit], -1)[b, tgt.clamp(min=0)]
    lp = lp + torch.where(is_ptr, t_lp, torch.zeros_like(t_lp))
    is_move = torch.isin(act, torch.tensor(MOVES, device=act.device))
    sigma = DEST_NOISE / DEST_SCALE
    d_lp = -((dest - d_o[b, unit]) ** 2).sum(-1) / (2 * sigma ** 2)
    lp = lp + torch.where(is_move, d_lp, torch.zeros_like(d_lp))
    return lp, a_lp


def update(net, ref, opt, dec: list[tuple], args, device: str) -> dict:
    net.eval()                                   # no dropout: the same policy that played
    parts = [batch(dec[i:i + args.minibatch], device) for i in range(0, len(dec), args.minibatch)]
    with torch.no_grad():
        old = [log_probs(net, *p[:5])[0] for p in parts]
        ref_lp = [log_probs(ref, *p[:5])[1] for p in parts]
    stats = {"pg": 0.0, "kl": 0.0, "clipped": 0.0, "n": 0}
    for _ in range(args.epochs):
        order = list(range(len(parts)))
        random.shuffle(order)
        for k in order:
            inputs, unit, act, tgt, dest, adv = parts[k]
            lp, a_lp = log_probs(net, inputs, unit, act, tgt, dest)
            ratio = torch.exp(lp - old[k])
            pg = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            kl = (a_lp.exp() * (a_lp - ref_lp[k])).sum(-1).mean()
            loss = pg + args.kl * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["pg"] += pg.item(); stats["kl"] += kl.item()
            stats["clipped"] += ((ratio - 1).abs() > args.clip).float().mean().item(); stats["n"] += 1
    return {k: round(v / max(1, stats["n"]), 4) for k, v in stats.items() if k != "n"}


def test_score(pool, test: list[dict], path: str, args) -> float:
    rows = rollouts(pool, test, args.test_draws, path, args, 10_000)
    v = [net_value(r) for r in rows if net_value(r) is not None]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", help="fight model to start from (default: the latest imitation model)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--scenarios", type=int, default=32, help="training scenarios per round")
    ap.add_argument("--draws", type=int, default=8, help="plays of each scenario per round")
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--test-draws", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--horizon", type=float, default=HORIZON_S, help="credit a decision with the next this many seconds")
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl", type=float, default=0.05, help="weight of the KL penalty to the imitation model")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 6) * 2 // 3),
                    help="worker processes playing scenarios (default: 2/3 of the logical cores; more add nothing measurable)")
    ap.add_argument("--overfit", type=int, help="diagnostic: train and test on only this many training scenarios")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = Path(args.init or latest_fight_model())
    model = FightModel.load(init, device)
    ref = FightModel.load(init, device).net.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    net = model.net
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    train = [json.loads(line) for line in S.scenarios_path("train").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in S.scenarios_path("test").read_text(encoding="utf-8").splitlines()]
    if args.overfit:
        train = test = train[:args.overfit]
    out = REPO_ROOT / "runs" / "fight_rl" / time.strftime("TvZ_T_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    model.config = {**model.config, "rl": {"init": str(init), **vars(args)}}
    history = []
    rng = random.Random(0)
    print(f"{len(train)} training scenarios, {len(test)} test; from {init}", flush=True)
    with ProcessPoolExecutor(args.parallel) as pool:
        best = None
        for rnd in range(args.rounds + 1):
            path = str(out / f"round{rnd}.pt")
            model.save(path)
            if rnd % args.test_every == 0 or rnd == args.rounds:
                score = test_score(pool, test, path, args)
                history.append({"round": rnd, "test_net": round(score, 1)})
                print(f"round {rnd}: test net {score:.0f}", flush=True)
                if best is None or score > best[0]:
                    best = (score, rnd)
                    model.save(out / "model.pt")
            if rnd == args.rounds:
                break
            t0 = time.time()
            scs = rng.sample(train, min(args.scenarios, len(train)))
            rows = rollouts(pool, scs, args.draws, path, args, rnd * 1000)
            dec = decisions(rows, args.horizon)
            v = [net_value(r) for r in rows if net_value(r) is not None]
            stats = update(net, ref, opt, dec, args, device) if dec else {}
            errors = sum(1 for r in rows if "error" in r)
            print(f"round {rnd + 1}: train net {np.mean(v):.0f} over {len(v)} plays, {len(dec)} decisions, "
                  f"{stats}, {errors} errors, {time.time() - t0:.0f} s", flush=True)
            history.append({"round": rnd + 1, "train_net": round(float(np.mean(v)), 1), **stats})
            for old in out.glob("round*.pt"):
                if old.name != f"round{rnd}.pt":
                    old.unlink()
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"best test net {best[0]:.0f} at round {best[1]} -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
