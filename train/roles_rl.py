"""Reinforcement learning for the role model (gary/policy/roles.py; docs/CONTROL.md step 5).

Starts from a role model of decisions, by default the hand rules distilled into it
(ingest/role_distill.py, train/roles.py --no-weights: as good as the rules in play), and improves
it on real pro positions: the home defenses from training games (eval/scenarios.py --split train),
Gary playing with learned roles (--manifest, default v017roleml; --seconds each, 120 by default:
eval/episodes.py). Each round Gary plays a batch of them
several times, drawing each unit's role from the model. Every turn is judged by what happened in
the HORIZON_S seconds after it (eval/score.py's events: value killed and lost, damage...; each
command Gary carried out costs a little), against the other plays of the same position over the
same stretch (group-relative advantages). Each unit's role is an action of its own, sharing its
turn's advantage; a PPO-style clipped update makes better ones likelier, and a KL penalty keeps
the model near where it started. Every few rounds the model plays the validation home defenses (a
tenth of the training games', never trained on) without drawing (its likeliest roles), which
picks the best checkpoint; the real test is afterwards, on the held-out games' home defenses:

    python -m train.roles_rl --rounds 60
    python -m eval.scenarios --run 114 --version v09roleml --controllers gary --fight-model runs/roles_rl/<run>/model.pt
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

from gary.agent.manifests import latest_role_model
from gary.policy.roles import EXTRA, RoleModel, tensors
from train.fight_cmd_rl import decisions, real_splits

REPO_ROOT = Path(__file__).resolve().parent.parent
HORIZON_S = 10


def play_real(args: tuple) -> dict:
    from eval import scenarios as S
    sc, path, draw, manifest, seconds = args
    r = S.run_one(sc, "gary", seconds, manifest, 1, fight_path=path, sample_seed=draw)
    r["drill"] = "real_" + sc["sha1"][:12]
    return r


def play_greedy(args: tuple) -> dict:
    from eval import scenarios as S
    sc, path, hands, manifest, seconds = args
    return S.run_one(sc, "gary", seconds, manifest, 1, fight_path=path, hands_seed=1 + hands)


def batch(decs: list, device: str):
    n = max(len(d["rows"]) for d, _ in decs)
    B = len(decs)
    rows = np.zeros((B, n, 11), np.int16)
    extra = np.zeros((B, n, EXTRA), np.float32)
    pad = np.ones((B, n), bool)
    role = np.full((B, n), -1, np.int64)
    times, adv = np.zeros(B, np.float32), np.zeros(B, np.float32)
    for b, (d, a) in enumerate(decs):
        m = min(len(d["rows"]), n)
        rows[b, :m], extra[b, :m], pad[b, :m] = d["rows"][:m], d["extra"][:m], False
        for j, y in zip(d["units"], d["roles"]):
            if j < m:
                role[b, j] = y
        times[b], adv[b] = d["time"], a
    t = lambda x: torch.as_tensor(x, device=device)
    return tensors(rows, extra, pad, times, device), t(role), t(adv)


def unit_log_probs(net, inputs, role):
    """Per unit: the log-probability of its role (where one was drawn), and its role distribution."""
    lp_all = F.log_softmax(net(*inputs), -1)
    mask = role >= 0
    lp = lp_all.gather(-1, role.clamp(min=0)[..., None])[..., 0]
    return lp, lp_all, mask


def update(net, ref, opt, decs: list, args, device: str) -> dict:
    net.eval()                                   # no dropout: the same policy that played
    random.shuffle(decs)
    parts = [batch(decs[i:i + args.minibatch], device) for i in range(0, len(decs), args.minibatch)]
    with torch.no_grad():
        old = [unit_log_probs(net, p[0], p[1])[0] for p in parts]
        ref_lp = [unit_log_probs(ref, p[0], p[1])[1] for p in parts]
    stats = {"pg": 0.0, "kl": 0.0, "clipped": 0.0, "n": 0}
    for _ in range(args.epochs):
        for k in random.sample(range(len(parts)), len(parts)):
            inputs, role, adv = parts[k]
            lp, lp_all, mask = unit_log_probs(net, inputs, role)
            ratio = torch.exp((lp - old[k]).clamp(-20, 20))[mask]
            a = adv[:, None].expand_as(lp)[mask]
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            kl = (lp_all.exp() * (lp_all - ref_lp[k])).sum(-1)[mask].mean()
            loss = pg + args.kl * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["pg"] += pg.item(); stats["kl"] += kl.item()
            stats["clipped"] += ((ratio - 1).abs() > args.clip).float().mean().item(); stats["n"] += 1
    return {k: round(v / max(1, stats["n"]), 4) for k, v in stats.items() if k != "n"}


def validate(pool, val: list[dict], path: str, draws: int, manifest: str, seconds: float) -> float:
    """The validation home defenses' mean net, each played draws times (Gary's hands seeded apart:
    one play of a fight is a coin flip)."""
    rows = [r for r in pool.map(play_greedy, [(sc, path, h, manifest, seconds) for sc in val for h in range(draws)],
                                chunksize=draws) if "error" not in r]
    return float(np.mean([r["net"] for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", help="role model of decisions to start from (default: the newest in runs/roles)")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--manifest", default="v017roleml", help="the Gary that plays (its roles: the model trained)")
    ap.add_argument("--seconds", type=float, default=120,
                    help="how long each home defense plays (eval/episodes.py: long enough for new units to join)")
    ap.add_argument("--real", type=int, default=32, help="training home defenses per round")
    ap.add_argument("--draws", type=int, default=8, help="plays of each per round")
    ap.add_argument("--test-every", type=int, default=10)
    ap.add_argument("--val-draws", type=int, default=4, help="plays of each validation home defense")
    ap.add_argument("--horizon", type=float, default=HORIZON_S)
    ap.add_argument("--command-cost", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl", type=float, default=0.05, help="weight of the KL penalty to the starting model")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 6) * 2 // 3))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = Path(args.init or latest_role_model())
    model = RoleModel.load(init, device)
    if "keep" not in model.config.get("roles", []):
        raise SystemExit(f"{init} isn't a model of decisions (no 'keep'): train one with --decisions or distill the rules")
    ref = RoleModel.load(init, device).net.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    net = model.net
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    out = REPO_ROOT / "runs" / "roles_rl" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    model.config = {**model.config, "rl": {"init": str(init), **vars(args)}}
    train_real, val_real = real_splits()
    print(f"from {init}; {len(train_real)} home defenses to train on ({args.real} x {args.draws} plays per round), "
          f"{len(val_real)} for validation", flush=True)
    history, best = [], None
    rng = random.Random(0)
    with ProcessPoolExecutor(args.parallel) as pool:
        for rnd in range(args.rounds + 1):
            path = str(out / f"round{rnd}.pt")
            model.save(path)
            if rnd % args.test_every == 0 or rnd == args.rounds:
                score = validate(pool, val_real, path, args.val_draws, args.manifest, args.seconds)
                print(f"round {rnd}: validation home defenses net {score:.0f}", flush=True)
                history.append({"round": rnd, "validation": round(score, 1)})
                if best is None or score > best[0]:
                    best = (score, rnd)
                    model.save(out / "model.pt")
            if rnd == args.rounds:
                break
            t0 = time.time()
            scs = rng.sample(train_real, min(args.real, len(train_real)))
            rows = list(pool.map(play_real, [(sc, path, k, args.manifest, args.seconds) for sc in scs
                                             for k in range(args.draws)], chunksize=args.draws))
            decs = [(d, a) for d, a in decisions(rows, args.horizon, args.command_cost) if d.get("units")]
            nets = [r["net"] for r in rows if "error" not in r]
            stats = update(net, ref, opt, decs, args, device) if decs else {}
            errors = sum(1 for r in rows if "error" in r)
            print(f"round {rnd + 1}: train net {np.mean(nets):.0f}, {len(decs)} turns, {stats}, {errors} errors, "
                  f"{time.time() - t0:.0f} s", flush=True)
            history.append({"round": rnd + 1, "train_net": round(float(np.mean(nets)), 1), **stats})
            for old in out.glob("round*.pt"):
                if old.name != f"round{rnd}.pt":
                    old.unlink()
            (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"best validation home defenses net {best[0]:.0f} at round {best[1]} -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
