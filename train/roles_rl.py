"""Reinforcement learning for the role model (gary/policy/roles.py; docs/CONTROL.md step 5).

Starts from a role model of decisions (default: the one imitated from attended pro units, told the
player's order rate, train/roles.py --attended --budget) and improves it on real pro positions: the
state-selected scenarios from training games (eval/scenarios.py --suite kinds --split train: attacks,
mixed fights, outnumbered units, harassment; --suite early: the home defenses), Gary playing with the
model (--manifest, default gt1+att+refresh+incontact: the role model deciding before contact and in it,
its moves kept current, told its own order rate from its hands, gary/policy/roles.py order_budget). Each
round Gary plays a batch of them several times, drawing each unit's role from the model. Every turn is judged by what happened in
the HORIZON_S seconds after it (eval/score.py's events: value killed and lost, damage...; each
command Gary carried out costs a little), against the other plays of the same position over the
same stretch (group-relative advantages). Each unit's role is an action of its own, sharing its
turn's advantage; a PPO-style clipped update makes better ones likelier, and a KL penalty keeps
the model near where it started. Every few rounds the model plays the validation scenarios (a
tenth of the training games', never trained on) without drawing (its likeliest roles), which
picks the best checkpoint; the real test is afterwards, on the held-out games' scenarios:

    python -m train.roles_rl --pipe 10          # first: do the weights reach the hands? (pipe())
    python -m train.roles_rl --rounds 60
    python -m eval.scenarios --suite kinds --run 79 --version gt1+att+refresh+incontact --controllers gary \
        --draws 8 --fight-model runs/roles_rl/<run>/model.pt

Its gates and stop rule, written before the first run: docs/PROGRESS.md, "The roles RL".
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

from eval import monitor, priority
from gary.policy.roles import RoleModel, tensors
from train.fight_cmd_rl import decisions, real_splits

REPO_ROOT = Path(__file__).resolve().parent.parent
HORIZON_S = 10


def play_real(args: tuple) -> dict:
    from eval import scenarios as S
    sc, path, draw, manifest, seconds = args
    r = S.run_one(sc, "gary", seconds, manifest, 1, fight_path=path, sample_seed=draw)
    r["drill"] = f"real_{S.scenario_id(sc)}_{sc['frame']}"
    return r


def play_greedy(args: tuple) -> dict:
    from eval import scenarios as S
    sc, path, hands, manifest, seconds = args
    return S.run_one(sc, "gary", seconds, manifest, 1, fight_path=path, hands_seed=1 + hands)


def batch(decs: list, device: str):
    n = max(len(d["rows"]) for d, _ in decs)
    B = len(decs)
    rows = np.zeros((B, n, 11), np.int16)
    extra = np.zeros((B, n, max(d["extra"].shape[1] for d, _ in decs)), np.float32)   # (EXTRA, +1: a budget model's)
    pad = np.ones((B, n), bool)
    role = np.full((B, n), -1, np.int64)
    times, adv = np.zeros(B, np.float32), np.zeros(B, np.float32)
    for b, (d, a) in enumerate(decs):
        m = min(len(d["rows"]), n)
        rows[b, :m], pad[b, :m] = d["rows"][:m], False
        extra[b, :m, :d["extra"].shape[1]] = d["extra"][:m]
        for j, y in zip(d["units"], d["roles"]):
            if j < m:
                role[b, j] = y
        times[b], adv[b] = d["time"], a
    t = lambda x: torch.as_tensor(x, device=device)
    return tensors(rows, extra, pad, times, device), t(role), t(adv)


def unit_log_probs(net, inputs, role, three=None):
    """Per unit: the log-probability of its role (where one was drawn), and its role distribution.
    three (override mode, gary/agent/roles.py RoleOverride): (keep, join, hold) indices: the
    distribution is then the rules (every decision but join and hold), join, hold."""
    lp_all = F.log_softmax(net(*inputs), -1)
    if three is not None:
        keep, join, hold = three
        others = [k for k in range(lp_all.shape[-1]) if k not in (join, hold)]
        lp_all = torch.stack([torch.logsumexp(lp_all[..., others], -1), lp_all[..., join], lp_all[..., hold]], -1)
        lookup = torch.full((max(three) + 1,), -1, dtype=torch.long, device=role.device)
        lookup[keep], lookup[join], lookup[hold] = 0, 1, 2
        role = torch.where(role >= 0, lookup[role.clamp(min=0, max=max(three))], role)
    mask = role >= 0
    lp = lp_all.gather(-1, role.clamp(min=0)[..., None])[..., 0]
    return lp, lp_all, mask


def update(net, ref, opt, decs: list, args, device: str) -> dict:
    net.eval()                                   # no dropout: the same policy that played
    random.shuffle(decs)
    parts = [batch(decs[i:i + args.minibatch], device) for i in range(0, len(decs), args.minibatch)]
    with torch.no_grad():
        old = [unit_log_probs(net, p[0], p[1], args.three)[0] for p in parts]
        ref_lp = [unit_log_probs(ref, p[0], p[1], args.three)[1] for p in parts]
    stats = {"pg": 0.0, "kl": 0.0, "clipped": 0.0, "n": 0}
    for _ in range(args.epochs):
        for k in random.sample(range(len(parts)), len(parts)):
            inputs, role, adv = parts[k]
            lp, lp_all, mask = unit_log_probs(net, inputs, role, args.three)
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
    """The validation scenarios' mean net, each played draws times (Gary's hands seeded apart:
    one play of a fight is a coin flip)."""
    rows = [r for r in pool.map(play_greedy, [(sc, path, h, manifest, seconds) for sc in val for h in range(draws)],
                                chunksize=draws) if "error" not in r]
    return float(np.mean([r["net"] for r in rows]))


def moved(pool, new: str, args, device: str) -> None:
    """The first gate (docs/PROGRESS.md, "The roles RL"): did the policy move off its start? The held-out suite
    played once a scenario with the new model drawing; on the states it saw, the mean KL from the start's role
    odds to the new model's per unit, and each decision's share under both."""
    from eval import scenarios as S
    test = [json.loads(line) for line in S.scenarios_path("test", args.suite).read_text(encoding="utf-8").splitlines()]
    rows = list(pool.map(play_real, [(sc, new, 1, args.manifest, args.seconds) for sc in test]))
    decs = [(d, 0.0) for r in rows if "error" not in r for d in r.get("log", []) if d.get("units")]
    start = RoleModel.load(new, "cpu").config.get("rl", {}).get("init", args.init_path)   # (the run's own start)
    a, b = RoleModel.load(new, device).net.eval(), RoleModel.load(start, device).net.eval()
    kls, pa, pb = [], 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(decs), args.minibatch):
            inputs, role, _ = batch(decs[i:i + args.minibatch], device)
            la, lb = F.log_softmax(a(*inputs), -1), F.log_softmax(b(*inputs), -1)
            mask = role >= 0
            kls.append((la.exp() * (la - lb)).sum(-1)[mask].cpu())
            pa = pa + la.exp()[mask].sum(0).cpu()
            pb = pb + lb.exp()[mask].sum(0).cpu()
    kl = torch.cat(kls)
    names = RoleModel.load(new, "cpu").config["roles"]
    print(f"{len(decs)} turns, {len(kl)} unit decisions on the held-out {args.suite} suite: KL(new || start) mean "
          f"{kl.mean():.4f}, median {kl.median():.4f}, 90th pct {kl.quantile(0.9):.4f}")
    pa, pb = pa / pa.sum(), pb / pb.sum()
    print("decision shares, start -> new: " + ", ".join(f"{n} {y:.1%} -> {x:.1%}" for n, x, y in zip(names, pa, pb)
                                                        if max(x, y) >= 0.005))


MOVE_ROLES = ("retreat", "pull_back", "reposition")


def pipe(pool, args, device: str) -> None:
    """Before a run: is the pipe from the role model's weights to Gary's hands open? On args.pipe held-out scenarios
    (spread over the kinds), the start model against itself pushed toward the move roles (args.bias added to their
    logits), with the manifest's skills (refresh on): if commands a minute and the units reached don't move, what RL
    changes in the weights can't reach the fight either."""
    from eval import scenarios as S
    test = [json.loads(line) for line in S.scenarios_path("test", args.suite).read_text(encoding="utf-8").splitlines()]
    by: dict[str, list] = {}
    for sc in test:
        by.setdefault(sc.get("kind"), []).append(sc)
    picked = [sc for group in zip(*by.values()) for sc in group][:args.pipe]
    model = RoleModel.load(args.init_path, device)
    names = model.config["roles"]
    with torch.no_grad():
        for r in MOVE_ROLES:
            model.net.role.bias[names.index(r)] += args.bias
    pushed = REPO_ROOT / "runs" / "roles_rl" / f"pipe_bias{args.bias:g}.pt"
    pushed.parent.mkdir(parents=True, exist_ok=True)
    model.save(pushed)
    moves = [names.index(r) for r in MOVE_ROLES]
    for label, path in (("start", args.init_path), (f"moves +{args.bias:g}", str(pushed))):
        rows = [r for r in pool.map(play_real, [(sc, path, k, args.manifest, args.seconds) for sc in picked
                                                for k in range(args.val_draws)], chunksize=args.val_draws)
                if "error" not in r]
        drawn = [k for r in rows for d in r.get("log", []) for k in d["roles"]]
        mean = lambda key: float(np.mean([r[key] for r in rows if key in r]))
        print(f"{label:12s} {len(rows)} plays: commands a minute {mean('commands_min'):.1f}, units reached "
              f"{mean('reached'):.0%}, fight orders {mean('fight_orders'):.1f}, move decisions "
              f"{np.isin(drawn, moves).mean() if drawn else 0:.1%} of {len(drawn)}, net {mean('net'):.0f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", help="role model of decisions to start from (default: the newest in runs/roles)")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--manifest", default="gt1+att+refresh+incontact", help="the Gary that plays (its roles: the model trained)")
    ap.add_argument("--suite", default="kinds", choices=["kinds", "early"], help="eval/scenarios.py's scenario suite")
    ap.add_argument("--override", action="store_true",
                    help="the model only overrides the rules (join, hold, else the rules): manifest v017override")
    ap.add_argument("--seconds", type=float, default=45, help="how long each scenario plays (as eval/scenarios.py scores)")
    ap.add_argument("--real", type=int, default=32, help="training scenarios per round")
    ap.add_argument("--draws", type=int, default=8, help="plays of each per round")
    ap.add_argument("--test-every", type=int, default=10)
    ap.add_argument("--val-draws", type=int, default=4, help="plays of each validation scenario")
    ap.add_argument("--horizon", type=float, default=HORIZON_S)
    ap.add_argument("--command-cost", type=float, default=0.0,
                    help="value a carried-out turn costs (0: the model is told its order rate; the hands' limit is the cost)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl", type=float, default=0.05, help="weight of the KL penalty to the starting model")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 6) * 2 // 3))
    ap.add_argument("--moved", help="a trained model: the first gate only (moved off its start? see moved())")
    ap.add_argument("--pipe", type=int, help="before a run: this many held-out scenarios, the start model against it "
                                             "pushed toward move roles (see pipe())")
    ap.add_argument("--bias", type=float, default=2.0, help="--pipe: added to the move roles' logits")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    if args.override and args.manifest == "v019roleml":
        args.manifest = "v017override"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from gary.policy.roles import latest_trained_on
    init = Path(args.init or latest_trained_on("response_T_att.npz"))
    args.init_path = str(init)
    if args.pipe:
        with ProcessPoolExecutor(args.parallel) as pool:
            pipe(pool, args, device)
        return
    if args.moved:
        with ProcessPoolExecutor(args.parallel) as pool:
            moved(pool, args.moved, args, device)
        return
    model = RoleModel.load(init, device)
    if "keep" not in model.config.get("roles", []):
        raise SystemExit(f"{init} isn't a model of decisions (no 'keep'): train one with --decisions or distill the rules")
    names = model.config["roles"]
    args.three = tuple(names.index(r) for r in ("keep", "join", "hold")) if args.override else None
    ref = RoleModel.load(init, device).net.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    net = model.net
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    out = REPO_ROOT / "runs" / "roles_rl" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    model.config = {**model.config, "rl": {"init": str(init), **vars(args)}}
    train_real, val_real = real_splits(args.suite)
    print(f"from {init}; {len(train_real)} scenarios to train on ({args.real} x {args.draws} plays per round), "
          f"{len(val_real)} for validation", flush=True)
    history, best = [], None
    rng = random.Random(0)
    with ProcessPoolExecutor(args.parallel) as pool:
        for rnd in monitor.track(range(args.rounds + 1), args.rounds + 1, "rounds"):
            path = str(out / f"round{rnd}.pt")
            model.save(path)
            if rnd % args.test_every == 0 or rnd == args.rounds:
                score = validate(pool, val_real, path, args.val_draws, args.manifest, args.seconds)
                print(f"round {rnd}: validation net {score:.0f}", flush=True)
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
    print(f"best validation net {best[0]:.0f} at round {best[1]} -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
