"""RL from the league's games (eval/league.py; docs/SELF_PLAY.md "The loop"): each decision a learned
layer logged (gary/decision_log.py: its inputs, the decision drawn, its log-probability) is scored by
what followed, then one policy-gradient update of that layer with a KL leash to the model it started
from (the imitation model: ARCHITECTURE's lambda_KL, as AlphaStar).

  roles_z   Gary-Z's fight roles (gary/bots/zerg_roles.py). A snapshot's outcome: the fight's net
            value over the next WINDOW (the enemy's units within FIGHT of the fight's middle lost, less
            its own; minerals + gas), the fight episode the role decisions were made in
  army_z    Gary-Z's army layer (gary_z.py _army_goal), and army_t the Terran's (gary/agent/army.py with
  army_t    draw): the side's share of all army supply WINDOW later, less now (what the army model's
            fight estimate learns), whatever the group did (held, stayed, went)

Outcomes come from the games' replays played again in OpenBW (the full state: this scores, it isn't an
input). Advantages are the outcomes standardized over the round; the update clips the probability
ratio against the logged decision's (PPO) and adds KL(start || new) on the same inputs. One round's
games are one batch: the next round, played with the updated model, is its judge (results, fight
outcomes, the classifier's gap).

    python -m train.league_rl --round r2 --layer roles_z
    python -m train.league_rl --round r2 --layer army_z
    python -m train.league_rl --round r2 --layer army_t
    (eval/league_loop.py calls train_pairs: the Zerg role model from paired fight episodes, the army
    layers frozen at imitation until the critic predicts more than who is ahead)
Output: the updated model (roles_z: runs/roles_rl/Z_<time>/; army_z: runs/army/TvZ_Z_<time>_rl/, the
newest Zerg army model Gary-Z loads; army_t: runs/army_rl/TvZ_T_<time>/, played with the option
army_rl) and report.json beside it.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval import monitor, priority

REPO = Path(__file__).resolve().parents[1]
WINDOW = 45 * 24                # frames: what followed a decision
FIGHT = 12 * 32                 # px: the fight's units, around its middle
CC, HALLS = 106, {131, 132, 133}


def _value(t: int) -> float:
    from gary import terran as T
    from gary import zerg as Z
    m, g = T.COST.get(t) or Z.COST.get(t) or (50, 0)
    return (m + g) / (2 if t in Z.PAIRS else 1)


def _snap(o: dict) -> dict:
    return {u["tag"]: (u["owner"], u["type"], u["x"], u["y"]) for u in o["units"] if u["owner"] != 11}


def outcomes(job: tuple) -> dict:
    """{frame: outcome} for one game's decisions of one layer."""
    rep, layer, frames = job
    from gary.env import Game
    from gary.policy.army import is_building, is_worker, supply_x2
    want = sorted({f for f in frames} | {f + WINDOW for f in frames})
    snaps: dict[int, dict] = {}
    g = Game.scenario(rep)
    try:
        slot = None
        for f in want:
            if f > g.frame:
                try:
                    g.step(f - g.frame)
                except Exception:
                    break
            o = g.observe()
            if slot is None:
                kinds = {CC} if layer.endswith("_t") else HALLS
                slot = next((u["owner"] for u in o["units"] if u["type"] in kinds), None)
            snaps[f] = _snap(o)
    finally:
        g.close()
    fighter = lambda t: supply_x2(t) > 0 and not is_worker(t) and not is_building(t)
    out = {}
    for f in frames:
        a, b = snaps.get(f), snaps.get(f + WINDOW)
        if a is None or b is None or slot is None:
            continue
        if layer.startswith("army"):
            share = lambda s: (lambda own, all_: own / all_ if all_ else 0.5)(
                sum(supply_x2(t) for o_, t, _, _ in s.values() if o_ == slot and fighter(t)),
                sum(supply_x2(t) for o_, t, _, _ in s.values() if fighter(t)))
            out[f] = share(b) - share(a)
        else:
            mine = [(x, y) for o_, t, x, y in a.values() if o_ == slot and fighter(t)]
            foes = [(x, y) for o_, t, x, y in a.values() if o_ != slot and fighter(t)]
            if not mine or not foes:
                continue
            m, e = np.array(mine, np.float32), np.array(foes, np.float32)
            d = np.sqrt(((m[:, None] - e[None]) ** 2).sum(-1)).min(1)
            mid = m[d < FIGHT].mean(0) if (d < FIGHT).any() else m[d.argmin()]
            near = {tag: v for tag, v in a.items() if (v[2] - mid[0]) ** 2 + (v[3] - mid[1]) ** 2 < FIGHT ** 2}
            lost = lambda own: sum(_value(v[1]) for tag, v in near.items() if (v[0] == slot) == own and tag not in b)
            out[f] = (lost(False) - lost(True)) / 100.0
    return out


def load_round(rnd: str, layer: str, workers: int):
    """The round's logged decisions of the layer, each with its outcome (and its game)."""
    files = sorted(glob.glob(str(REPO / "runs" / "league" / rnd / "*" / "*.decisions.npz")))
    logs = []
    for path in files:
        d = np.load(path)
        if f"{layer}/frame" in d.files:
            logs.append((path, {k.split("/", 1)[1]: d[k] for k in d.files if k.startswith(layer + "/")}))
    jobs = [(p.replace(".decisions.npz", ".rep"), layer, sorted(set(int(f) for f in log["frame"]))) for p, log in logs]
    with ProcessPoolExecutor(workers) as pool:
        outs = list(monitor.track(pool.map(outcomes, jobs), len(jobs), "games scored"))
    keep = {k: [] for k in logs[0][1]} if logs else {}
    reward, game = [], []
    for gi, ((_, log), out) in enumerate(zip(logs, outs)):
        for i, f in enumerate(log["frame"]):
            if int(f) in out:
                for k in keep:
                    keep[k].append(log[k][i])
                reward.append(out[int(f)])
                game.append(gi)
    return {k: np.stack(v) for k, v in keep.items()}, np.array(reward, np.float32), np.array(game), len(logs)


def update_roles(data, adv, args, device):
    from gary.bots.zerg_roles import latest_zerg_model
    from gary.policy.roles import RoleModel, tensors
    start = latest_zerg_model()
    model = RoleModel.load(start, device)
    net, ref = model.net.eval(), copy.deepcopy(model.net).eval()   # (no dropout: the logged odds had none)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    n = len(adv)
    stats = []
    for epoch in range(args.epochs):
        order = np.random.default_rng(epoch).permutation(n)
        for s in range(0, n, args.batch):
            idx = order[s:s + args.batch]
            count = data["count"][idx].astype(int)
            k = int(count.max())
            pad = np.arange(k)[None, :] >= count[:, None]
            inputs = tensors(data["rows"][idx][:, :k].astype(np.int16), data["extra"][idx][:, :k].astype(np.float32), pad,
                             data["time"][idx].astype(np.float32), device)
            drawn = torch.as_tensor(data["drawn"][idx][:, :k].astype(np.int64), device=device)
            old = torch.as_tensor(data["logp"][idx][:, :k], device=device)
            mask = drawn >= 0
            logp = F.log_softmax(net(*inputs), -1)
            with torch.no_grad():
                logp0 = F.log_softmax(ref(*inputs), -1)
            new = logp.gather(-1, drawn.clamp(min=0)[..., None])[..., 0]
            a = torch.as_tensor(adv[idx], device=device)[:, None].expand_as(new)
            ratio = torch.exp(new - old)
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a)[mask].mean()
            kl = (logp0.exp() * (logp0 - logp)).sum(-1)[mask].mean()
            loss = pg + args.kl * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            stats.append((pg.item(), kl.item()))
    return model, start, stats


def update_army(data, adv, args, device, race):
    from gary.policy.army import ArmyModel
    found = sorted((REPO / "runs" / "army").glob(f"TvZ_{race.upper()}_*/model.pt"))
    if race == "t":
        found = sorted((REPO / "runs" / "army_rl").glob("TvZ_T_*/model.pt")) or found
    start = found[-1]
    model = ArmyModel.load(start, device)
    net, ref = model.net.eval(), copy.deepcopy(model.net).eval()   # (no dropout: the logged odds had none)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    n = len(adv)
    stats = []
    for epoch in range(args.epochs):
        order = np.random.default_rng(epoch).permutation(n)
        for s in range(0, n, args.batch):
            idx = order[s:s + args.batch]
            grid = torch.as_tensor(data["grid"][idx].astype(np.float32), device=device).view(len(idx), -1, 16, 16)
            glob_ = torch.as_tensor(data["glob"][idx].astype(np.float32), device=device)
            cell = torch.as_tensor(data["cell"][idx].astype(np.int64), device=device)
            attack = torch.as_tensor(data["attack"][idx].astype(np.int64), device=device)
            old = torch.as_tensor(data["logp"][idx].astype(np.float32), device=device)
            where, kind, _ = net(grid, glob_)
            with torch.no_grad():
                where0, kind0, _ = ref(grid, glob_)
            lw, lk = F.log_softmax(where, -1), F.log_softmax(kind, -1)
            new = lw.gather(1, cell[:, None])[:, 0] + lk.gather(1, attack[:, None])[:, 0]
            a = torch.as_tensor(adv[idx], device=device)
            ratio = torch.exp(new - old)
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            l0w, l0k = F.log_softmax(where0, -1), F.log_softmax(kind0, -1)
            kl = ((l0w.exp() * (l0w - lw)).sum(-1) + (l0k.exp() * (l0k - lk)).sum(-1)).mean()
            loss = pg + args.kl * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            stats.append((pg.item(), kl.item()))
    return model, start, stats


def load_pairs(rounds: list[str]):
    """Training pairs (both arms the same model) from the rounds' games: each arm's decisions with its
    advantage, the arm's outcome less the pair's mean (the position's luck cancelled); and every pair's
    arms (for the gate and the diagnostics)."""
    files = [f for r in rounds for f in sorted(glob.glob(str(REPO / "runs" / "league" / r / "*" / "*.pairs.npz")))]
    dec = {k: [] for k in ("rows", "extra", "count", "time", "drawn", "logp")}
    adv = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        if "dec/pair" not in d.files:
            continue
        reward, model = d["reward"], d["model"]
        same = model[:, 0] == model[:, 1]
        a = reward - reward.mean(1, keepdims=True)
        for i in range(len(d["dec/pair"])):
            pi, arm = int(d["dec/pair"][i]), int(d["dec/arm"][i])
            if not same[pi] or reward[pi, 0] == reward[pi, 1]:
                continue                                  # (A/B pairs judge; equal outcomes teach nothing)
            for k in dec:
                dec[k].append(d[f"dec/{k}"][i])
            adv.append(a[pi, arm])
    return {k: np.stack(v) for k, v in dec.items()} if adv else {}, np.array(adv, np.float32), len(files)


def train_pairs(rounds: list[str], start: Path, imitation: Path, out: Path, lr: float = 3e-5, epochs: int = 4,
                batch: int = 256, clip: float = 0.2, kl: float = 0.1) -> dict:
    """One PPO update of the Zerg role model from the rounds' training pairs, KL-leashed to the imitation
    model (not the round's start: no drift compounding over rounds). The new model and report.json in out."""
    from gary.policy.roles import RoleModel, tensors
    data, adv, n_files = load_pairs(rounds)
    if not len(adv):
        return {"decisions": 0}
    adv = (adv / (adv.std() + 1e-6)).astype(np.float32)   # (already centred per pair)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = RoleModel.load(start, device)
    net = model.net.eval()                                 # (no dropout: the logged odds had none)
    ref = RoleModel.load(imitation, device).net.eval()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n, stats = len(adv), []

    def kl_to_ref(idx):
        count = data["count"][idx].astype(int)
        k = int(count.max())
        pad = np.arange(k)[None, :] >= count[:, None]
        inputs = tensors(data["rows"][idx][:, :k].astype(np.int16), data["extra"][idx][:, :k].astype(np.float32), pad,
                         data["time"][idx].astype(np.float32), device)
        return inputs, k

    for epoch in range(epochs):
        order = np.random.default_rng(epoch).permutation(n)
        for s0 in range(0, n, batch):
            idx = order[s0:s0 + batch]
            inputs, k = kl_to_ref(idx)
            drawn = torch.as_tensor(data["drawn"][idx][:, :k].astype(np.int64), device=device)
            old = torch.as_tensor(data["logp"][idx][:, :k].astype(np.float32), device=device)
            mask = drawn >= 0
            logp = F.log_softmax(net(*inputs), -1)
            with torch.no_grad():
                logp0 = F.log_softmax(ref(*inputs), -1)
            new = logp.gather(-1, drawn.clamp(min=0)[..., None])[..., 0]
            a = torch.as_tensor(adv[idx], device=device)[:, None].expand_as(new)
            ratio = torch.exp(new - old)
            pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)[mask].mean()
            klv = (logp0.exp() * (logp0 - logp)).sum(-1)[mask].mean()
            loss = pg + kl * klv
            opt.zero_grad()
            loss.backward()
            opt.step()
            stats.append((pg.item(), klv.item()))
    out.mkdir(parents=True, exist_ok=True)
    model.net.eval().cpu()
    model.config = {**model.config, "rl": {"rounds": rounds, "from": str(start), "imitation": str(imitation),
                                           "kl": kl, "lr": lr, "epochs": epochs, "decisions": int(n)}}
    model.save(out / "model.pt")
    report = {"rounds": rounds, "from": str(start), "imitation": str(imitation), "decisions": int(n), "files": n_files,
              "pg_first": stats[0][0], "pg_last": stats[-1][0], "kl_to_imitation": stats[-1][1]}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True)
    ap.add_argument("--layer", required=True, choices=("roles_z", "army_z", "army_t"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl", type=float, default=0.1, help="the KL leash's weight")
    ap.add_argument("--workers", type=int, default=4)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    torch.manual_seed(0)
    data, reward, game, n_games = load_round(args.round, args.layer, args.workers)
    if not len(reward):
        raise SystemExit(f"no {args.layer} decisions with outcomes in round {args.round}")
    adv = ((reward - reward.mean()) / (reward.std() + 1e-6)).astype(np.float32)
    print(f"{args.layer}: {len(reward)} decisions from {n_games} games; outcome mean {reward.mean():+.3f}, "
          f"sd {reward.std():.3f}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.layer == "roles_z":
        model, start, stats = update_roles(data, adv, args, device)
        out = REPO / "runs" / "roles_rl" / f"Z_{stamp}"
    else:
        race = args.layer[-1]
        model, start, stats = update_army(data, adv, args, device, race)
        out = REPO / "runs" / ("army" if race == "z" else "army_rl") / (f"TvZ_Z_{stamp}_rl" if race == "z" else f"TvZ_T_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    model.net.eval().cpu()
    model.config = {**model.config, "rl": {"round": args.round, "from": str(start), "kl": args.kl, "lr": args.lr,
                                           "epochs": args.epochs, "decisions": int(len(reward))}}
    model.save(out / "model.pt")
    report = {"layer": args.layer, "round": args.round, "from": str(start), "decisions": int(len(reward)),
              "games": n_games, "outcome_mean": float(reward.mean()), "outcome_sd": float(reward.std()),
              "first": {"pg": stats[0][0], "kl": stats[0][1]}, "last": {"pg": stats[-1][0], "kl": stats[-1][1]}}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"KL to the start: {stats[-1][1]:.4f} after {len(stats)} steps; saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
