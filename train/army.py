"""Train Gary's army model (gary/policy/army.py) on the army training set.

Behavior cloning: for each army order a pro gave, predict where the group went (one of 256 cells
of the mirrored 16x16 map grid) and whether it was a move or an attack. Games are split by replay
(the same 10% held out as the macro model), and scored on the held-out games:

  cell top-1 / top-5    the destination cell is the model's first (or among its first five) guesses
  within 1 cell         the first guess is the destination or one of its 8 neighbors
  far orders            the same, only for orders sending the group 3+ cells away (attacks,
                        retreats, defending another base), the decisions that matter most
  kind                  move vs attack accuracy
  fight                 the fight estimate: correlation with what happened (the change in the
                        player's share of all army supply over the next 45 s), and for attacks that
                        clearly went well or badly (|change| >= 5 points), how often it called it
Baselines: "stay" (the group's own cell) and "most common destination cell".

Memory and fight labels come from the macro training set (ingest/macro_dataset.py) of the same
game: what the player had seen each second, and both players' real armies.

Usage:
  python -m train.army --race T
Output: runs/army/<matchup>_<race>_<time>/model.pt and report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))
from inventory import data_root  # noqa: E402
from train.select import GameFilter  # noqa: E402

from gary.policy.army import (ARMY_SUPPLY, CELLS, CHANNELS, GRID, MEMORY_S, VALUE_HORIZON_S,  # noqa: E402
                              VALUE_SCALE, ArmyModel, ArmyNet, ArmySpec, encode_global)
from train.macro import is_test  # noqa: E402  (same held-out games as the macro model)

MAX_PER_PLAYER = 160          # samples per player-game (keeps memory in check, spreads over games)


def army_dir() -> Path:
    return data_root() / "interim" / "army" / "v1"


def player_games(matchup: str, race: str, keep=None) -> list[tuple[str, int, bool, str, int]]:
    """(army npz, player index, held out, macro npz, the same player's index there)."""
    macro_dir = data_root() / "interim" / "macro" / "v1"
    with open(macro_dir / "index.jsonl", encoding="utf-8") as f:
        macro = {r["sha1"]: r for r in map(json.loads, f) if r.get("ok")}
    out = []
    with open(army_dir() / "index.jsonl", encoding="utf-8") as f:
        rows = {r["sha1"]: r for r in map(json.loads, f)}
    for r in rows.values():
        if not r.get("ok") or r["matchup"] != matchup or r["sha1"] not in macro or (keep and not keep(r["sha1"])):
            continue
        path = str(army_dir() / r["sha1"][:2] / f"{r['sha1']}.npz")
        mpath = str(macro_dir / r["sha1"][:2] / f"{r['sha1']}.npz")
        mslots = [p["slot"] for p in macro[r["sha1"]]["players"]]
        for i, p in enumerate(r["players"]):
            if p["race"] == race and p["samples"] > 0 and p["slot"] in mslots:
                out.append((path, i, is_test(r["sha1"]), mpath, mslots.index(p["slot"])))
    return out


def scan(job: tuple) -> tuple[set, set]:
    d = np.load(job[0])
    i = job[1]
    return (set(np.nonzero(d[f"p{i}_units"].max(0))[0].tolist()),
            set(np.nonzero(d[f"p{i}_seen"].max(0))[0].tolist()))


def load_player(args: tuple) -> tuple[np.ndarray, ...]:
    (path, i, held, mpath, mi), spec_d = args
    spec = ArmySpec(**spec_d)
    d = np.load(path)
    n = len(d[f"p{i}_target"])
    rng = np.random.default_rng(hash(path) % 2**32 + i)
    idx = np.sort(rng.choice(n, MAX_PER_PLAYER, replace=False)) if n > MAX_PER_PLAYER and not held else np.arange(n)
    frames = d[f"p{i}_frame"][idx]
    # from the macro data, each second: what this player had seen, and both real armies
    m = np.load(mpath)
    mf, seen_all = m["frames"], m["seen"][:, mi].astype(np.float32)
    own_sup = m["units"][:, mi, :, 0].astype(np.float32) @ ARMY_SUPPLY
    enemy_sup = m["units"][:, 1 - mi, :, 0].astype(np.float32) @ ARMY_SUPPLY
    share = own_sup / (own_sup + enemy_sup + 1.0)
    t = np.clip(np.searchsorted(mf, frames, side="right") - 1, 0, len(mf) - 1)
    seen_mem = np.stack([seen_all[max(0, k - MEMORY_S):k + 1].max(0) for k in t])
    later = np.minimum(t + VALUE_HORIZON_S, len(mf) - 1)
    value = (share[later] - share[t]).astype(np.float32)
    grid = d[f"p{i}_grid"][idx]
    glob = encode_global(spec, frames.astype(np.float32), d[f"p{i}_eco"][idx].astype(np.float32),
                         d[f"p{i}_sup"][idx].astype(np.float32), d[f"p{i}_units"][idx].astype(np.float32),
                         d[f"p{i}_seen"][idx].astype(np.float32), seen_mem, own_sup[t], seen_mem @ ARMY_SUPPLY)
    group_cell = grid[:, 8].argmax(1).astype(np.int16)
    return grid, glob.astype(np.float16), d[f"p{i}_target"][idx], d[f"p{i}_kind"][idx], group_cell, value


def load(jobs: list, spec: ArmySpec, workers: int) -> dict:
    parts = {False: [], True: []}
    with ProcessPoolExecutor(workers) as pool:
        for job, res in zip(jobs, pool.map(load_player, [(j, spec.to_dict()) for j in jobs], chunksize=16)):
            parts[job[2]].append(res)
    return {("test" if held else "train"): tuple(np.concatenate([r[k] for r in rs]) for k in range(6))
            for held, rs in parts.items()}


def cell_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Chebyshev distance between cells, in cells."""
    return np.maximum(np.abs(a // GRID - b // GRID), np.abs(a % GRID - b % GRID))


def evaluate(net: ArmyNet, data: tuple, common: int, device: str) -> dict:
    grid, glob, target, kind, group, value = data
    net.eval()
    top5, kinds, vals = [], [], []
    with torch.no_grad():
        for s in range(0, len(target), 8192):
            g = torch.from_numpy(grid[s:s + 8192]).to(device).float().view(-1, CHANNELS, GRID, GRID)
            where, k, v = net(g, torch.from_numpy(glob[s:s + 8192]).to(device).float(),
                              torch.from_numpy(target[s:s + 8192].astype(np.int64)).to(device),
                              torch.from_numpy(kind[s:s + 8192].astype(np.int64)).to(device))
            top5.append(where.topk(5, -1).indices.cpu().numpy())
            kinds.append(k.argmax(-1).cpu().numpy())
            vals.append(v.cpu().numpy() / VALUE_SCALE)
    top5, kinds, vals = np.concatenate(top5), np.concatenate(kinds), np.concatenate(vals)
    clear = (kind == 1) & (np.abs(value) >= 0.05)
    t = target.astype(np.int64)
    first = top5[:, 0]
    far = cell_dist(t, group.astype(np.int64)) >= 3
    r = lambda x: round(float(x), 4)
    return {
        "orders": int(len(t)), "far_share": r(far.mean()),
        "cell_top1": r((first == t).mean()), "cell_top5": r((top5 == t[:, None]).any(1).mean()),
        "within1": r((cell_dist(first, t) <= 1).mean()),
        "far_within1": r((cell_dist(first, t) <= 1)[far].mean()),
        "far_top5": r((top5 == t[:, None]).any(1)[far].mean()),
        "kind_acc": r((kinds == kind).mean()),
        "baseline_stay_within1": r((cell_dist(group.astype(np.int64), t) <= 1).mean()),
        "baseline_stay_far_within1": r((cell_dist(group.astype(np.int64), t) <= 1)[far].mean()),
        "baseline_common_within1": r((cell_dist(np.full_like(t, common), t) <= 1).mean()),
        "baseline_kind_acc": r(max((kind == 0).mean(), (kind == 1).mean())),
        "fight_corr": r(np.corrcoef(vals, value)[0, 1]),
        "fight_sign_clear_attacks": r((np.sign(vals) == np.sign(value))[clear].mean()),
        "clear_attacks": int(clear.sum()),
        "fight_mae": r(np.abs(vals - value).mean()), "baseline_fight_mae": r(np.abs(value).mean()),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--race", required=True, choices=["T", "Z", "P"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    GameFilter.add_args(ap)
    args = ap.parse_args()
    keep = GameFilter.from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    jobs = player_games(args.matchup, args.race, keep)
    print(f"{len(jobs)} player-games ({keep.describe()})", flush=True)
    own, seen = set(), set()
    with ProcessPoolExecutor(args.workers) as pool:
        for o, s in pool.map(scan, [j for j in jobs if not j[2]], chunksize=32):
            own |= o
            seen |= s
    spec = ArmySpec(race=args.race, own_types=sorted(own), seen_types=sorted(seen))
    data = load(jobs, spec, args.workers)
    grid, glob, target, kind, group, value = data["train"]
    print(f"{len(jobs)} player-games; orders: train {len(target):,}, test {len(data['test'][2]):,} "
          f"({time.time() - t0:.0f} s)", flush=True)
    common = int(np.bincount(target.astype(np.int64), minlength=CELLS).argmax())

    config = {"net": {"ch": args.channels}, "matchup": args.matchup}
    net = ArmyNet(spec.n_global, **config["net"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(target) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    gt = torch.from_numpy(grid)                     # uint8 on the CPU; batches move to the GPU
    xt = torch.from_numpy(glob)
    yt = torch.from_numpy(target.astype(np.int64))
    kt = torch.from_numpy(kind.astype(np.int64))
    vt = torch.from_numpy(value * VALUE_SCALE)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(yt))
        total = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            g = gt[idx].to(device, non_blocking=True).float().view(-1, CHANNELS, GRID, GRID)
            y, kk = yt[idx].to(device), kt[idx].to(device)
            where, k, v = net(g, xt[idx].to(device).float(), y, kk)
            loss = F.cross_entropy(where, y) + 0.5 * F.cross_entropy(k, kk) + \
                0.5 * F.smooth_l1_loss(v, vt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        res = evaluate(net, data["test"], common, device)
        print(f"epoch {epoch + 1}: loss {total / len(yt):.3f}  within1 {res['within1']}  "
              f"far within1 {res['far_within1']}  top5 {res['cell_top5']}  kind {res['kind_acc']}  "
              f"fight corr {res['fight_corr']}  clear-attack calls {res['fight_sign_clear_attacks']}", flush=True)

    out = REPO_ROOT / "runs" / "army" / f"{args.matchup}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    ArmyModel(spec, net.cpu(), config).save(out / "model.pt")
    report = {"matchup": args.matchup, "race": args.race, "player_games": len(jobs),
              "train_orders": int(len(target)), "epochs": args.epochs,
              "minutes": round((time.time() - t0) / 60, 1), "test": res}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
