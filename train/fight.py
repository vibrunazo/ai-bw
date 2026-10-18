"""Train Gary's fight model (gary/policy/fight.py) on the fight training set (#2).

Behavior cloning: for each fight snapshot, what the pro told each of their units to do in the
next half second, which unit it targeted, and where moves went. Terran's units (Gary plays
Terran). Games split by replay, the same 10% held out as the other models. Scored on held-out
games:

  action        accuracy over the player's units, and per action how often the model names it
                when the pro did it (recall); baseline: always "nothing"
  target        for orders aimed at a unit, the model's first pick is the pro's target;
                baseline: the nearest unit of the right kind (enemy to attack, mineral to gather)
  destination   for moves, the median angle between the model's and the pro's direction

Usage:
  python -m train.fight
Output: runs/fight/<matchup>_<race>_<time>/model.pt and report.json
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

from gary.policy.fight import ACTIONS, DEST_SCALE, MAX_UNITS, OTARGET, SIDE, X, Y, FightModel, FightNet, tensors  # noqa: E402
from train.macro import is_test  # noqa: E402

PER_GAME = 100                  # snapshots per game, at most
KEEP_IDLE = 0.3                 # share of snapshots where the pro ordered nothing that we keep
A = {a: i for i, a in enumerate(ACTIONS)}
TARGETED = {A["attack_unit"]: 0, A["gather"]: 2, A["own_unit"]: 1}   # action -> side of its target


def fight_dir(version: str = "v1") -> Path:
    return data_root() / "interim" / "fight" / version


def jobs_for(matchup: str, race: str, version: str = "v1", keep=None) -> list[tuple[str, int, bool]]:
    out = []
    with open(fight_dir(version) / "index.jsonl", encoding="utf-8") as f:
        rows = {r["sha1"]: r for r in map(json.loads, f)}
    for r in rows.values():
        if not r.get("ok") or r["matchup"] != matchup or (keep and not keep(r["sha1"])):
            continue
        for i, p in enumerate(r["players"]):
            if p["race"] == race and p["snapshots"] > 0:
                out.append((str(fight_dir(version) / r["sha1"][:2] / f"{r['sha1']}.npz"), i, is_test(r["sha1"])))
    return out


def load_game(job: tuple) -> tuple[np.ndarray, ...] | None:
    """Padded snapshots of one player in one game."""
    path, player, held = job
    d = np.load(path)
    starts, units = d["snap_start"], d["units"]
    act, act_t, dx, dy = d["act"], d["act_target"], d["act_dx"], d["act_dy"]
    rng = np.random.default_rng(int(path[-12:-4], 16) + player)
    keep = []
    for k in np.nonzero(d["snap_player"] == player)[0]:
        a = act[starts[k]:starts[k + 1]]
        if (a > 0).any() or rng.random() < KEEP_IDLE:
            keep.append(k)
    if len(keep) > PER_GAME and not held:
        keep = sorted(rng.choice(keep, PER_GAME, replace=False))
    if not keep:
        return None
    n = len(keep)
    U = np.zeros((n, MAX_UNITS, 11), np.int16)
    PAD = np.ones((n, MAX_UNITS), bool)
    ACT = np.full((n, MAX_UNITS), -1, np.int8)
    TGT = np.full((n, MAX_UNITS), -1, np.int16)
    DST = np.zeros((n, MAX_UNITS, 2), np.int16)
    T = np.zeros(n, np.float32)
    for j, k in enumerate(keep):
        s, e = starts[k], starts[k + 1]
        rows = units[s:e]
        order = np.argsort(rows[:, X].astype(np.int32) ** 2 + rows[:, Y].astype(np.int32) ** 2)[:MAX_UNITS]
        remap = np.full(e - s, -1, np.int16)
        remap[order] = np.arange(len(order))
        sel = rows[order].copy()
        ot = sel[:, OTARGET]
        sel[:, OTARGET] = np.where(ot >= 0, remap[np.maximum(ot, 0)], -1)
        m = len(order)
        U[j, :m], PAD[j, :m] = sel, False
        ACT[j, :m] = act[s:e][order]
        t = act_t[s:e][order]
        TGT[j, :m] = np.where(t >= 0, remap[np.maximum(t, 0)], -1)
        DST[j, :m, 0], DST[j, :m, 1] = dx[s:e][order], dy[s:e][order]
        T[j] = d["snap_frame"][k] * 42 / 1000.0            # game seconds
    return U, PAD, ACT, TGT, DST, T, np.full(n, held)


def load_all(jobs: list, workers: int) -> dict:
    parts = []
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(load_game, jobs, chunksize=16):
            if r is not None:
                parts.append(r)
    allp = [np.concatenate([p[k] for p in parts]) for k in range(7)]
    held = allp[6]
    return {"train": tuple(a[~held] for a in allp[:6]), "test": tuple(a[held] for a in allp[:6])}


def batch(data: tuple, idx: np.ndarray, device: str):
    U, PAD, ACT, TGT, DST, T = (a[idx] for a in data)
    inputs = tensors(U, PAD, T, device)
    return inputs, (torch.as_tensor(ACT.astype(np.int64), device=device),
                    torch.as_tensor(TGT.astype(np.int64), device=device),
                    torch.as_tensor(DST.astype(np.float32), device=device) / DEST_SCALE)


def losses(net: FightNet, inputs, labels, weights: torch.Tensor | None = None) -> torch.Tensor:
    act_l, tgt_l, dst_l = net(*inputs)
    act, tgt, dst = labels
    own = act >= 0
    loss = F.cross_entropy(act_l[own], act[own], weight=weights)
    aimed = own & (tgt >= 0)
    if aimed.any():
        loss = loss + F.cross_entropy(tgt_l[aimed], tgt[aimed])
    moving = own & ((act == A["move"]) | (act == A["attack_move"]))
    if moving.any():
        loss = loss + F.smooth_l1_loss(dst_l[moving], dst[moving].clamp(-4, 4))
    return loss


@torch.no_grad()
def evaluate(net: FightNet, data: tuple, device: str) -> dict:
    net.eval()
    acts, preds, tgt_hit, tgt_base, angles = [], [], [], [], []
    U, PAD = data[0], data[1]
    for s in range(0, len(U), 2048):
        idx = np.arange(s, min(s + 2048, len(U)))
        inputs, (act, tgt, dst) = batch(data, idx, device)
        act_l, tgt_l, dst_l = net(*inputs)
        own = act >= 0
        acts.append(act[own].cpu().numpy())
        preds.append(act_l.argmax(-1)[own].cpu().numpy())
        aimed = own & (tgt >= 0)
        tgt_hit.append((tgt_l.argmax(-1)[aimed] == tgt[aimed]).cpu().numpy())
        # baseline: the nearest unit of the side the action aims at
        u = torch.as_tensor(U[idx].astype(np.float32), device=device)
        xy = u[..., [X, Y]]
        dist = torch.cdist(xy, xy)
        side = u[..., SIDE].long()
        want = torch.zeros_like(act)
        for a, sd in TARGETED.items():
            want = torch.where(act == a, torch.full_like(act, sd), want)
        bad = (side[:, None, :] != want[:, :, None]) | inputs[4][:, None, :]
        dist = dist.masked_fill(bad, 1e9)
        dist = dist + torch.eye(dist.shape[-1], device=device)[None] * 1e9
        tgt_base.append((dist.argmin(-1)[aimed] == tgt[aimed]).cpu().numpy())
        moving = own & ((act == A["move"]) | (act == A["attack_move"]))
        a1 = torch.atan2(dst_l[..., 1], dst_l[..., 0])[moving]
        a2 = torch.atan2(dst[..., 1], dst[..., 0])[moving]
        diff = torch.remainder(a1 - a2 + np.pi, 2 * np.pi) - np.pi
        angles.append(diff.abs().cpu().numpy() * 180 / np.pi)
    acts, preds = np.concatenate(acts), np.concatenate(preds)
    th, tb, ang = np.concatenate(tgt_hit), np.concatenate(tgt_base), np.concatenate(angles)
    r = lambda x: round(float(x), 4)
    recall = {ACTIONS[a]: r((preds[acts == a] == a).mean()) for a in range(len(ACTIONS)) if (acts == a).sum() >= 50}
    return {"units": int(len(acts)), "action_acc": r((preds == acts).mean()),
            "baseline_action_acc": r((acts == 0).mean()), "commanded_acc": r((preds == acts)[acts > 0].mean()),
            "recall": recall, "target_acc": r(th.mean()), "baseline_target_acc": r(tb.mean()),
            "aimed_units": int(len(th)), "move_angle_median_deg": r(np.median(ang)), "moves": int(len(ang))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--race", default="T", choices=["T", "Z", "P"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only this many player-games (quick tests)")
    ap.add_argument("--class-weight", type=float, default=0.5,
                    help="weigh each action by frequency^-this in the loss (rare actions count more); 0 = off")
    GameFilter.add_args(ap)
    args = ap.parse_args()
    keep = GameFilter.from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    jobs = jobs_for(args.matchup, args.race, keep=keep)[:args.limit]
    print(keep.describe(), flush=True)
    data = load_all(jobs, args.workers)
    n_train, n_test = len(data["train"][0]), len(data["test"][0])
    print(f"{len(jobs)} player-games; snapshots: train {n_train:,}, test {n_test:,} ({time.time() - t0:.0f} s)", flush=True)
    config = {"net": {"d": 128, "layers": 3, "heads": 4}, "matchup": args.matchup, "race": args.race}
    net = FightNet(**config["net"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (n_train // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    # rare actions (attacking a unit, stim, hold...) would otherwise never beat "nothing" and "move"
    acts = data["train"][2]
    counts = np.bincount(acts[acts >= 0].astype(np.int64), minlength=len(ACTIONS)).astype(np.float64)
    w = np.where(counts > 0, (counts / counts.sum()) ** -args.class_weight, 0.0)
    weights = torch.tensor(w / (w * counts).sum() * counts.sum(), dtype=torch.float32, device=device)
    print("action weights:", {a: round(float(x), 2) for a, x in zip(ACTIONS, weights.tolist())}, flush=True)
    config["class_weight"] = args.class_weight
    for epoch in range(args.epochs):
        net.train()
        perm = np.random.permutation(n_train)
        total, nb = 0.0, 0
        for s in range(0, n_train, args.batch):
            inputs, labels = batch(data["train"], perm[s:s + args.batch], device)
            loss = losses(net, inputs, labels, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total, nb = total + loss.item(), nb + 1
        res = evaluate(net, data["test"], device)
        print(f"epoch {epoch + 1}: loss {total / nb:.3f}  action {res['action_acc']} (none-baseline "
              f"{res['baseline_action_acc']})  commanded {res['commanded_acc']}  target {res['target_acc']} "
              f"(nearest {res['baseline_target_acc']})  move angle {res['move_angle_median_deg']} deg  "
              f"({time.time() - t0:.0f} s)", flush=True)
    out = REPO_ROOT / "runs" / "fight" / f"{args.matchup}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    FightModel(net.cpu(), config).save(out / "model.pt")
    report = {"matchup": args.matchup, "race": args.race, "train_snapshots": n_train, "epochs": args.epochs,
              "minutes": round((time.time() - t0) / 60, 1), "test": res}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
