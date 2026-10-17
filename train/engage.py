"""The attack-or-hold decision, learned from the pros (its supervised start, before RL on paired branches):
at each army order in pro TvZ games, did the player send the group into the enemy (2+ cells away, to
a cell next to enemy buildings it knew or enemy army it saw lately) or not. The one army contrast
measured above the chaos: the whole army attacking or holding moves the next 90 s three times as
much as the hands' scatter does (eval/army_effect.py), against the army model's own alternatives,
mostly nearby cells (loops a1, a2 learned nothing from those).

The model: the imitation army model (the same inputs and their encoding, so it plays with the army
layer's tracker as is), its move/attack head retrained as engage/hold on the pros' orders (the rest of
the network fine-tuned gently). Games split by replay as the army model's (the same held-out games).
Scored on them: accuracy, AUC, the base rate.

    python -m train.engage --race T
Output: runs/engage/TvZ_<race>_<time>/model.pt (an army model: its kind head is engage) and report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
NEAR = 1                        # cells: a target this close to known enemy presence is into the enemy
MOVED = 2                       # cells: the group sent at least this far


def engage_labels(grid: np.ndarray, target: np.ndarray, group: np.ndarray) -> np.ndarray:
    """1 where the order sent the group 2+ cells to a cell next to enemy presence the player knew of
    (channels 3: enemy army visible, 6: enemy buildings ever seen, 7: enemy army seen in 30 s)."""
    from train.army import GRID, cell_dist
    enemy = (grid[:, 3] > 0) | (grid[:, 6] > 0) | (grid[:, 7] > 0)        # (n, 256)
    e = enemy.reshape(-1, GRID, GRID)
    ty, tx = target // GRID, target % GRID
    near = np.zeros(len(target), bool)
    for dy in range(-NEAR, NEAR + 1):
        for dx in range(-NEAR, NEAR + 1):
            y, x = np.clip(ty + dy, 0, GRID - 1), np.clip(tx + dx, 0, GRID - 1)
            near |= e[np.arange(len(target)), y, x]
    return (near & (cell_dist(target.astype(np.int64), group.astype(np.int64)) >= MOVED)).astype(np.int64)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", default="T", choices=["T", "Z"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    from gary.policy.army import CHANNELS, GRID, ArmyModel
    from train.army import load, player_games
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = sorted((REPO / "runs" / "army").glob(f"TvZ_{args.race}_*/model.pt"))[-1]
    base = ArmyModel.load(start)
    t0 = time.time()
    jobs = player_games("TvZ", args.race)
    data = load(jobs, base.spec, args.workers)
    grid, glob, target, kind, group, _ = data["train"]
    y = engage_labels(grid, target, group)
    tg, tx_, tt, _, tgr, _ = data["test"]
    ty = engage_labels(tg, tt, tgr)
    print(f"{len(jobs)} player-games; orders train {len(y):,} (engage {y.mean():.1%}), test {len(ty):,} "
          f"(engage {ty.mean():.1%}) ({time.time() - t0:.0f} s)", flush=True)
    net = copy.deepcopy(base.net).to(device)
    # the move/attack head relearned as engage/hold; the trunk fine-tuned at a tenth of the rate
    head = list(net.kind.parameters())
    trunk = [p for n, p in net.named_parameters() if not n.startswith("kind.")]
    opt = torch.optim.AdamW([{"params": head, "lr": args.lr}, {"params": trunk, "lr": args.lr / 10}], weight_decay=1e-4)
    gt, xt, yt = torch.from_numpy(grid), torch.from_numpy(glob), torch.from_numpy(y)
    # (unweighted: the model is drawn from in play, so its odds must be the pros' rate; the rarer class
    # weighted up made its mean 0.30 against the pros' 0.20, and Gary-T's army went in too often)
    w = torch.ones(2, dtype=torch.float32, device=device)

    def evaluate():
        net.eval()
        ps = []
        with torch.no_grad():
            for s in range(0, len(ty), 4096):
                g = torch.from_numpy(tg[s:s + 4096]).to(device).float().view(-1, CHANNELS, GRID, GRID)
                _, k, _ = net(g, torch.from_numpy(tx_[s:s + 4096]).to(device).float())
                ps.append(torch.softmax(k, -1)[:, 1].cpu().numpy())
        net.train()
        p = np.concatenate(ps)
        return {"auc": round(float(roc_auc_score(ty, p)), 3), "acc_at_half": round(float(((p > 0.5) == ty).mean()), 3),
                "mean_p": round(float(p.mean()), 3), "base_rate": round(float(ty.mean()), 3)}
    print("before (the move/attack head):", evaluate(), flush=True)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(yt))
        total = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            g = gt[idx].to(device).float().view(-1, CHANNELS, GRID, GRID)
            _, k, _ = net(g, xt[idx].to(device).float())
            loss = F.cross_entropy(k, yt[idx].to(device), weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}: loss {total / len(yt):.4f}  test {evaluate()}", flush=True)
    res = evaluate()
    out = REPO / "runs" / "engage" / f"TvZ_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    model = ArmyModel(base.spec, net.cpu().eval(), {**base.config, "engage": {"from": str(start), "near": NEAR,
                                                                                 "moved": MOVED}})
    model.save(out / "model.pt")
    (out / "report.json").write_text(json.dumps({"from": str(start), "orders_train": int(len(y)),
                                                  "orders_test": int(len(ty)), "test": res}, indent=1), encoding="utf-8")
    print(f"saved {out / 'model.pt'}: {res}")


if __name__ == "__main__":
    main()
