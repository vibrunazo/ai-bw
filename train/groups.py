"""Train the grouping model (gary/policy/grouping_model.py) on the pros' keys (ingest/key_labels.py's data): for each
snapshot, every pair of the player's own labeled fighters whose keys are known, 1 if they were on one key, 0 if not.
Games split as train/roles.py splits them (sha1 % 10 == 1 held out). Scored on the held-out pairs: precision and
recall at 0.5 and how the units partition into keys (gary/policy/grouping_model.py partition: pairs grouped
together that the pros had on one key), against the scripted provider's rule (the same type within 6 tiles).

    python -m train.groups --data data/interim/roles/response_T_att_again_keys.npz
Output: runs/groups/<time>/model.pt and report.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval import monitor, priority
from gary.policy.fight import SIDE, TYPE, X, Y
from gary.policy.grouping_model import GroupModel, GroupNet, partition, tensors

REPO_ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    d = np.load(path)
    val_game = np.array([int(s[:8], 16) % 10 == 1 for s in d["sha1"]])
    val = val_game[d["game"]]
    own = (d["labels"] >= 0) & (d["key"] >= -1) & (d["rows"][..., SIDE] == 1)
    keep = own.sum(1) >= 2
    arrays = (d["rows"], d["count"], d["time"], d["key"], own)
    pick = lambda m: tuple(a[m & keep] for a in arrays)
    return pick(~val), pick(val)


def batch(data, idx, device):
    rows, count, time_s, key, own = (a[idx] for a in data)
    n = int(count.max())
    rows, key, own = rows[:, :n], key[:, :n].astype(int), own[:, :n]
    pad = np.arange(n)[None, :] >= count[:, None]
    same = (key[:, :, None] == key[:, None, :]) & (key[:, :, None] >= 0)
    mask = own[:, :, None] & own[:, None, :] & ~np.eye(n, dtype=bool)[None]
    t = lambda x: torch.as_tensor(x, device=device)
    return tensors(rows, pad, time_s, device), t(same.astype(np.float32)), t(mask)


def evaluate(net, data, device, rule: bool = False) -> dict:
    net.eval()
    tp = fp = fn = 0
    part_tp = part_fp = part_fn = 0
    with torch.no_grad():
        for s in range(0, len(data[0]), 256):
            idx = np.arange(s, min(s + 256, len(data[0])))
            inputs, same, mask = batch(data, idx, device)
            if rule:                        # (the scripted provider's rule: the same type within 6 tiles)
                rows = data[0][idx][:, :same.shape[1]]
                ty = torch.as_tensor(rows[..., TYPE].astype(np.int64), device=device)
                xy = torch.as_tensor(rows[..., [X, Y]].astype(np.float32), device=device)
                prob = ((ty[:, :, None] == ty[:, None, :]) & (torch.cdist(xy, xy) < 6 * 32)).float()
            else:
                prob = torch.sigmoid(net(*inputs))
            pred = (prob > 0.5) & mask
            tp += int((pred & (same > 0)).sum()); fp += int((pred & (same == 0)).sum())
            fn += int((~pred & mask & (same > 0)).sum())
            p, sm, mk = prob.cpu().numpy(), same.cpu().numpy(), mask.cpu().numpy()
            for b in range(len(idx)):       # (partitions: up to 4 keys of 2+, the provider's way)
                units = np.nonzero(mk[b].any(1))[0]
                if len(units) < 2:
                    continue
                groups = partition(p[b][np.ix_(units, units)], 4)
                together = np.zeros((len(units), len(units)), bool)
                for g in groups:
                    together[np.ix_(g, g)] = True
                np.fill_diagonal(together, False)
                truth = sm[b][np.ix_(units, units)] > 0
                np.fill_diagonal(truth, False)
                part_tp += int((together & truth).sum()); part_fp += int((together & ~truth).sum())
                part_fn += int((~together & truth).sum())
    net.train()
    f = lambda a, b: round(a / max(1, a + b), 3)
    return {"pairs": {"precision": f(tp, fp), "recall": f(tp, fn)},
            "partition": {"precision": f(part_tp, part_fp), "recall": f(part_tp, part_fn)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/interim/roles/response_T_att_again_keys.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    train, val = load(Path(args.data))
    print(f"train {len(train[0])} snapshots, validation {len(val[0])}", flush=True)
    config = {"net": {"d": 96, "layers": 2, "heads": 4, "emb": 48}, "data": args.data}
    net = GroupNet(**config["net"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    rule = evaluate(net, val, device, rule=True)
    print(f"the scripted rule (same type within 6 tiles): {rule}", flush=True)
    best, best_state, history = -1.0, None, []
    for epoch in monitor.track(range(args.epochs), args.epochs, "epochs"):
        order = rng.permutation(len(train[0]))
        total = 0.0
        for s in range(0, len(order), args.batch):
            inputs, same, mask = batch(train, order[s:s + args.batch], device)
            logits = net(*inputs)
            loss = F.binary_cross_entropy_with_logits(logits[mask], same[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(same)
        res = evaluate(net, val, device)
        pp = res["partition"]
        score = 2 * pp["precision"] * pp["recall"] / max(1e-9, pp["precision"] + pp["recall"])
        history.append({"epoch": epoch, "loss": round(total / len(order), 4), "val": res})
        print(f"epoch {epoch} loss {total / len(order):.4f} val {res}", flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    out = REPO_ROOT / "runs" / "groups" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    GroupModel(net.cpu(), config).save(out / "model.pt")
    (out / "report.json").write_text(json.dumps({"rule": rule, "history": history}, indent=1), encoding="utf-8")
    print(f"saved {out / 'model.pt'}; the rule: {rule}; best partition F1 {best:.3f}")


if __name__ == "__main__":
    main()
