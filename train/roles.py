"""Train Gary's role model (gary/policy/roles.py) on the role dataset (ingest/role_dataset.py).

Imitation: for each fight snapshot, the role each of the pro's units had over the next half
second. With --decisions, only when it changed (the pro decided something: the unit's role
differs from its role half a second before), else "keep": what a player decides, not what their
units are still doing. Scenarios split as the RL trainer splits them (sha1 % 10 == 1 held out for validation).
Rare roles (pull back, retreat) weigh more in the loss, by the inverse square root of their share.
Scored on the held-out scenarios: accuracy, and per role how often the model names it when the
pro did it (recall) and how often the pro did it when the model names it (precision); baseline:
always "fire".

    python -m train.roles
Output: runs/roles/<time>/model.pt and report.json
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
from gary.policy.roles import DECISIONS, ROLES, RoleModel, RoleNet, tensors
from eval.micro_roles import NEAR
from ingest.inventory import data_root

PRE_VAL: list = []                      # the validation set's labels before contact only (if the data has them)

REPO_ROOT = Path(__file__).resolve().parent.parent


def decision_labels(d) -> np.ndarray:
    """A unit's role where it changed from half a second before (or the unit wasn't labeled then),
    else keep."""
    labels, tags, game, time_s = d["labels"], d["tags"], d["game"], d["time"]
    keep = len(ROLES)
    out = labels.copy()
    prev: dict = {}
    for i in range(len(labels)):
        frame = int(round(time_s[i] * 1000 / 42))
        now = {}
        for j in range(int(d["count"][i])):
            y = int(labels[i, j])
            if y < 0:
                continue
            key = (int(game[i]), int(tags[i, j]))
            p = prev.get(key)
            if p is not None and frame - p[0] <= 14 and p[1] == y:
                out[i, j] = keep
            now[key] = (frame, y)
        prev.update(now)
    return out


def load(path: Path, decisions: bool = False, move: bool = False):
    """Train and validation arrays, and the label names (a dataset can name its own: the hand rules'
    decisions, ingest/role_distill.py)."""
    d = np.load(path)
    val_game = np.array([int(s[:8], 16) % 10 == 1 for s in d["sha1"]])
    val = val_game[d["game"]]
    if "names" in d:
        names = [str(x) for x in d["names"]]
        labels = d["labels"]
    else:
        names = DECISIONS if decisions else ROLES
        labels = decision_labels(d) if decisions else d["labels"]
    arrays = {"rows": d["rows"], "extra": d["extra"], "labels": labels, "count": d["count"], "time": d["time"]}
    keys = ("rows", "extra", "labels", "count", "time")
    if "move" in d and move:            # (ingest/response_dataset.py: each unit's move over the half second)
        arrays["move"] = d["move"].astype(np.float32)
        keys += ("move",)
    pick = lambda m: tuple(arrays[k][m] for k in keys)
    if "near" in d:                     # (ingest/response_dataset.py: labels before contact scored apart too)
        arrays["labels_pre"] = np.where(d["near"] >= NEAR, labels, -1).astype(labels.dtype)
        PRE_VAL[:] = [tuple(arrays["labels_pre" if k == "labels" else k][val] for k in keys)]
    return pick(~val), pick(val), names


def batch(data, idx, device, with_move: bool = False):
    rows, extra, labels, count, time_s = (a[idx] for a in data[:5])
    n = int(count.max())
    rows, extra, labels = rows[:, :n], extra[:, :n], labels[:, :n]
    pad = np.arange(n)[None, :] >= count[:, None]
    out = tensors(rows, extra, pad, time_s, device), torch.as_tensor(labels.astype(np.int64), device=device)
    if with_move:
        return out + (torch.as_tensor(data[5][idx][:, :n], device=device),)
    return out


def evaluate(net, data, device, names=ROLES) -> dict:
    net.eval()
    pred, true, err, still, moved_role = [], [], [], [], []
    with_move = len(data) > 5 and net.move is not None
    with torch.no_grad():
        for s in range(0, len(data[0]), 512):
            got = batch(data, np.arange(s, min(s + 512, len(data[0]))), device, with_move)
            inputs, labels = got[0], got[1]
            if with_move:
                logits, move = net(*inputs, with_move=True)
                m = (labels >= 0) & ~torch.isnan(got[2]).any(-1)
                err.append((move[m] - got[2][m]).norm(dim=-1).cpu().numpy())
                still.append(got[2][m].norm(dim=-1).cpu().numpy())
                moved_role.append(labels[m].cpu().numpy())
            else:
                logits = net(*inputs)
            p = logits.argmax(-1)
            m = labels >= 0
            pred.append(p[m].cpu().numpy())
            true.append(labels[m].cpu().numpy())
    net.train()
    pred, true = np.concatenate(pred), np.concatenate(true)
    move_report = {}
    if err:
        err, still, moved_role = np.concatenate(err), np.concatenate(still), np.concatenate(moved_role)
        for r in ("reposition", "pull_back", "retreat", "close_in"):
            k = names.index(r) if r in names else -1
            sel = moved_role == k
            if sel.any():               # (against guessing no move: the move's own length)
                move_report[r] = {"error_tiles": round(float(np.median(err[sel])), 2),
                                  "no_move_tiles": round(float(np.median(still[sel])), 2)}
    out = {"units": len(true), "accuracy": round(float((pred == true).mean()), 3),
           "baseline_fire": round(float((true == 0).mean()), 3)}
    if move_report:
        out["move"] = move_report
    for k, r in enumerate(names):
        tp = int(((pred == k) & (true == k)).sum())
        out[r] = {"share": round(float((true == k).mean()), 3),
                  "recall": round(tp / max(1, int((true == k).sum())), 3),
                  "precision": round(tp / max(1, int((pred == k).sum())), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(data_root() / "interim" / "roles" / "roles_train.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--decisions", action="store_true", help="learn when the role changes (else keep)")
    ap.add_argument("--no-weights", action="store_true", help="plain cross entropy (rare roles not weighted up)")
    ap.add_argument("--extra", type=int, help="read only the first this many per-unit extras (6: without the terrain)")
    ap.add_argument("--move", action="store_true", help="also learn where each unit moves (the data's move)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="label smoothing: every decision keeps some probability (room to explore for RL)")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    train, val, names = load(Path(args.data), args.decisions, args.move)
    decisions = "keep" in names
    y = train[2][train[2] >= 0]
    share = np.bincount(y, minlength=len(names)) / len(y)
    weights = torch.as_tensor(1 / np.sqrt(np.maximum(share, 1e-3)), dtype=torch.float32, device=device)
    weights = weights / weights[0] if not args.no_weights else torch.ones_like(weights)
    print(f"train {len(train[0])} snapshots, {len(y)} units; validation {len(val[0])} snapshots; "
          + ", ".join(f"{r} {s:.1%}" for r, s in zip(names, share)), flush=True)
    config = {"net": {"d": args.d, "layers": args.layers, "heads": 4, "n_roles": len(names),
                      "n_extra": args.extra or int(train[1].shape[-1]), "move": bool(args.move)},
              "data": args.data, "roles": names}
    net = RoleNet(**config["net"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    best, best_state, history = -1.0, None, []
    for epoch in monitor.track(range(args.epochs), args.epochs, "epochs"):
        order = rng.permutation(len(train[0]))
        total = 0.0
        for s in range(0, len(order), args.batch):
            got = batch(train, order[s:s + args.batch], device, args.move)
            inputs, labels = got[0], got[1]
            if args.move:
                logits, move = net(*inputs, with_move=True)
                m = (labels >= 0) & ~torch.isnan(got[2]).any(-1)
                move_loss = F.smooth_l1_loss(move[m], got[2][m]) if m.any() else move.sum() * 0
            else:
                logits, move_loss = net(*inputs), 0.0
            loss = F.cross_entropy(logits.reshape(-1, len(names)), labels.reshape(-1), weight=weights, ignore_index=-1,
                                   label_smoothing=args.smooth) + move_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(inputs[0])
        res = evaluate(net, val, device, names)
        # model choice: over the roles that drive commands (not just the majority), mean recall; for
        # decisions mean F1 (a decision named wrongly is a needless command)
        f1 = lambda r: 2 * res[r]["recall"] * res[r]["precision"] / max(1e-9, res[r]["recall"] + res[r]["precision"])
        score = np.mean([f1(r) if decisions else res[r]["recall"] for r in ("fire", "close_in", "pull_back", "retreat")])
        history.append({"epoch": epoch, "loss": round(total / len(order), 4), "val": res, "score": round(float(score), 3)})
        print(f"epoch {epoch:2d} loss {total / len(order):.3f} val acc {res['accuracy']:.3f} "
              + " ".join(f"{r} {res[r]['recall']:.2f}/{res[r]['precision']:.2f}" for r in ROLES[:4]), flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    out = REPO_ROOT / "runs" / "roles" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    RoleModel(net.cpu(), config).save(out / "model.pt")
    final = evaluate(net.to(device), val, device, names)
    report = {"val": final, "history": history}
    if PRE_VAL:
        pre = evaluate(net, PRE_VAL[0], device, names)
        report["val_before_contact"] = pre
        print("before contact (recall/precision): "
              + ", ".join(f"{r} {pre[r]['recall']:.2f}/{pre[r]['precision']:.2f} ({pre[r]['share']:.0%})" for r in names)
              + f"; accuracy {pre['accuracy']}" + (f"; moves {pre['move']}" if "move" in pre else ""))
    if "move" in final:
        print(f"moves (median error, tiles; and if it guessed no move): {final['move']}")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved {out / 'model.pt'}; validation (recall/precision): "
          + ", ".join(f"{r} {final[r]['recall']:.2f}/{final[r]['precision']:.2f}" for r in names)
          + f"; accuracy {final['accuracy']} (always fire: {final['baseline_fire']})")


if __name__ == "__main__":
    main()
