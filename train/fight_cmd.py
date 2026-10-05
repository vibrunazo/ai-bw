"""Train Gary's fight command model (gary/policy/fight_cmd.py) on the fight training set (#2).

Behavior cloning, one command at a time: for each fight snapshot, the commands the pro gave in
the next half second. They're rebuilt from the fight set's per-unit labels: the player's units
told the same thing in that window (the same action and target unit, or for moves the same
point) were one command, and they were its selection (as far as it's in the snapshot).
Snapshots without a command teach "nothing". With --memory (fight set v3), the model also sees
the pro's own commands of the last seconds (gary/policy/fight_memory.py). Terran's units, the
same 10% of games held out as the other models. Scored on held-out games:

  command      the command type: accuracy, and when the pro commanded, how often the model's
               likeliest real command is the pro's (baseline: "nothing" / the most common command)
  selection    given the type: IoU of the selected units with the pro's (baseline: all own units)
  target       given type and selection, the pro's target unit (baseline: the nearest unit of the
               right side to the selection)
  destination  for moves, the pro's cell or a neighbor (baseline: the selection's own cell)

Usage:
  python -m train.fight_cmd                      # fight set v2, no memory
  python -m train.fight_cmd --memory --data v3   # with memory, from the latest command model
Output: runs/fight_cmd/<matchup>_<race>_<time>/model.pt and report.json
"""

from __future__ import annotations

import argparse
import bisect
import functools
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

from gary.agent.manifests import latest_fight_model  # noqa: E402
from gary.policy.fight import ACTIONS, OTARGET, SIDE, X, Y, tensors  # noqa: E402
from gary.policy.fight_cmd import GRID, MAX_UNITS, MOVES, POINTER, CommandModel, CommandNet, cell_of  # noqa: E402
from gary.policy.fight_memory import HIST_K, history_inputs  # noqa: E402
from train.fight import KEEP_IDLE, PER_GAME, jobs_for  # noqa: E402

A = {a: i for i, a in enumerate(ACTIONS)}
TARGET_SIDE = {A["attack_unit"]: 0, A["gather"]: 2, A["own_unit"]: 1}
HIST_KEYS = ("UK", "UN", "TOK", "TK", "TA")


def load_game(job: tuple, memory: bool = False):
    """Command examples of one player in one game, as a dict of arrays: units, padding, command,
    selection, target row, destination cell, time, held out, and with memory the history inputs."""
    path, player, held = job
    d = np.load(path)
    starts, units = d["snap_start"], d["units"]
    act, act_t, dx, dy = d["act"], d["act_target"], d["act_dx"], d["act_dy"]
    rng = np.random.default_rng(int(path[-12:-4], 16) + player)
    snaps = list(np.nonzero(d["snap_player"] == player)[0])
    if len(snaps) > PER_GAME and not held:
        snaps = sorted(rng.choice(snaps, PER_GAME, replace=False))
    keys = ("U", "PAD", "CMD", "SEL", "TGT", "CELL", "T") + (HIST_KEYS if memory else ())
    out = {k: [] for k in keys}
    if memory:                                   # the player's commands, oldest first
        cs, cu = d["cmd_start"], d["cmd_units"]
        idx = np.nonzero(d["cmd_player"] == player)[0]
        cmds = [(int(d["cmd_frame"][i]), int(d["cmd_kind"][i]), tuple(cu[cs[i]:cs[i + 1]].tolist()),
                 int(d["cmd_x"][i]), int(d["cmd_y"][i]), int(d["cmd_target"][i])) for i in idx]
        cmd_frames = [c[0] for c in cmds]
        flip = tuple(bool(v) for v in d["flip"][player])

    def add(rows, n, cmd, sel, tgt, cell, t, hist):
        u = np.zeros((MAX_UNITS, 11), np.int16)
        u[:n] = rows
        out["U"].append(u); out["PAD"].append(np.arange(MAX_UNITS) >= n); out["CMD"].append(cmd)
        out["SEL"].append(sel); out["TGT"].append(tgt); out["CELL"].append(cell); out["T"].append(t)
        if memory:
            u_kind, u_num, tok, t_kind, t_absent = hist
            uk = np.zeros(MAX_UNITS, np.int8); uk[:n] = u_kind
            un = np.zeros((MAX_UNITS, u_num.shape[1]), np.float16); un[:n] = u_num
            out["UK"].append(uk); out["UN"].append(un); out["TOK"].append(tok)
            out["TK"].append(t_kind.astype(np.int8)); out["TA"].append(t_absent)

    for k in snaps:
        s, e = starts[k], starts[k + 1]
        rows = units[s:e]
        order = np.argsort(rows[:, X].astype(np.int32) ** 2 + rows[:, Y].astype(np.int32) ** 2)[:MAX_UNITS]
        n = len(order)
        remap = np.full(e - s, -1, np.int16)
        remap[order] = np.arange(n)
        r = rows[order].copy()
        ot = r[:, OTARGET]
        r[:, OTARGET] = np.where(ot >= 0, remap[np.maximum(ot, 0)], -1)
        a, t = act[s:e][order], act_t[s:e][order]
        t = np.where(t >= 0, remap[np.maximum(t, 0)], -1)
        ax = r[:, X].astype(np.int32) + dx[s:e][order]
        ay = r[:, Y].astype(np.int32) + dy[s:e][order]
        frame = int(d["snap_frame"][k])
        time_s = frame * 42 / 1000.0
        hist = None
        if memory:                               # commands up to the snapshot (later ones are the labels)
            j = bisect.bisect_right(cmd_frames, frame)
            hist = history_inputs(d["unit_tag"][s:e][order].tolist(), r, (int(d["snap_cx"][k]), int(d["snap_cy"][k])),
                                  flip, frame, cmds[max(0, j - HIST_K):j])
        groups: dict[tuple, list[int]] = {}
        for i in np.nonzero(a > 0)[0]:
            ai = int(a[i])
            key = (ai, int(t[i])) if ai in POINTER else \
                (ai, int(ax[i]) // 8, int(ay[i]) // 8) if ai in MOVES else (ai,)
            groups.setdefault(key, []).append(int(i))
        if not groups:
            if rng.random() < KEEP_IDLE:
                add(r, n, 0, np.zeros(MAX_UNITS, bool), -1, -1, time_s, hist)
            continue
        for key, members in groups.items():
            sel = np.zeros(MAX_UNITS, bool)
            sel[members] = True
            tgt = key[1] if key[0] in POINTER and key[1] not in members else -1
            cell = int(cell_of(ax[members[0]], ay[members[0]])) if key[0] in MOVES else -1
            add(r, n, key[0], sel, tgt, cell, time_s, hist)
    if not out["U"]:
        return None
    res = {k: np.stack(v) if k not in ("CMD", "TGT", "CELL", "T") else np.array(v) for k, v in out.items()}
    for k in ("CMD", "TGT", "CELL"):
        res[k] = res[k].astype(np.int64)
    res["T"] = res["T"].astype(np.float32)
    res["HELD"] = np.full(len(res["U"]), held)
    return res


def load_all(jobs: list, workers: int, memory: bool) -> dict:
    parts = []
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(functools.partial(load_game, memory=memory), jobs, chunksize=16):
            if r is not None:
                parts.append(r)
    allp = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    held = allp.pop("HELD")
    return {"train": {k: v[~held] for k, v in allp.items()}, "test": {k: v[held] for k, v in allp.items()}}


def batch(data: dict, idx: np.ndarray, device: str):
    g = lambda k: data[k][idx]
    inputs = tensors(g("U"), g("PAD"), g("T"), device)
    lab = lambda x: torch.as_tensor(x, device=device)
    hist = None
    if "UK" in data:
        hist = (lab(g("UK").astype(np.int64)), lab(g("UN").astype(np.float32)), lab(g("TOK")),
                lab(g("TK").astype(np.int64)), lab(g("TA")))
    return inputs, hist, (lab(g("CMD")), lab(g("SEL")), lab(g("TGT")), lab(g("CELL")))


def forward(net: CommandNet, inputs, hist, labels):
    cmd, sel, tgt, cell = labels
    return net(*inputs, cmd, sel, hist)


def losses(net: CommandNet, inputs, hist, labels, weights: torch.Tensor | None = None) -> torch.Tensor:
    cmd, sel, tgt, cell = labels
    cmd_l, sel_l, tgt_l, dst_l = forward(net, inputs, hist, labels)
    loss = F.cross_entropy(cmd_l, cmd, weight=weights)
    side, pad = inputs[1], inputs[4]
    own = (side == 1) & ~pad & (cmd > 0)[:, None]
    if own.any():
        loss = loss + F.binary_cross_entropy_with_logits(sel_l[own], sel[own].float())
    aimed = tgt >= 0
    if aimed.any():
        loss = loss + F.cross_entropy(tgt_l[aimed], tgt[aimed])
    moving = cell >= 0
    if moving.any():
        loss = loss + F.cross_entropy(dst_l[moving], cell[moving])
    return loss


@torch.no_grad()
def evaluate(net: CommandNet, data: dict, device: str) -> dict:
    net.eval()
    cmds, preds, preds_real, ious, ious_base, hits, hits_base, near, near_base = ([] for _ in range(9))
    U = data["U"]
    common = int(np.bincount(data["CMD"][data["CMD"] > 0], minlength=len(ACTIONS)).argmax())
    for s in range(0, len(U), 2048):
        idx = np.arange(s, min(s + 2048, len(U)))
        inputs, hist, labels = batch(data, idx, device)
        cmd, sel, tgt, cell = labels
        cmd_l, sel_l, tgt_l, dst_l = forward(net, inputs, hist, labels)
        cmds.append(cmd.cpu().numpy())
        preds.append(cmd_l.argmax(-1).cpu().numpy())
        preds_real.append((cmd_l[:, 1:].argmax(-1) + 1).cpu().numpy())
        side, pad = inputs[1], inputs[4]
        own = (side == 1) & ~pad
        acted = cmd > 0
        pick = (sel_l > 0) & own
        iou = (pick & sel).sum(1) / (pick | sel).sum(1).clamp(min=1)
        base = (own & sel).sum(1) / (own | sel).sum(1).clamp(min=1)
        ious.append(iou[acted].cpu().numpy()); ious_base.append(base[acted].cpu().numpy())
        aimed = tgt >= 0
        hits.append((tgt_l.argmax(-1) == tgt)[aimed].cpu().numpy())
        # baseline: the unit of the right side nearest to the selection's center
        u = torch.as_tensor(U[idx].astype(np.float32), device=device)
        xy = u[..., [X, Y]]
        sf = sel.float()
        center = (xy * sf[..., None]).sum(1) / sf.sum(1, keepdim=True).clamp(min=1)
        dist = (xy - center[:, None]).norm(dim=-1)
        want = torch.zeros_like(cmd)
        for a, sd in TARGET_SIDE.items():
            want = torch.where(cmd == a, torch.full_like(cmd, sd), want)
        dist = dist.masked_fill((side != want[:, None]) | pad | sel, 1e9)
        hits_base.append((dist.argmin(-1) == tgt)[aimed].cpu().numpy())
        moving = cell >= 0
        guess = dst_l.argmax(-1)
        own_cell = torch.as_tensor(cell_of(center[:, 0].cpu().numpy(), center[:, 1].cpu().numpy()), device=device)
        close = lambda c: ((c % GRID - cell % GRID).abs() <= 1) & ((c // GRID - cell // GRID).abs() <= 1)
        near.append(close(guess)[moving].cpu().numpy()); near_base.append(close(own_cell)[moving].cpu().numpy())
    cmds, preds, preds_real = np.concatenate(cmds), np.concatenate(preds), np.concatenate(preds_real)
    cat = lambda x: np.concatenate(x)
    r = lambda x: round(float(np.mean(x)), 4)
    acted = cmds > 0
    recall = {ACTIONS[a]: r(preds_real[cmds == a] == a) for a in range(1, len(ACTIONS)) if (cmds == a).sum() >= 50}
    return {"examples": int(len(cmds)), "commands": int(acted.sum()),
            "command_acc": r(preds == cmds), "baseline_command_acc": r(cmds == 0),
            "commanded_acc": r(preds_real[acted] == cmds[acted]), "baseline_commanded_acc": r(cmds[acted] == common),
            "recall": recall, "selection_iou": r(cat(ious)), "baseline_selection_iou": r(cat(ious_base)),
            "target_acc": r(cat(hits)), "baseline_target_acc": r(cat(hits_base)),
            "dest_near": r(cat(near)), "baseline_dest_near": r(cat(near_base))}


def warm_start(net: CommandNet, args, device: str, config: dict) -> None:
    """Memory models start from the latest command model (the new inputs' weights at zero: the
    same model, until it learns to use them); others from the per-unit fight model's encoder."""
    if args.memory:
        from gary.agent.manifests import latest_command_model
        init = latest_command_model(args.matchup, args.race)
        state = torch.load(init, map_location=device, weights_only=True)["state"]
        own = net.state_dict()
        w = state.pop("inp.weight")
        new_w = torch.zeros_like(own["inp.weight"])
        new_w[:, :w.shape[1]] = w
        state["inp.weight"] = new_w
        net.load_state_dict(state, strict=False)
    else:
        init = latest_fight_model(args.matchup, args.race)
        state = torch.load(init, map_location=device, weights_only=True)["state"]
        enc = {k: v for k, v in state.items() if k.split(".")[0] in ("type_emb", "side_emb", "order_emb", "inp", "body")}
        net.load_state_dict(enc, strict=False)
    config["init_from"] = str(init.relative_to(REPO_ROOT))
    print(f"starting from {init}", flush=True)


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
                    help="weigh each command type by frequency^-this in the loss; 0 = off")
    ap.add_argument("--no-init", action="store_true", help="start from scratch")
    ap.add_argument("--data", default="v2", help="fight set version (ingest/fight_dataset.py)")
    ap.add_argument("--memory", action="store_true", help="the pro's recent commands as input (needs --data v3)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    jobs = jobs_for(args.matchup, args.race, args.data)[:args.limit]
    data = load_all(jobs, args.workers, args.memory)
    n_train, n_test = len(data["train"]["U"]), len(data["test"]["U"])
    print(f"{len(jobs)} player-games; examples: train {n_train:,}, test {n_test:,} ({time.time() - t0:.0f} s)", flush=True)
    config = {"net": {"d": 128, "layers": 3, "heads": 4, "memory": args.memory}, "matchup": args.matchup,
              "race": args.race, "class_weight": args.class_weight, "data": args.data}
    net = CommandNet(**config["net"]).to(device)
    if not args.no_init:
        warm_start(net, args, device, config)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (n_train // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    counts = np.bincount(data["train"]["CMD"], minlength=len(ACTIONS)).astype(np.float64)
    w = np.where(counts > 0, (counts / counts.sum()) ** -args.class_weight, 0.0)
    weights = torch.tensor(w / (w * counts).sum() * counts.sum(), dtype=torch.float32, device=device)
    print("command counts:", {a: int(c) for a, c in zip(ACTIONS, counts)}, flush=True)
    if not args.no_init:
        res = evaluate(net, data["test"], device)
        print(f"start: command {res['command_acc']}  commanded {res['commanded_acc']}  selection IoU "
              f"{res['selection_iou']}  target {res['target_acc']}  dest near {res['dest_near']}", flush=True)
    for epoch in range(args.epochs):
        net.train()
        perm = np.random.permutation(n_train)
        total, nb = 0.0, 0
        for s in range(0, n_train, args.batch):
            inputs, hist, labels = batch(data["train"], perm[s:s + args.batch], device)
            loss = losses(net, inputs, hist, labels, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total, nb = total + loss.item(), nb + 1
        res = evaluate(net, data["test"], device)
        print(f"epoch {epoch + 1}: loss {total / nb:.3f}  command {res['command_acc']} (none {res['baseline_command_acc']})  "
              f"commanded {res['commanded_acc']} (most common {res['baseline_commanded_acc']})  "
              f"selection IoU {res['selection_iou']} (all own {res['baseline_selection_iou']})  "
              f"target {res['target_acc']} (nearest {res['baseline_target_acc']})  "
              f"dest near {res['dest_near']} (stay {res['baseline_dest_near']})  ({time.time() - t0:.0f} s)", flush=True)
    out = REPO_ROOT / "runs" / "fight_cmd" / f"{args.matchup}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    CommandModel(net.cpu(), config).save(out / "model.pt")
    report = {"matchup": args.matchup, "race": args.race, "train_examples": n_train, "epochs": args.epochs,
              "minutes": round((time.time() - t0) / 60, 1), "test": res}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
