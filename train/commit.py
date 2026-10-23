"""The army's commitments, learned from the pros: every 5 s, each army cluster's next step: keep (no order
that moves it 2+ cells), engage (an order sending it into the enemy) or move (any other order of 2+ cells),
given the map, the numbers, and the cluster's commitment (its last order: engage or move, and its age).
The unit of the army's decision is the commitment, held until the state calls for another, not a draw
every few seconds: e1's paired branches showed a single draw undone 30 s later in 85-94% of pairs, both
arms converging, nothing to credit (docs/PROGRESS.md). Pros re-decide on events: the keep class is
learned from the state, so it learns the triggers (a base attacked, an army sighted); and with the
commitment as an input, the hysteresis (pros attacking keep attacking).

The model: the imitation army model, its move/attack head relearned as keep/engage/move, the numbers
widened by the commitment (new weights from zero), the destination head trained on the orders among the
samples, the rest fine-tuned at a tenth of the rate. Data: ingest/army_dataset.py v4 (its "tick" samples).
Games split by replay as the army model's. Scored on held-out games:

  auc_order       an order (engage or move) vs keep
  rate            orders per cluster-minute, predicted vs the pros', by game minute
  engage_auc      engage vs move, among orders
  hysteresis      P(engage), committed to engaging vs to moving: the model's and the pros'
  where_top5      the order's cell among the destination head's first five

    python -m train.commit --race T
Output: runs/commit/TvZ_<race>_<time>/model.pt (an army model: spec.commit, kind head keep/engage/move)
and report.json
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
sys.path.insert(0, str(REPO / "ingest"))
from inventory import data_root  # noqa: E402

from eval import monitor, priority  # noqa: E402
from gary.policy.army import (ARMY_SUPPLY, GRID, MEMORY_S, OTHER_INPUTS, ArmyModel, ArmyNet, ArmySpec,  # noqa: E402
                              commit_inputs, encode_global)
from train.macro import is_test  # noqa: E402
from train.select import GameFilter  # noqa: E402

MAX_TICKS = 400                 # samples per player-game in training (held-out games: all)
KEEP, ENGAGE, MOVE = 0, 1, 2


def jobs(matchup: str, race: str, version: str = "v4", keep=None) -> list[tuple]:
    """(army npz, player index, held out, macro npz, the player's index there)."""
    army = data_root() / "interim" / "army" / version
    macro_dir = data_root() / "interim" / "macro" / "v1"
    with open(macro_dir / "index.jsonl", encoding="utf-8") as f:
        macro = {r["sha1"]: r for r in map(json.loads, f) if r.get("ok")}
    out = []
    with open(army / "index.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if not r.get("ok") or r["matchup"] != matchup or r["sha1"] not in macro or (keep and not keep(r["sha1"])):
                continue
            mslots = [p["slot"] for p in macro[r["sha1"]]["players"]]
            for i, p in enumerate(r["players"]):
                if p["race"] == race and p.get("ticks", 0) > 0 and p["slot"] in mslots:
                    out.append((str(army / r["sha1"][:2] / f"{r['sha1']}.npz"), i, is_test(r["sha1"]),
                                str(macro_dir / r["sha1"][:2] / f"{r['sha1']}.npz"), mslots.index(p["slot"])))
    return out


def _others(d, i: int) -> np.ndarray:
    """others_inputs for every tick sample of player i: the clusters sampled at the same frame are the others."""
    from gary.policy.army import others_inputs
    frames, grid, commit = d[f"p{i}_tick_frame"], d[f"p{i}_tick_grid"], d[f"p{i}_tick_commit"]
    out = np.zeros((len(frames), OTHER_INPUTS), np.float32)
    start = 0
    while start < len(frames):
        end = start
        while end < len(frames) and frames[end] == frames[start]:
            end += 1
        ks = list(range(start, end))
        centers = [int(grid[k, 8].argmax()) for k in ks]
        sups = [float(grid[k, 8].sum()) for k in ks]
        army = float(grid[start, 0].sum())
        out[start:end] = others_inputs(centers, sups, [int(commit[k]) for k in ks], army)
        start = end
    return out


def load_player(args: tuple) -> tuple:
    (path, i, held, mpath, mi), spec_d = args
    spec = ArmySpec(**spec_d)
    d = np.load(path)
    n = len(d[f"p{i}_tick_label"])
    others = _others(d, i) if spec.others else None       # (before sampling: a frame's clusters together)
    rng = np.random.default_rng(hash(path) % 2**32 + i)
    idx = np.sort(rng.choice(n, MAX_TICKS, replace=False)) if n > MAX_TICKS and not held else np.arange(n)
    frames = d[f"p{i}_tick_frame"][idx]
    m = np.load(mpath)
    mf, seen_all = m["frames"], m["seen"][:, mi].astype(np.float32)
    own_sup = m["units"][:, mi, :, 0].astype(np.float32) @ ARMY_SUPPLY
    t = np.clip(np.searchsorted(mf, frames, side="right") - 1, 0, len(mf) - 1)
    seen_mem = np.stack([seen_all[max(0, k - MEMORY_S):k + 1].max(0) for k in t]) if len(t) else np.zeros((0, 228), np.float32)
    grid = d[f"p{i}_tick_grid"][idx]
    if spec.losses:                                     # (the 10th channel: own losses lately, v3)
        grid = np.concatenate([grid, d[f"p{i}_tick_loss"][idx][:, None]], 1)
    if spec.static:                                     # (then the enemy's static defense known, v4)
        grid = np.concatenate([grid, d[f"p{i}_tick_static"][idx][:, None]], 1)
    commit = d[f"p{i}_tick_commit"][idx]
    glob = encode_global(spec, frames.astype(np.float32), d[f"p{i}_tick_eco"][idx].astype(np.float32),
                         grid[:, 8].sum(1).astype(np.float32), d[f"p{i}_tick_units"][idx].astype(np.float32),
                         d[f"p{i}_tick_seen"][idx].astype(np.float32), seen_mem, own_sup[t], seen_mem @ ARMY_SUPPLY,
                         commit_inputs(commit, d[f"p{i}_tick_age"][idx]), others[idx] if others is not None else None)
    return (grid, glob.astype(np.float16), d[f"p{i}_tick_label"][idx].astype(np.int64),
            d[f"p{i}_tick_target"][idx].astype(np.int64), commit.astype(np.int64), frames)


def load(js: list, spec: ArmySpec, workers: int) -> dict:
    parts = {False: [], True: []}
    with ProcessPoolExecutor(workers) as pool:
        results = pool.map(load_player, [(j, spec.to_dict()) for j in js], chunksize=16)
        for job, res in zip(js, monitor.track(results, len(js), "player-games loaded")):
            parts[job[2]].append(res)
    return {("test" if held else "train"): tuple(np.concatenate([r[k] for r in rs]) for k in range(6))
            for held, rs in parts.items()}


def widen(base: ArmyModel, spec: ArmySpec) -> ArmyNet:
    """The army model's network with the numbers widened by the commitment and the map by the losses (and
    static defense) channels (new weights zero: it starts as the army model) and a fresh keep/engage/move head."""
    old = base.net
    ch = base.config["net"].get("ch", 64)
    net = ArmyNet(spec.n_global, ch=ch, n_kind=3, channels=spec.channels)
    state = {k: v for k, v in old.state_dict().items() if not k.startswith("kind.")}
    w = state["glob.0.weight"]
    state["glob.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], spec.n_global - w.shape[1])], 1)
    w = state["inp.weight"]
    state["inp.weight"] = torch.cat([w, torch.zeros(w.shape[0], spec.channels - w.shape[1], *w.shape[2:])], 1)
    net.load_state_dict(state, strict=False)
    return net


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--race", default="T", choices=["T", "Z", "P"])
    ap.add_argument("--epochs", type=int, default=4)
    GameFilter.add_args(ap)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--others", action="store_true", help="the other clusters' commitments as inputs (ArmySpec.others)")
    ap.add_argument("--static", action="store_true",
                    help="the enemy's ground static defense as a map channel (ArmySpec.static; army dataset v4)")
    ap.add_argument("--eval", help="score this commit model on the held-out games and stop (no training)")
    ap.add_argument("--out", help="the run folder (default runs/commit/<matchup>_<race>_<time>: what Gary loads as newest)")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from sklearn.metrics import roc_auc_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = sorted((REPO / "runs" / "army").glob(f"{args.matchup}_{args.race}_*/model.pt"))[-1]
    base = ArmyModel.load(start)
    spec = ArmySpec(base.spec.race, base.spec.own_types, base.spec.seen_types, commit=True, losses=True, others=args.others,
                    static=args.static)
    if args.eval:
        evaluated = ArmyModel.load(args.eval)
        spec = evaluated.spec
    t0 = time.time()
    keep = GameFilter.from_args(args)
    js = jobs(args.matchup, args.race, keep=keep)
    print(f"{len(js)} player-games ({keep.describe()})", flush=True)
    data = load(js, spec, args.workers)
    grid, glob, label, target, commit, _ = data["train"]
    tg, tglob, tl, tt, tc, tf = data["test"]
    print(f"{len(js)} player-games; ticks train {len(label):,} (keep/engage/move {np.bincount(label, minlength=3)}), "
          f"test {len(tl):,} ({time.time() - t0:.0f} s)", flush=True)
    net = (evaluated.net if args.eval else widen(base, spec)).to(device)
    head = list(net.kind.parameters()) + list(net.glob.parameters())
    trunk = [p for n, p in net.named_parameters() if not n.startswith(("kind.", "glob."))]
    opt = torch.optim.AdamW([{"params": head, "lr": args.lr}, {"params": trunk, "lr": args.lr / 10}], weight_decay=1e-4)
    G, X, Y, TG = map(torch.from_numpy, (grid, glob, label, target))

    def predict(g, x):
        net.eval()
        ks, ws = [], []
        with torch.no_grad():
            for s in range(0, len(g), 4096):
                w, k, _ = net(torch.from_numpy(g[s:s + 4096]).to(device).float().view(-1, spec.channels, GRID, GRID),
                              torch.from_numpy(x[s:s + 4096]).to(device).float())
                ks.append(torch.softmax(k, -1).cpu().numpy())
                ws.append(torch.topk(w, 5, -1).indices.cpu().numpy())
        net.train()
        return np.concatenate(ks), np.concatenate(ws)

    def evaluate() -> dict:
        p, w5 = predict(tg, tglob)
        order = tl > KEEP
        out = {"auc_order": round(float(roc_auc_score(order, 1 - p[:, KEEP])), 3)}
        o = order
        if o.any() and (tl[o] == ENGAGE).any() and (tl[o] == MOVE).any():
            pe = p[o, ENGAGE] / (p[o, ENGAGE] + p[o, MOVE])
            out["engage_auc"] = round(float(roc_auc_score(tl[o] == ENGAGE, pe)), 3)
        minute = tf // 1440
        out["rate"] = {int(m): [round(float(order[minute == m].mean() * 12), 2), round(float((1 - p[minute == m, KEEP]).mean() * 12), 2)]
                       for m in range(4, 16) if (minute == m).sum() > 200}
        out["hysteresis"] = {name: [round(float((tl[tc == c] == ENGAGE).mean()), 3), round(float(p[tc == c, ENGAGE].mean()), 3)]
                             for c, name in ((1, "committed engage"), (2, "committed move"), (0, "none")) if (tc == c).any()}
        out["where_top5"] = round(float((w5[o] == tt[o, None]).any(1).mean()), 3) if o.any() else None
        return out
    print("before:", evaluate(), flush=True)
    if args.eval:
        return
    for epoch in monitor.track(range(args.epochs), args.epochs, "epochs"):
        perm = torch.randperm(len(Y))
        total = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            w, k, _ = net(G[idx].to(device).float().view(-1, spec.channels, GRID, GRID), X[idx].to(device).float())
            y = Y[idx].to(device)
            loss = F.cross_entropy(k, y)
            o = y > KEEP
            if o.any():                                     # (the destination, on the orders among them)
                loss = loss + 0.5 * F.cross_entropy(w[o], TG[idx].to(device)[o])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}: loss {total / len(Y):.4f}  test {evaluate()}", flush=True)
    res = evaluate()
    out = Path(args.out) if args.out else REPO / "runs" / "commit" / f"{args.matchup}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    model = ArmyModel(spec, net.cpu().eval(), {**base.config, "net": {**base.config["net"], "n_kind": 3,
                                                                        "channels": spec.channels},
                                               "commit": {"from": str(start), "classes": ["keep", "engage", "move"]}})
    model.save(out / "model.pt")
    (out / "report.json").write_text(json.dumps({"from": str(start), "ticks_train": int(len(Y)),
                                                  "ticks_test": int(len(tl)), "test": res}, indent=1), encoding="utf-8")
    print(f"saved {out / 'model.pt'}: {res}")


if __name__ == "__main__":
    main()
