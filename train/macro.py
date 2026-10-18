"""Train Gary's macro model (gary/policy/macro.py) on the macro training set.

Behavior cloning: for every few seconds of a player's game, predict that player's next production
decision and how long until it. One model per race. Games are split by replay (10% held out, never
seen in training), and the model is scored on the held-out games against a baseline that knows
only the game clock (the most common next decision at that minute).

Reported on held-out games:
  top-1 / top-3   the model's first (or any of its first three) guesses is the player's real next
                  decision; also for "non-routine" decisions only (everything except workers,
                  overlords and supply buildings, which are most of the volume)
  timing          median and mean error of the predicted seconds until that decision

Usage:
  python -m train.macro --race T                 # Terran in TvZ
  python -m train.macro --race Z --epochs 8
Output: runs/macro/<matchup>_<race>_<time>/model.pt and report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))
from inventory import data_root  # noqa: E402
from train.select import GameFilter  # noqa: E402

from gary.policy.macro import OTHER, FeatureSpec, MacroModel, MacroNet, encode  # noqa: E402

STRIDE = 3 * 24             # a sample every 3 s of game time
MIN_CLASS_COUNT = 100       # rarer decisions share the OTHER class
MAX_DELAY_S = 120           # "when" target is clipped here (long idle stretches at game end)
ROUTINE = {(0, 7), (1, 41), (0, 64), (1, 42), (3, 109), (3, 156)}  # workers, overlords, supply


def macro_dir() -> Path:
    return data_root() / "interim" / "macro" / "v1"


def is_test(sha1: str) -> bool:
    return int(sha1[:8], 16) % 10 == 0


def player_games(matchup: str, race: str, keep=None) -> list[tuple[str, int, int | None, bool]]:
    """(npz path, player index, style, held out) for every player of this race; keep: a GameFilter."""
    out = []
    with open(macro_dir() / "index.jsonl", encoding="utf-8") as f:
        rows = {r["sha1"]: r for r in map(json.loads, f)}
    for r in rows.values():
        if not r.get("ok") or r["matchup"] != matchup or (keep and not keep(r["sha1"])):
            continue
        path = str(macro_dir() / r["sha1"][:2] / f"{r['sha1']}.npz")
        for i, p in enumerate(r["players"]):
            if p["race"] == race:
                out.append((path, i, p["style"], is_test(r["sha1"])))
    return out


def scan(job: tuple) -> tuple[set, set, Counter]:
    path, i = job[0], job[1]
    d = np.load(path)
    own = set(np.nonzero(d["units"][:, i, :, 0].max(0))[0].tolist())
    seen = set(np.nonzero(d["seen"][:, i].max(0))[0].tolist())
    c = d["cmds"]
    c = c[c[:, 1] == i]
    return own, seen, Counter(zip(c[:, 2].tolist(), c[:, 3].tolist()))


def build_spec(race: str, jobs: list, workers: int) -> FeatureSpec:
    own, seen, counts = set(), set(), Counter()
    with ProcessPoolExecutor(workers) as pool:
        for o, s, c in pool.map(scan, [j for j in jobs if not j[3]], chunksize=32):
            own |= o
            seen |= s
            counts.update(c)
    decisions = sorted(d for d, n in counts.items() if n >= MIN_CLASS_COUNT) + [OTHER]
    styles = sorted({j[2] for j in jobs if j[2] is not None and j[2] >= 0})
    return FeatureSpec(race=race, own_types=sorted(own), seen_types=sorted(seen),
                       decisions=decisions, styles=styles)


def max_counts(job: tuple) -> np.ndarray:
    """(31, 228): the most of each own unit type this player had in each game minute."""
    d = np.load(job[0])
    units, frames = d["units"][:, job[1], :, 0], d["frames"]
    out = np.zeros((31, 228), dtype=np.float32)
    minute = np.minimum(frames // (24 * 60), 30)
    np.maximum.at(out, minute, units)
    out[minute.max() + 1:] = np.nan            # minutes after the game ended don't count
    return out


def count_caps(jobs: list, workers: int, q: float = 90) -> dict[int, list[int]]:
    """Plausibility caps: for each unit type and game minute, how many the pros had (q-th
    percentile over players). Gary won't make more of something than that: the model has never
    seen states beyond it and keeps asking for more (e.g. a 30th missile turret)."""
    with ProcessPoolExecutor(workers) as pool:
        per_player = np.stack(list(pool.map(max_counts, [j for j in jobs if not j[3]], chunksize=32)))
    caps = np.nanpercentile(per_player, q, axis=0)         # (31, 228)
    caps = np.fmax.accumulate(np.nan_to_num(caps, nan=0.0), axis=0)   # never lower later on
    return {t: [int(round(c)) for c in caps[:, t]] for t in range(228) if caps[:, t].max() > 0}


def samples(args: tuple) -> tuple[np.ndarray, ...]:
    """Features and targets for one player-game."""
    (path, i, style, _), spec_d = args
    spec = FeatureSpec(**spec_d)
    d = np.load(path)
    frames, cmds = d["frames"], d["cmds"]
    c = cmds[cmds[:, 1] == i]
    empty = (np.zeros((0, spec.size), np.float16), np.zeros(0, np.int16), np.zeros(0, np.float32),
             np.zeros(0, np.int8))
    if len(c) == 0:
        return empty
    cls = np.array([spec.decision_index(a, u) for a, u in c[:, 2:4].tolist()], dtype=np.int64)
    cum = np.zeros((len(c) + 1, len(spec.decisions)), dtype=np.float32)
    cum[np.arange(1, len(c) + 1), cls] = 1
    cum = np.cumsum(cum, axis=0)                     # cum[k] = counts of the first k decisions
    ts = np.arange(0, len(frames), STRIDE // 24)
    nxt = np.searchsorted(c[:, 0], frames[ts], side="right")   # first decision after the moment
    ts, nxt = ts[nxt < len(c)], nxt[nxt < len(c)]
    if len(ts) == 0:
        return empty
    f = frames[ts]
    last = np.where(nxt > 0, c[np.maximum(nxt - 1, 0), 0], 0)
    seen_max = np.maximum.accumulate(d["seen"][:, i].astype(np.float32), axis=0)[ts]
    x = encode(spec, f, d["eco"][ts, i].astype(np.float32), d["units"][ts, i].astype(np.float32),
               d["seen"][ts, i].astype(np.float32), seen_max, cum[nxt], (f - last) / 24.0, style)
    delay = np.minimum((c[nxt, 0] - f) / 24.0, MAX_DELAY_S).astype(np.float32)
    minute = np.minimum(f // (24 * 60), 30).astype(np.int8)
    return x.astype(np.float16), cls[nxt].astype(np.int16), delay, minute


def load_split(jobs: list, spec: FeatureSpec, workers: int) -> dict:
    parts = {False: [], True: []}
    with ProcessPoolExecutor(workers) as pool:
        for job, res in zip(jobs, pool.map(samples, [(j, spec.to_dict()) for j in jobs], chunksize=16)):
            parts[job[3]].append(res)
    out = {}
    for held, rs in parts.items():
        out["test" if held else "train"] = tuple(np.concatenate([r[k] for r in rs]) for k in range(4))
    return out


def evaluate(net: MacroNet, data: tuple, spec: FeatureSpec, prior: np.ndarray, device: str) -> dict:
    x, y, delay, minute = data
    net.eval()
    hits1, hits3, pred_s = [], [], []
    with torch.no_grad():
        for s in range(0, len(x), 16384):
            xb = torch.from_numpy(x[s:s + 16384].astype(np.float32)).to(device)
            logits, when = net(xb)
            t3 = logits.topk(3, -1).indices.cpu().numpy()
            yb = y[s:s + 16384]
            hits1.append(t3[:, 0] == yb)
            hits3.append((t3 == yb[:, None]).any(1))
            pred_s.append(torch.expm1(when).clamp(min=0).cpu().numpy())
    hits1, hits3, pred_s = np.concatenate(hits1), np.concatenate(hits3), np.concatenate(pred_s)
    routine = np.array([spec.decisions[k] in ROUTINE for k in y])
    base = prior[minute] == y
    err = np.abs(pred_s - delay)
    return {
        "samples": int(len(y)),
        "top1": round(float(hits1.mean()), 4), "top3": round(float(hits3.mean()), 4),
        "top1_nonroutine": round(float(hits1[~routine].mean()), 4),
        "top3_nonroutine": round(float(hits3[~routine].mean()), 4),
        "baseline_top1": round(float(base.mean()), 4),
        "baseline_top1_nonroutine": round(float(base[~routine].mean()), 4),
        "timing_median_err_s": round(float(np.median(err)), 2),
        "timing_mean_err_s": round(float(err.mean()), 2),
        "baseline_timing_median_err_s": round(float(np.median(np.abs(np.median(delay) - delay))), 2),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--race", required=True, choices=["T", "Z", "P"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--caps-only", metavar="RUN_DIR", help="only (re)compute caps.json for a trained run")
    GameFilter.add_args(ap)
    args = ap.parse_args()
    keep = GameFilter.from_args(args)
    if args.caps_only:
        caps = count_caps(player_games(args.matchup, args.race, keep), args.workers)
        (Path(args.caps_only) / "caps.json").write_text(json.dumps(caps), encoding="utf-8")
        print(f"wrote {Path(args.caps_only) / 'caps.json'} ({len(caps)} unit types)")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    jobs = player_games(args.matchup, args.race, keep)
    print(f"{len(jobs)} player-games ({sum(j[3] for j in jobs)} held out; {keep.describe()})", flush=True)
    spec = build_spec(args.race, jobs, args.workers)
    print(f"features {spec.size}, decision classes {len(spec.decisions)}, styles {len(spec.styles)}", flush=True)
    data = load_split(jobs, spec, args.workers)
    xtr, ytr, dtr, mtr = data["train"]
    print(f"samples: train {len(ytr):,}, test {len(data['test'][1]):,}  ({time.time() - t0:.0f} s)", flush=True)

    # baseline: the most common next decision at each game minute (training games)
    prior = np.zeros(31, dtype=np.int64)
    for m in range(31):
        sel = ytr[mtr == m]
        prior[m] = np.bincount(sel.astype(np.int64)).argmax() if len(sel) else prior[max(m - 1, 0)]

    config = {"net": {"hidden": args.hidden, "layers": args.layers, "dropout": 0.1},
              "matchup": args.matchup, "stride_s": STRIDE // 24, "max_delay_s": MAX_DELAY_S}
    net = MacroNet(spec.size, len(spec.decisions), **config["net"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(ytr) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    # the whole training set fits on the GPU (fp16), which keeps batches fast
    xt = torch.from_numpy(xtr).to(device)
    yt = torch.from_numpy(ytr.astype(np.int64)).to(device)
    dt = torch.from_numpy(dtr).to(device)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(yt), device=device)
        total = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            logits, when = net(xt[idx].float())
            loss = F.cross_entropy(logits, yt[idx]) + 0.5 * F.l1_loss(when, torch.log1p(dt[idx]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        res = evaluate(net, data["test"], spec, prior, device)
        print(f"epoch {epoch + 1}: loss {total / len(yt):.3f}  test top1 {res['top1']}  "
              f"top3 {res['top3']}  non-routine top1 {res['top1_nonroutine']}  "
              f"timing median err {res['timing_median_err_s']} s", flush=True)

    out = REPO_ROOT / "runs" / "macro" / f"{args.matchup}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    MacroModel(spec, net.cpu(), config).save(out / "model.pt")
    (out / "caps.json").write_text(json.dumps(count_caps(jobs, args.workers)), encoding="utf-8")
    report = {"matchup": args.matchup, "race": args.race, "player_games": len(jobs), "games": keep.describe(),
              "train_samples": int(len(ytr)), "features": spec.size, "classes": len(spec.decisions),
              "epochs": args.epochs, "minutes": round((time.time() - t0) / 60, 1), "test": res}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
