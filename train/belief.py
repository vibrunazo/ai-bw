"""Train Gary's belief model: what the enemy has, from what the player has seen of it.

The army model's fight estimate only knows the enemy units Gary has seen lately, so it takes the
fog for empty: on Python, 8 marines walked out into 14 zerglings nobody had seen yet (a fight
estimate of +1%). Pros know a hatch first at 5:30 has zerglings, seen or not. This model learns
that from pro games: from the player's view of the enemy so far (fog of war), the enemy's real
army.

Data: the macro training set (ingest/macro_dataset.py), which has, every second of each game,
both players' real units and what each could see of the other. A sample every SAMPLE_EVERY
seconds, from the point of view of a player of --race against the other race:

  inputs   game time; for each enemy unit and building type: how many are in sight now, the most
           in sight at once over the last 30 s, 90 s, 3 min and ever; seconds since an enemy army
           unit was last in sight; the player's own supply and workers
  targets  the enemy's real counts of each army type, workers and town halls (a Poisson rate each,
           so the expected count is unbiased), and its army supply's median and 80th percentile
           (QUANTILES): the cautious estimate a fight decision wants

Games are split by replay (the same 10% held out as the other models). Scored on the held-out
games: the error in the enemy's army supply, and how often the enemy has 8+ supply more army than
believed (the danger: walking out into it), against two baselines: what was in sight over the last
3 minutes (most at once), and the average for that game time.

    python -m train.belief --race T
Output: runs/belief/<matchup>_<race>_<time>/model.pt and report.json
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
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))
sys.path.insert(0, str(REPO_ROOT))
from inventory import data_root  # noqa: E402

SAMPLE_EVERY = 10                       # seconds between samples
START, END = 60, 20 * 60                # seconds of game time sampled
WINDOWS = (30, 90, 180)                 # "most in sight at once" over these many seconds
RACE_ID = {"T": "T", "Z": "Z", "P": "P"}
# the enemy's types the inputs cover (by race), and the army types the targets count, with
# supply x2 (BW counts half supply)
ENEMY_TYPES = {
    "Z": [37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 62, 103, 131, 132, 133, 135, 136, 137, 138, 139, 140,
          141, 142, 143, 144, 146, 149],
    "T": [0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 30, 32, 34, 58, 106, 107, 109, 110, 111, 112, 113, 114, 115,
          116, 117, 120, 122, 123, 124, 125],
    "P": [61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 83, 84, 154, 155, 156, 157, 159, 160, 162, 163, 164,
          165, 166, 167, 169, 170, 171, 172],
}
ARMY_SUPPLY = {  # supply x2
    "Z": {37: 1, 38: 2, 39: 8, 43: 4, 44: 4, 45: 4, 46: 4, 47: 1, 62: 4, 103: 4},
    "T": {0: 2, 1: 2, 2: 4, 3: 4, 5: 4, 30: 4, 8: 4, 9: 4, 11: 4, 12: 12, 32: 2, 34: 2, 58: 4},
    "P": {65: 4, 66: 4, 61: 4, 67: 8, 68: 8, 63: 4, 70: 6, 71: 4, 72: 12, 83: 8, 60: 4},
}
WORKERS = {"Z": 41, "T": 7, "P": 64}
TOWN_HALLS = {"Z": (131, 132, 133), "T": (106,), "P": (154,)}
DANGER = 16                             # supply x2: the enemy has this much more army than believed
QUANTILES = (0.5, 0.8)                  # army supply quantiles the model also predicts
SUP_SCALE = 100.0                       # army supply x2 is predicted divided by this


def held_out(sha1: str) -> bool:
    return int(sha1[:8], 16) % 10 == 0


def targets_of(units: np.ndarray, enemy: str) -> np.ndarray:
    """(T, K) the enemy's real counts: its army types, workers, town halls."""
    army = list(ARMY_SUPPLY[enemy])
    cols = [units[:, t] for t in army] + [units[:, WORKERS[enemy]], units[:, list(TOWN_HALLS[enemy])].sum(1)]
    return np.stack(cols, 1).astype(np.float32)


def features(seen: np.ndarray, own_units: np.ndarray, eco: np.ndarray, frames: np.ndarray,
             enemy: str, own: str) -> np.ndarray:
    """(T, F) inputs for every second of one player's game: seen (T, 228) the enemy in sight each
    second; own_units (T, 228); eco (T, 4)."""
    types = ENEMY_TYPES[enemy]
    s = seen[:, types].astype(np.float32)                    # (T, k)
    T = len(s)
    cols = [frames[:, None] / (24 * 600.0), np.log1p(s)]
    for w in WINDOWS:
        m = np.zeros_like(s)
        for t in range(T):
            m[t] = s[max(0, t - w + 1):t + 1].max(0)
        cols.append(np.log1p(m))
    cols.append(np.log1p(np.maximum.accumulate(s, axis=0)))
    army = np.array([t in ARMY_SUPPLY[enemy] for t in types])
    any_army = s[:, army].sum(1) > 0
    since = np.zeros(T, np.float32)
    last = -10 ** 6
    for t in range(T):
        if any_army[t]:
            last = t
        since[t] = min(t - last, 600)
    cols.append(since[:, None] / 600.0)
    cols.append(eco[:, 2:3] / 400.0)                         # own supply used x2
    cols.append(own_units[:, WORKERS[own]:WORKERS[own] + 1].astype(np.float32) / 80.0)
    return np.concatenate(cols, 1).astype(np.float32)


def load_game(args: tuple) -> tuple | None:
    path, i, enemy, own, out = args
    try:
        d = np.load(path)
    except Exception:
        return None
    frames = d["frames"]
    if len(frames) < START + SAMPLE_EVERY:
        return None
    X = features(d["seen"][:, i], d["units"][:, i, :, 0], d["eco"][:, i], frames, enemy, own)
    Y = targets_of(d["units"][:, 1 - i, :, 0], enemy)
    # baseline: most in sight at once over the last 3 minutes, as army supply
    sup = np.array([ARMY_SUPPLY[enemy][t] for t in ARMY_SUPPLY[enemy]], np.float32)
    seen_army = d["seen"][:, i][:, list(ARMY_SUPPLY[enemy])].astype(np.float32)
    recent = np.stack([seen_army[max(0, t - 179):t + 1].max(0) for t in range(len(frames))]) @ sup
    idx = np.arange(START, min(len(frames), END), SAMPLE_EVERY)
    return X[idx], Y[idx], recent[idx], frames[idx], out


def jobs_for(race: str) -> list[tuple]:
    enemy = {"T": "Z", "Z": "T"}.get(race, "Z")
    matchup = "TvZ" if {race, enemy} == {"T", "Z"} else None
    jobs = []
    root = data_root() / "interim" / "macro" / "v1"
    with open(root / "index.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if not r.get("ok") or r.get("matchup") != matchup or len(r.get("players", [])) != 2:
                continue
            for i, p in enumerate(r["players"]):
                if p["race"] == race and r["players"][1 - i]["race"] == enemy:
                    jobs.append((str(root / r["sha1"][:2] / f"{r['sha1']}.npz"), i, enemy, race, held_out(r["sha1"])))
    return jobs


class BeliefNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def army_supply(counts: np.ndarray, enemy: str) -> np.ndarray:
    sup = np.array(list(ARMY_SUPPLY[enemy].values()), np.float32)
    return counts[:, :len(sup)] @ sup


def score(name: str, pred_sup: np.ndarray, true_sup: np.ndarray, t: np.ndarray) -> dict:
    """Errors in army supply (x2 -> real supply), overall and by game time."""
    out = {}
    for lo, hi in ((0, 10 ** 9), (3, 6), (6, 10), (10, 15)):
        m = (t >= lo * 60 * 24) & (t < hi * 60 * 24)
        if not m.any():
            continue
        err = (pred_sup[m] - true_sup[m]) / 2
        key = "all" if hi == 10 ** 9 else f"{lo}-{hi} min"
        out[key] = {"mae": round(float(np.abs(err).mean()), 2), "bias": round(float(err.mean()), 2),
                    "danger": round(float((true_sup[m] - pred_sup[m] >= DANGER).mean()), 3), "n": int(m.sum())}
    print(f"  {name:28s} " + "  ".join(f"{k}: mae {v['mae']:5.2f} bias {v['bias']:+6.2f} danger {v['danger']:.1%}"
                                         for k, v in out.items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", default="T", choices=["T", "Z"])
    ap.add_argument("--games", type=int, help="at most this many player-games (quick runs)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    from eval import priority
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    jobs = jobs_for(args.race)[:args.games]
    enemy = jobs[0][2]
    print(f"{len(jobs)} player-games ({sum(j[4] for j in jobs)} held out)", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(load_game, jobs, chunksize=16) if r is not None]
    tr = [r for r in rows if not r[4]]
    te = [r for r in rows if r[4]]
    cat = lambda rs, k: np.concatenate([r[k] for r in rs])
    Xtr, Ytr, Xte, Yte = cat(tr, 0), cat(tr, 1), cat(te, 0), cat(te, 1)
    ftr, fte, recent_te = cat(tr, 3), cat(te, 3), cat(te, 2)
    print(f"samples: {len(Xtr):,} train, {len(Xte):,} held out; {Xtr.shape[1]} inputs, {Ytr.shape[1]} targets", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = BeliefNet(Xtr.shape[1], Ytr.shape[1] + len(QUANTILES)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    st = torch.tensor(army_supply(Ytr, enemy) / SUP_SCALE, device=dev)
    q = torch.tensor(QUANTILES, device=dev)
    K = Ytr.shape[1]
    for ep in range(args.epochs):
        perm = torch.randperm(len(xt), device=dev)
        tot = 0.0
        for b in range(0, len(xt), 2048):
            i = perm[b:b + 2048]
            out = net(xt[i])
            counts = nn.functional.poisson_nll_loss(out[:, :K], yt[i], log_input=True)
            diff = st[i, None] - out[:, K:]                      # pinball loss for the quantiles
            loss = counts + torch.maximum(q * diff, (q - 1) * diff).mean() * 10
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
        print(f"epoch {ep + 1}: loss {tot / len(xt):.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(Xte, device=dev)).cpu().numpy()
    pred, quant = np.exp(out[:, :K]), out[:, K:] * SUP_SCALE

    true_sup = army_supply(Yte, enemy)
    # time-only baseline: the training set's mean army supply in each 30 s bucket
    tr_sup = army_supply(Ytr, enemy)
    bucket = lambda f: np.minimum(f // (24 * 30), 60)
    means = np.zeros(61, np.float32)
    for b in range(61):
        m = bucket(ftr) == b
        means[b] = tr_sup[m].mean() if m.any() else 0
    report = {"race": args.race, "enemy": enemy, "games": len(rows), "train": len(Xtr), "held_out": len(Xte),
              "enemy army supply (held out)": {}}
    print("enemy army supply, held-out games (supply; bias > 0: believed more than there was; "
          "danger: the enemy has 8+ supply more than believed):")
    r = report["enemy army supply (held out)"]
    r["seen in the last 3 min"] = score("seen in the last 3 min", recent_te, true_sup, fte)
    r["average for the time"] = score("average for the game time", means[bucket(fte)], true_sup, fte)
    r["belief model"] = score("belief model (expected)", army_supply(pred, enemy), true_sup, fte)
    for j, qq in enumerate(QUANTILES):
        r[f"belief model, {int(qq * 100)}th percentile"] = score(f"belief model ({int(qq * 100)}th pct)", quant[:, j], true_sup, fte)

    out = REPO_ROOT / "runs" / "belief" / f"Tv{enemy}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "n_in": Xtr.shape[1], "n_out": Ytr.shape[1] + len(QUANTILES),
                "quantiles": QUANTILES, "sup_scale": SUP_SCALE, "race": args.race,
                "enemy": enemy, "army_types": list(ARMY_SUPPLY[enemy]), "enemy_types": ENEMY_TYPES[enemy],
                "windows": WINDOWS}, out / "model.pt")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
