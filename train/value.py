"""Gary's value network (ARCHITECTURE.md P3, docs/CONTROL.md step 7): how good a moment is.

Reinforcement learning so far judged each decision by the raw score of what followed, and in games
this chaotic (half a second of Gary's hands moves the next 90 s by +-425 net value; a pair of
branches differs by sd 848) that buried every effect. A critic that predicts how a moment will go
lets each decision be judged by how much it changed the prediction, and lets a change to Gary be
judged from far fewer games.

Input: the macro model's description of the moment from the player's own view (gary/policy/macro.py
encode: time, money, supply, own units, the enemy in sight now and the most seen, the decisions so
far). Heads:

  win     will this player win: pro TvZ games (data/interim/build_orders.jsonl's winner) every
          STRIDE_S seconds, and Gary's decided games
  net60   the net value (eval/score.py: killed minus lost) over the next 60 s, /1000: Gary's games
  net180  ...over the next 180 s

Gary's games: Gary (v0.18, some with its macro drawing decisions for variety) against the opponent
pool (the 9-pool, Zerg v2, the replayed pros on three maps), its view every GARY_EVERY_S seconds.

    python -m train.value --gary-seeds 24                 # -> runs/value/TvZ_T_<stamp>/model.pt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eval import monitor, priority

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))
STRIDE_S = 10
GARY_EVERY_S = 5
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
OPPONENTS = ("dummy", "v2", "replayed")
VERSIONS = ("v018", "v018+sample_t05", "v018+watch")
MINUTES = 12


class ValueNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.win = nn.Linear(hidden, 1)
        self.net = nn.Linear(hidden, 2)          # net60, net180 (/1000)

    def forward(self, x):
        h = self.body(x)
        return self.win(h)[:, 0], self.net(h)


# --- pro games ---------------------------------------------------------------------------------

def pro_jobs() -> list[tuple]:
    from inventory import data_root
    won = {}
    with open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r.get("matchup") == "TvZ" and r.get("race") == "T" and r.get("won") is not None:
                won[r["sha1"]] = bool(r["won"])
    root = data_root() / "interim" / "macro" / "v1"
    jobs = []
    with open(root / "index.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r.get("ok") and r.get("matchup") == "TvZ" and r["sha1"] in won and len(r["players"]) == 2:
                for i, p in enumerate(r["players"]):
                    if p["race"] == "T":
                        jobs.append((str(root / r["sha1"][:2] / f"{r['sha1']}.npz"), i, won[r["sha1"]], r["sha1"]))
    return jobs


def pro_samples(args: tuple) -> tuple[np.ndarray, np.ndarray, str]:
    """The Terran's view every STRIDE_S seconds of one pro game, and whether it won."""
    (path, i, won, sha1), spec_d = args
    from gary.policy.macro import FeatureSpec, encode
    spec = FeatureSpec(**spec_d)
    try:
        d = np.load(path)
    except Exception:
        return np.zeros((0, spec.size), np.float16), np.zeros(0, np.float32), sha1
    frames, cmds = d["frames"], d["cmds"]
    ts = np.arange(0, len(frames), STRIDE_S)
    if len(ts) < 6:
        return np.zeros((0, spec.size), np.float16), np.zeros(0, np.float32), sha1
    c = cmds[cmds[:, 1] == i]
    cls = np.array([spec.decision_index(a, u) for a, u in c[:, 2:4].tolist()], dtype=np.int64) if len(c) else np.zeros(0, np.int64)
    cum = np.zeros((len(c) + 1, len(spec.decisions)), dtype=np.float32)
    if len(c):
        cum[np.arange(1, len(c) + 1), cls] = 1
    cum = np.cumsum(cum, axis=0)
    f = frames[ts]
    done = np.searchsorted(c[:, 0], f, side="right") if len(c) else np.zeros(len(ts), np.int64)
    last = np.where(done > 0, c[np.maximum(done - 1, 0), 0] if len(c) else 0, 0)
    seen_max = np.maximum.accumulate(d["seen"][:, i].astype(np.float32), axis=0)[ts]
    x = encode(spec, f, d["eco"][ts, i].astype(np.float32), d["units"][ts, i].astype(np.float32),
               d["seen"][ts, i].astype(np.float32), seen_max, cum[done], (f - last) / 24.0, None)
    return x.astype(np.float16), np.full(len(ts), float(won), np.float32), sha1


# --- Gary's games ------------------------------------------------------------------------------

def gary_game(args: tuple) -> dict:
    """Gary's view every GARY_EVERY_S seconds of one game, the score's events, and the result."""
    opponent, mp, seed, version = args
    from eval.block_branch import Watch, _step
    from eval.walls_bench import Match
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 0, version, MAPS[mp], opponent)
    w = Watch(m)
    w.look()
    xs, frames = [], []
    end = MINUTES * 60 * 24
    while m.game.frame < end and _step(m, w):
        if m.game.frame % (GARY_EVERY_S * 24) < 2 and m.gary.macro is not None:
            g = m.gary
            try:
                obs = g.hi.observe()
            except StopIteration:                              # Gary was eliminated
                break
            mine = [u for u in obs["units"] if u["owner"] == g.slot]
            import copy
            tracker = copy.copy(g.macro.tracker)               # (its seen_max is kept by the macro)
            tracker.seen_max = g.macro.tracker.seen_max.copy()
            xs.append(tracker.features(obs, g.macro.in_production(mine))[0].astype(np.float16))
            frames.append(m.game.frame)
    players = {p["slot"] for p in m.game.observe()["players"]}
    result = "won" if m.slot["T"] in players and m.slot["Z"] not in players else \
        "lost" if m.slot["T"] not in players else "time"
    out = {"game": f"{opponent}/{mp}/{seed}/{version}", "x": np.array(xs), "frames": frames,
           "events": w.score.events, "result": result}
    m.game.close()
    return out


def gary_targets(g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per sample: win (1/0, nan if undecided), and the net value over the next 60 and 180 s (/1000;
    nan where the game ended sooner)."""
    ev = np.array(g["events"], np.float64).reshape(-1, 2)
    fr = np.array(g["frames"])
    last = fr.max() if len(fr) else 0
    win = {"won": 1.0, "lost": 0.0}.get(g["result"], np.nan)
    nets = []
    for h in (60, 180):
        e = fr + h * 24
        v = [ev[(ev[:, 0] >= a) & (ev[:, 0] < b), 1].sum() / 1000 if b <= last or g["result"] != "time" else np.nan
             for a, b in zip(fr, e)] if len(ev) else [0.0] * len(fr)
        nets.append(v)
    return np.full(len(fr), win, np.float32), np.array(nets, np.float32).T, fr


# --- training ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gary-seeds", type=int, default=24)
    ap.add_argument("--pro-games", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=14)
    ap.add_argument("--reuse", help="a run folder whose collected data to train on again")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from gary.agent.manifests import model_path
    from gary.policy.macro import MacroModel
    spec = MacroModel.load(str(model_path("macro/latest"))).spec
    run = REPO_ROOT / "runs" / "value" / f"TvZ_T_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    src = Path(args.reuse) if args.reuse else run
    if not args.reuse:
        jobs = pro_jobs()
        random.Random(0).shuffle(jobs)
        jobs = jobs[:args.pro_games]
        gjobs = [(o, mp, s, v) for o in OPPONENTS for mp in MAPS for s in range(1, args.gary_seeds + 1) for v in VERSIONS]
        with ProcessPoolExecutor(args.parallel) as pool:
            pro = list(monitor.track(pool.map(pro_samples, [(j, spec.to_dict()) for j in jobs], chunksize=16), len(jobs),
                                     "pro games"))
            np.savez_compressed(src / "pro.npz", x=np.concatenate([p[0] for p in pro]), win=np.concatenate([p[1] for p in pro]),
                                game=np.concatenate([[p[2]] * len(p[1]) for p in pro]))
            gary = list(monitor.track(pool.map(gary_game, gjobs), len(gjobs), "Gary's games"))
        with open(src / "gary.jsonl", "w", encoding="utf-8") as f:
            for g in gary:
                f.write(json.dumps({**g, "x": g["x"].astype(np.float32).round(3).tolist()}) + "\n")
    pro = np.load(src / "pro.npz")
    gary = [json.loads(l) for l in open(src / "gary.jsonl", encoding="utf-8")]
    # held out: a tenth of the pro games (as the other models), every fifth of Gary's
    pro_test = np.array([int(s[:8], 16) % 10 == 0 for s in pro["game"]])
    gx, gwin, gnet, ggame = [], [], [], []
    for k, g in enumerate(gary):
        if not g["x"]:
            continue
        w, n, _ = gary_targets(g)
        gx.append(np.array(g["x"], np.float32))
        gwin.append(w)
        gnet.append(n)
        ggame += [k % 5 == 0] * len(w)
    gx, gwin, gnet, gtest = np.concatenate(gx), np.concatenate(gwin), np.concatenate(gnet), np.array(ggame)
    X = np.concatenate([pro["x"].astype(np.float32), gx])
    W = np.concatenate([pro["win"], gwin])
    N = np.concatenate([np.full((len(pro["win"]), 2), np.nan, np.float32), gnet])
    test = np.concatenate([pro_test, gtest])
    is_gary = np.concatenate([np.zeros(len(pro["win"]), bool), np.ones(len(gwin), bool)])
    mean, std = X[~test].mean(0), X[~test].std(0) + 1e-3
    Xn = torch.tensor((X - mean) / std)
    Wt, Nt = torch.tensor(W), torch.tensor(N)
    net = ValueNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    tr = np.where(~test)[0]
    print(f"samples: pro {len(pro['win'])} ({int(pro_test.sum())} held out), Gary {len(gwin)} from {len(gary)} games "
          f"({int(gtest.sum())} held out); Gary's games won {np.mean([g['result'] == 'won' for g in gary]):.0%}, "
          f"lost {np.mean([g['result'] == 'lost' for g in gary]):.0%}", flush=True)
    for ep in range(args.epochs):
        np.random.default_rng(ep).shuffle(tr)
        for b in range(0, len(tr), 1024):
            i = torch.tensor(tr[b:b + 1024])
            lw, ln = net(Xn[i])
            wm, nm = ~torch.isnan(Wt[i]), ~torch.isnan(Nt[i])
            loss = (F.binary_cross_entropy_with_logits(lw[wm], Wt[i][wm]) if wm.any() else 0.0) + \
                (F.smooth_l1_loss(ln[nm], Nt[i][nm]) if nm.any() else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        lw, ln = net(Xn[torch.tensor(np.where(test)[0])])
    pw, pn = torch.sigmoid(lw).numpy(), ln.numpy()
    Wte, Nte, gte = W[test], N[test], is_gary[test]

    def auc(y, p):
        m = ~np.isnan(y)
        y, p = y[m], p[m]
        if len(set(y.tolist())) < 2:
            return float("nan")
        order = np.argsort(p)
        r = np.empty(len(p))
        r[order] = np.arange(1, len(p) + 1)
        n1 = y.sum()
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))

    print(f"held out: win AUC pro {auc(Wte[~gte], pw[~gte]):.3f}, Gary {auc(Wte[gte], pw[gte]):.3f}")
    for j, h in enumerate((60, 180)):
        y, p = Nte[gte, j], pn[gte, j]
        m = ~np.isnan(y)
        r2 = 1 - np.mean((y[m] - p[m]) ** 2) / np.var(y[m]) if m.any() else float("nan")
        print(f"  net{h}: sd {np.std(y[m]) * 1000:.0f}, explained {r2:.1%} (the noise left: sd {np.sqrt(np.mean((y[m] - p[m]) ** 2)) * 1000:.0f})")
    torch.save({"state": net.state_dict(), "n_in": X.shape[1], "mean": mean.tolist(), "std": std.tolist(),
                "spec": spec.to_dict()}, run / "model.pt")
    print(f"-> {run / 'model.pt'}")


if __name__ == "__main__":
    main()
