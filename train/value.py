"""Train Gary's value network (gary/policy/value.py; docs/CONTROL.md step 7): how good a moment is.

Reinforcement learning so far judged each decision by the raw score of what followed, and in games
this chaotic (half a second of Gary's hands moves the next 90 s by +-425 net value; a pair of
branches differs by sd 848) that buried every effect. A critic that predicts how a moment will go
lets each decision be judged by how much it changed the prediction, and lets a change to Gary be
judged from far fewer games.

Inputs: gary/policy/value.py (the macro model's view of the moment, the belief model's estimate of
the enemy army, the trends of the last 30 s). Heads and their data:

  win     will this player win: pro TvZ games (data/interim/build_orders.jsonl's winner) every
          STRIDE_S seconds, and Gary's decided games
  net60   the score's net value (eval/score.py: killed minus lost) over the next 60 s, /1000:
          Gary's games
  net180  ...over the next 180 s
  disc    the net value ahead, discounted (half-life DISC_HALF_S), by bootstrapping: the next
          GARY_EVERY_S seconds' net value plus the head's own value after (refreshed every REFRESH batches)

Gary's games: v0.19 with each build style forced in turn (and the macro unsteered), half of them
with the macro drawing its decisions (temperature 0.5) for variety, against the opponent pool (the
9-pool, Zerg v2, the replayed pros on three maps); its view every GARY_EVERY_S seconds.

    python -m train.value --gary-seeds 16                 # -> runs/value/TvZ_T_<stamp>/model.pt
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

from eval import monitor, priority
from gary.policy.value import GAMMA_S, LOOKBACK_S, ValueNet

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))
STRIDE_S = 10
GARY_EVERY_S = 5
REFRESH = 50                                   # batches between refreshes of the bootstrapped head's targets
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
OPPONENTS = ("dummy", "v2", "replayed")
STYLES = (None, 1, 2, 3, 4)
VERSIONS = ("v019", "v019+sample_t05")         # by seed: even, odd
MINUTES = 12


# --- pro games ---------------------------------------------------------------------------------

def pro_jobs(race: str = "T", matchup: str = "TvZ", keep=None) -> list[tuple]:
    from inventory import data_root
    won = {}
    with open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r.get("matchup") == matchup and r.get("race") == race and r.get("won") is not None:
                won[r["sha1"]] = bool(r["won"])
    root = data_root() / "interim" / "macro" / "v1"
    jobs = []
    with open(root / "index.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if (r.get("ok") and r.get("matchup") == matchup and r["sha1"] in won and len(r["players"]) == 2
                    and not (keep and not keep(r["sha1"]))):
                for i, p in enumerate(r["players"]):
                    if p["race"] == race:
                        jobs.append((str(root / r["sha1"][:2] / f"{r['sha1']}.npz"), i, won[r["sha1"]], r["sha1"]))
    return jobs


_belief: dict = {}


def _belief_model(race: str = "T"):
    if race not in _belief:
        from gary.agent.manifests import _newest
        from gary.policy.belief import BeliefModel
        torch.set_num_threads(1)
        _belief[race] = BeliefModel.load(_newest("belief", "TvZ", race))
    return _belief[race]


def pro_samples(args: tuple) -> tuple[np.ndarray, np.ndarray, str]:
    """The player's view (race T or Z) every STRIDE_S seconds of one pro game, and whether it won."""
    (path, i, won, sha1), spec_d, race = args
    enemy = {"T": "Z", "Z": "T"}[race]
    from gary.policy.belief import WORKERS, features_series
    from gary.policy.macro import FeatureSpec, encode
    from gary.policy.value import EXTRAS, army_supply, belief_outputs, extras, no_style
    spec = FeatureSpec(**spec_d)
    empty = np.zeros((0, spec.size + len(EXTRAS)), np.float16), np.zeros(0, np.float32), sha1
    try:
        d = np.load(path)
    except Exception:
        return empty
    frames, cmds = d["frames"], d["cmds"]
    ts = np.arange(0, len(frames), STRIDE_S)
    if len(ts) < 6:
        return empty
    c = cmds[cmds[:, 1] == i]
    cls = np.array([spec.decision_index(a, u) for a, u in c[:, 2:4].tolist()], dtype=np.int64) if len(c) else np.zeros(0, np.int64)
    cum = np.zeros((len(c) + 1, len(spec.decisions)), dtype=np.float32)
    if len(c):
        cum[np.arange(1, len(c) + 1), cls] = 1
    cum = np.cumsum(cum, axis=0)
    f = frames[ts]
    done = np.searchsorted(c[:, 0], f, side="right") if len(c) else np.zeros(len(ts), np.int64)
    last = np.where(done > 0, c[np.maximum(done - 1, 0), 0] if len(c) else 0, 0)
    seen = d["seen"][:, i].astype(np.float32)
    seen_max = np.maximum.accumulate(seen, axis=0)[ts]
    x = no_style(spec, encode(spec, f, d["eco"][ts, i].astype(np.float32), d["units"][ts, i].astype(np.float32),
                              seen[ts], seen_max, cum[done], (f - last) / 24.0, None))
    # the extras (each row of the arrays is a second)
    b = features_series(d["seen"][:, i], d["units"][:, i, :, 0], d["eco"][:, i], frames, enemy, race)[ts]
    own = army_supply(d["units"][:, i, :, 1].astype(np.float32), race)
    workers = d["units"][:, i, WORKERS[race], 1].astype(np.float32)
    seen_army = army_supply(seen, enemy)
    ago = np.maximum(ts - LOOKBACK_S, 0)
    seen_army_max = np.array([seen_army[a:t + 1].max() for a, t in zip(ago, ts)])
    ex = extras(belief_outputs(_belief_model(race), b), own[ts], own[ago], workers[ts] - workers[ago], seen_army[ts],
                seen_army_max)
    return np.concatenate([x, ex], 1).astype(np.float16), np.full(len(ts), float(won), np.float32), sha1


# --- Gary's games ------------------------------------------------------------------------------

def gary_game(args: tuple) -> dict:
    """Gary's view every GARY_EVERY_S seconds of one game, the score's events, and the result."""
    opponent, mp, seed, version, style = args
    from eval.block_branch import Watch, _step
    from eval.walls_bench import Match
    from gary.policy.value import ValueView
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 0, version, MAPS[mp], opponent, style=style)
    w = Watch(m)
    w.look()
    view = ValueView(m.gary, _belief_model())
    xs, frames = [], []
    end = MINUTES * 60 * 24
    while m.game.frame < end and _step(m, w):
        if m.game.frame % 24 < 2 and m.gary.macro is not None:
            try:
                obs = m.gary.hi.observe()
            except StopIteration:                              # Gary was eliminated
                break
            view.look(obs)
            if m.game.frame % (GARY_EVERY_S * 24) < 2:
                xs.append(view.features(obs)[0].astype(np.float16))
                frames.append(m.game.frame)
    players = {p["slot"] for p in m.game.observe()["players"]}
    result = "won" if m.slot["T"] in players and m.slot["Z"] not in players else \
        "lost" if m.slot["T"] not in players else "time"
    out = {"game": f"{opponent}/{mp}/{seed}/{version}/{style}", "x": np.array(xs), "frames": frames,
           "events": w.score.events, "result": result}
    m.game.close()
    return out


def gary_targets(g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per sample: win (1/0, nan if undecided); the net value over the next 60 and 180 s, and over the
    next GARY_EVERY_S s (the bootstrapped head's step) (/1000; nan where the game was cut off first)."""
    ev = np.array(g["events"], np.float64).reshape(-1, 2)
    fr = np.array(g["frames"])
    last = fr.max() if len(fr) else 0
    win = {"won": 1.0, "lost": 0.0}.get(g["result"], np.nan)
    nets = []
    for h in (60, 180, GARY_EVERY_S):
        e = fr + h * 24
        v = [ev[(ev[:, 0] >= a) & (ev[:, 0] < b), 1].sum() / 1000 if b <= last or g["result"] != "time" else np.nan
             for a, b in zip(fr, e)] if len(ev) else [0.0] * len(fr)
        nets.append(v)
    return np.full(len(fr), win, np.float32), np.array(nets, np.float32).T, fr


# --- training ----------------------------------------------------------------------------------

def auc(y, p):
    m = ~np.isnan(y)
    y, p = y[m], p[m]
    if len(set(y.tolist())) < 2:
        return float("nan")
    r = np.empty(len(p))
    r[np.argsort(p)] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gary-seeds", type=int, default=16)
    ap.add_argument("--pro-games", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--parallel", type=int, default=14)
    ap.add_argument("--reuse", help="a run folder whose collected data to train on again")
    ap.add_argument("--no-extras", action="store_true", help="ablation: the extras zeroed (the macro model's view alone)")
    ap.add_argument("--race", default="T", choices=["T", "Z"], help="whose value: the Zerg's from the pro games alone (its "
                    "win head: who is ahead; Gary-Z's own games, for the net heads, are not played here yet)")
    ap.add_argument("--matchup", default="TvZ")
    from train.select import GameFilter
    GameFilter.add_args(ap)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    keep = GameFilter.from_args(args)
    from gary.agent.manifests import _newest
    from gary.policy.macro import MacroModel
    spec = MacroModel.load(str(_newest("macro", args.matchup, args.race))).spec
    run = REPO_ROOT / "runs" / "value" / f"{args.matchup}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    src = Path(args.reuse) if args.reuse else run
    if not args.reuse:
        jobs = pro_jobs(args.race, args.matchup, keep)
        print(f"{len(jobs)} pro player-games ({keep.describe()})", flush=True)
        random.Random(0).shuffle(jobs)
        jobs = jobs[:args.pro_games]
        gjobs = [(o, mp, s, VERSIONS[s % 2], st) for o in OPPONENTS for mp in MAPS for s in range(1, args.gary_seeds + 1)
                 for st in STYLES] if args.race == "T" else []
        with ProcessPoolExecutor(args.parallel) as pool:
            pro = list(monitor.track(pool.map(pro_samples, [(j, spec.to_dict(), args.race) for j in jobs], chunksize=16), len(jobs),
                                     "pro games"))
            np.savez_compressed(src / "pro.npz", x=np.concatenate([p[0] for p in pro]), win=np.concatenate([p[1] for p in pro]),
                                game=np.concatenate([[p[2]] * len(p[1]) for p in pro]))
            gary = list(monitor.track(pool.map(gary_game, gjobs), len(gjobs), "Gary's games"))
        with open(src / "gary.jsonl", "w", encoding="utf-8") as f:
            for g in gary:
                f.write(json.dumps({**g, "x": g["x"].astype(np.float32).round(3).tolist()}) + "\n")
    pro = np.load(src / "pro.npz")
    gary = [g for g in (json.loads(l) for l in open(src / "gary.jsonl", encoding="utf-8")) if g["x"]]
    # held out: a tenth of the pro games (as the other models), Gary's games of every fifth seed
    # (all the styles of a seed together: they share the game's start)
    n0 = len(pro["win"])
    pro_test = np.array([int(s[:8], 16) % 10 == 0 for s in pro["game"]])
    gx, gwin, gnet, gtest, gnext, gend, gmin, gopp = [], [], [], [], [], [], [], []
    k = n0
    for g in gary:
        w, n, fr = gary_targets(g)
        gx.append(np.array(g["x"], np.float32))
        gwin.append(w)
        gnet.append(n)
        gtest.append(np.full(len(w), int(g["game"].split("/")[2]) % 5 == 0))
        nxt = np.arange(k + 1, k + len(w) + 1)             # the next sample of the same game, -1 after its last
        nxt[-1] = -1
        gnext.append(nxt)
        gend.append(np.full(len(w), g["result"] != "time"))
        gmin.append(fr / 1440)
        gopp.append(np.array([g["game"].split("/")[0]] * len(w), object))
        k += len(w)
    gnet = np.concatenate(gnet) if gnet else np.zeros((0, 3), np.float32)
    X = np.concatenate([pro["x"].astype(np.float32)] + gx)
    W = np.concatenate([pro["win"]] + gwin)
    N = np.concatenate([np.full((n0, 2), np.nan, np.float32), gnet[:, :2]])
    R = np.concatenate([np.full(n0, np.nan, np.float32), gnet[:, 2]])          # each step's net value
    NXT = np.concatenate([np.full(n0, -1)] + gnext)
    ENDED = np.concatenate([np.zeros(n0, bool)] + gend)
    test = np.concatenate([pro_test] + gtest)
    is_gary = np.concatenate([np.zeros(n0, bool), np.ones(len(X) - n0, bool)])
    t_min = np.concatenate([np.full(n0, np.nan)] + gmin)
    opp = np.concatenate([np.full(n0, "", object)] + gopp)
    if args.no_extras:
        from gary.policy.value import EXTRAS
        X[:, -len(EXTRAS):] = 0
    mean, std = X[~test].mean(0), X[~test].std(0) + 1e-3
    Xn = torch.tensor((X - mean) / std)
    Wt, Nt = torch.tensor(W), torch.tensor(N)
    net = ValueNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    tr = np.where(~test)[0]
    step = GAMMA_S ** GARY_EVERY_S
    print(f"samples: pro {n0} ({int(pro_test.sum())} held out), Gary {len(X) - n0} from {len(gary)} games "
          f"({int(test[n0:].sum())} held out); Gary's games won {np.mean([g['result'] == 'won' for g in gary]):.0%}, "
          f"lost {np.mean([g['result'] == 'lost' for g in gary]):.0%}", flush=True)
    gary_rows = np.arange(n0, len(X))

    def bootstrapped() -> torch.Tensor:
        """The bootstrapped head's targets from the net as it is now: r + gamma^step * disc(next);
        after a game's last sample nothing more if it ended, unknown (left out) at the time limit."""
        net.eval()
        with torch.no_grad():
            disc = np.full(len(X), np.nan, np.float32)
            if n0 < len(X):                        # (none of Gary's games: the pro win head alone, --race Z)
                disc[n0:] = torch.cat([net(Xn[i:i + 65536])[1][:, 2] for i in range(n0, len(X), 65536)]).numpy()
        net.train()
        boot = np.where(NXT >= 0, disc[np.maximum(NXT, 0)], np.where(ENDED, 0.0, np.nan))
        return torch.tensor((R + step * boot).astype(np.float32))

    for ep in monitor.track(range(args.epochs), args.epochs, "training"):
        np.random.default_rng(ep).shuffle(tr)
        for n, b in enumerate(range(0, len(tr), 1024)):
            if n % REFRESH == 0:                   # (each refresh looks GARY_EVERY_S further ahead)
                D = bootstrapped()
            i = torch.tensor(tr[b:b + 1024])
            lw, ln = net(Xn[i])
            wm, nm, dm = ~torch.isnan(Wt[i]), ~torch.isnan(Nt[i]), ~torch.isnan(D[i])
            loss = (F.binary_cross_entropy_with_logits(lw[wm], Wt[i][wm]) if wm.any() else 0.0) + \
                    (F.smooth_l1_loss(ln[:, :2][nm], Nt[i][nm]) if nm.any() else 0.0) + \
                    (F.smooth_l1_loss(ln[:, 2][dm], D[i][dm]) if dm.any() else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        lw, ln = net(Xn[torch.tensor(np.where(test)[0])])
    pw, pn = torch.sigmoid(lw).numpy(), ln.numpy()
    Wte, Nte, gte, tte, ote = W[test], N[test], is_gary[test], t_min[test], opp[test]
    print(f"held out: win AUC pro {auc(Wte[~gte], pw[~gte]):.3f}, Gary {auc(Wte[gte], pw[gte]):.3f}")
    for o in OPPONENTS:
        s = gte & (ote == o)
        print(f"  {o:8s} " + ", ".join(f"{lo}-{hi} min {auc(Wte[s & (tte >= lo) & (tte < hi)], pw[s & (tte >= lo) & (tte < hi)]):.2f}"
                                       for lo, hi in ((0, 4), (4, 7), (7, 99))))
    for j, h in enumerate((60, 180)):
        y, p = Nte[gte, j], pn[gte, j]
        m = ~np.isnan(y)
        r2 = 1 - np.mean((y[m] - p[m]) ** 2) / np.var(y[m]) if m.any() else float("nan")
        print(f"  net{h}: sd {np.std(y[m]) * 1000:.0f}, explained {r2:.1%} (the noise left: sd {np.sqrt(np.mean((y[m] - p[m]) ** 2)) * 1000:.0f})")
    if n0 == len(X):                                      # (the pro win head alone, --race Z: nothing of Gary's to judge)
        torch.save({"state": net.state_dict(), "n_in": X.shape[1], "mean": mean.tolist(), "std": std.tolist(),
                    "spec": spec.to_dict(), "heads": ["win", "net60", "net180", "disc"], "gamma_s": GAMMA_S}, run / "model.pt")
        print(f"-> {run / 'model.pt'}")
        return
    with torch.no_grad():
        dp = torch.cat([net(Xn[i:i + 65536])[1][:, 2] for i in range(n0, len(X), 65536)]).numpy()
    G = np.zeros(len(X) - n0)
    for k in range(len(X) - n0 - 1, -1, -1):                # the discounted score to the game's end
        nx = NXT[n0 + k]
        G[k] = (R[n0 + k] if np.isfinite(R[n0 + k]) else 0.0) + (step * G[nx - n0] if nx >= 0 else 0.0)
    left = np.array([t for g in gary for t in (np.array(g["frames"]).max() - np.array(g["frames"])) / 1440])
    far = test[n0:] & ((left >= 6) | ENDED[n0:])
    print(f"  disc: against the discounted score to the end (held out, 6+ min left or ended): "
          f"explained {1 - np.mean((G[far] - dp[far]) ** 2) / np.var(G[far]):.1%} (sd {np.std(G[far]) * 1000:.0f})")
    torch.save({"state": net.state_dict(), "n_in": X.shape[1], "mean": mean.tolist(), "std": std.tolist(),
                "spec": spec.to_dict(), "heads": ["win", "net60", "net180", "disc"], "gamma_s": GAMMA_S}, run / "model.pt")
    print(f"-> {run / 'model.pt'}")


if __name__ == "__main__":
    main()
