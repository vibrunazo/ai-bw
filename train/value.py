"""Train Gary's value network (gary/policy/value.py; docs/CONTROL.md step 7): how good a moment is.

Reinforcement learning so far judged each decision by the raw score of what followed, and in games
this chaotic (half a second of Gary's hands moves the next 90 s by +-425 net value; a pair of
branches differs by sd 848) that buried every effect. A critic that predicts how a moment will go
lets each decision be judged by how much it changed the prediction, and lets a change to Gary be
judged from far fewer games.

Inputs: gary/policy/value.py (the macro model's view of the moment, the belief model's estimate of
the enemy army, the trends of the last 30 s). Heads and their data:

  win     will this player win: the matchup's pro games (data/interim/build_orders.jsonl's winner) every
          STRIDE_S seconds, and Gary's decided games
  net60   the score's net value (eval/score.py: killed minus lost) over the next 60 s, /1000:
          Gary's games
  net180  ...over the next 180 s
  disc    the net value ahead, discounted (half-life DISC_HALF_S), by bootstrapping: the next
          GARY_EVERY_S seconds' net value plus the head's own value after (refreshed every REFRESH batches)

Gary's games: the league's of the matchup (runs/league/<round>, the Garys of every version against each other; in a
mirror both sides), each replay
re-simulated as the pros' are (ingest/macro_dataset.py), so both read the same features; its view every
GARY_EVERY_S seconds, the net value from the units' deaths (a unit's value, Terran +, as eval/pairs.py
scores its arms, + for the player). The win head learns from the decided games; games at the time limit teach the net heads.
Held out: the pros' games by hash, Gary's by seed (every fifth).

The critic reads the moment, not what the player meant or knows of the pros: the build style, the map, the year
and the matchup are left at "unknown" and the decisions so far are the player's own (in play: ValueView, not the
pros' history the macro model may see).

    python -m train.value --rounds "BL121?" "WS121?" --race T     # -> runs/value/TvZ_T_<stamp>/model.pt
    python -m train.value --since 2026-10-23                      # every round of the league since then
    python -m train.value --matchup PvT --race P                  # the Protoss's critic in PvT (its belief model first)
"""

from __future__ import annotations

import argparse
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
MORPH_FRAMES, MORPH_PX = 48, 64                # a unit gone with the player's new one this near: a morph, not a death
SKIP_ROUNDS = ("CP", "BC_")                    # rounds with paired branches (their replays aren't one game)


# --- pro games ---------------------------------------------------------------------------------

def pro_jobs(race: str = "T", matchup: str = "TvZ", keep=None) -> list[tuple]:
    from inventory import data_root
    won = {}
    with open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r.get("matchup") == matchup and r.get("race") == race and r.get("won") is not None:
                won[r["sha1"], r["player"]] = bool(r["won"])     # (by player: a mirror's two rows share the game)
    root = data_root() / "interim" / "macro" / "v1"
    jobs = []
    with open(root / "index.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if (r.get("ok") and r.get("matchup") == matchup and len(r["players"]) == 2
                    and not (keep and not keep(r["sha1"]))):
                for i, p in enumerate(r["players"]):
                    if p["race"] == race and (r["sha1"], p["name"]) in won:
                        jobs.append((str(root / r["sha1"][:2] / f"{r['sha1']}.npz"), i, won[r["sha1"], p["name"]],
                                     r["sha1"]))
    return jobs


_belief: dict = {}


def _belief_model(race: str, matchup: str):
    if (matchup, race) not in _belief:
        from gary.agent.manifests import model_for
        from gary.policy.belief import BeliefModel
        torch.set_num_threads(1)
        _belief[matchup, race] = BeliefModel.load(model_for("belief", matchup, race))
    return _belief[matchup, race]


def pro_samples(args: tuple) -> tuple[np.ndarray, np.ndarray, str]:
    """The player's view (of race in matchup) every STRIDE_S seconds of one pro game, and whether it won."""
    (path, i, won, sha1), spec_d, race, matchup = args
    from gary.policy.macro import FeatureSpec
    from gary.policy.value import EXTRAS
    spec = FeatureSpec(**spec_d)
    empty = np.zeros((0, spec.size + len(EXTRAS)), np.float16), np.zeros(0, np.float32), sha1
    try:
        d = np.load(path)
    except Exception:
        return empty
    ts = np.arange(0, len(d["frames"]), STRIDE_S)
    if len(ts) < 6:
        return empty
    return view_rows(d, i, spec, race, matchup, ts).astype(np.float16), np.full(len(ts), float(won), np.float32), sha1


def view_rows(d, i: int, spec, race: str, matchup: str, ts: np.ndarray) -> np.ndarray:
    """The critic's view (gary/policy/value.py) of player i of a game's arrays (ingest/macro_dataset.py: a row a
    second) at the rows ts."""
    from gary.races import enemy_of
    enemy = enemy_of(matchup, race).race
    from gary.policy.belief import WORKERS, features_series
    from gary.policy.value import army_supply, belief_outputs, extras, no_style
    from train.macro import encode_rows
    frames = d["frames"]
    seen = d["seen"][:, i].astype(np.float32)
    x = no_style(spec, encode_rows(d, i, spec, ts)[0])
    # the extras (each row of the arrays is a second)
    b = features_series(d["seen"][:, i], d["units"][:, i, :, 0], d["eco"][:, i], frames, enemy, race)[ts]
    own = army_supply(d["units"][:, i, :, 1].astype(np.float32), race)
    workers = d["units"][:, i, WORKERS[race], 1].astype(np.float32)
    seen_army = army_supply(seen, enemy)
    ago = np.maximum(ts - LOOKBACK_S, 0)
    seen_army_max = np.array([seen_army[a:t + 1].max() for a, t in zip(ago, ts)])
    ex = extras(belief_outputs(_belief_model(race, matchup), b), own[ts], own[ago], workers[ts] - workers[ago], seen_army[ts],
                seen_army_max)
    return np.concatenate([x, ex], 1)


# --- Gary's games ------------------------------------------------------------------------------

def league_jobs(rounds: list[str], since: str | None, max_games: int, matchup: str = "TvZ") -> list[tuple]:
    """(replay, side A's result, round/map/seed, side A's race) of the league's games in the matchup: the rounds named
    (globs), or every round saved since a date, newest first, up to max_games."""
    from gary.agent.manifests import MANIFESTS
    import glob
    import os
    root = REPO_ROOT / "runs" / "league"
    paths = sorted({p for r in rounds for p in glob.glob(str(root / r / "round.json"))}) if rounds else \
        [str(p) for p in root.glob("*/round.json")]
    if since:
        t0 = time.mktime(time.strptime(since, "%Y-%m-%d"))
        paths = [p for p in paths if os.path.getmtime(p) >= t0]
    jobs = []
    for p in sorted(paths, key=os.path.getmtime, reverse=True):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        base = str(d.get("terran", "")).split("+")[0]           # (its options may be gone: the base names the race)
        if str(d.get("round", "")).startswith(SKIP_ROUNDS) or d.get("matchup", "TvZ") != matchup or base not in MANIFESTS:
            continue
        a_race = MANIFESTS[base].get("race", "T")
        for g in d.get("games", []):
            if Path(g["replay"]).exists():
                jobs.append((g["replay"], g["result"], f"{d['round']}/{g['map']}/{g['seed']}", a_race))
    return jobs[:max_games]


def deaths(lines: list[dict], slot: int) -> list[tuple[int, float]]:
    """(frame, value) of each unit's death, + for the player in slot (an enemy unit died), - against; a unit gone
    with a new one of its player's at its spot (a drone into an Extractor) morphed and isn't counted."""
    import math
    from gary import protoss as P
    from gary import terran as T
    from gary import zerg as Z
    ev = [r for r in lines if r["type"] == "event"]
    born = [r for r in ev if r["ev"] in ("start", "done")]
    out = []
    for g in (r for r in ev if r["ev"] == "gone"):
        if any(b["slot"] == g["slot"] and b["tag"] != g["tag"] and abs(b["frame"] - g["frame"]) <= MORPH_FRAMES
               and math.dist((b["x"], b["y"]), (g["x"], g["y"])) <= MORPH_PX for b in born):
            continue
        m, gas = T.COST.get(g["unit"]) or Z.COST.get(g["unit"]) or P.COST.get(g["unit"]) or (0, 0)
        v = (m + gas) / (2 if g["unit"] in Z.PAIRS else 1)
        out.append((g["frame"], -v if g["slot"] == slot else v))
    return out


def league_game(args: tuple) -> dict:
    """One league game re-simulated: one player's view every GARY_EVERY_S seconds, the deaths' values (+ for that
    player), its result (won, lost, time). The player: the race's (in a mirror, its nth in slot order: 0 side A, 1
    side B)."""
    (replay, result, game, a_race), spec_d, race, matchup, nth = args
    from macro_dataset import RACES, game_arrays, run_resim
    from gary.policy.macro import FeatureSpec
    out = {"game": game, "x": np.zeros((0, 0)), "frames": [], "events": [], "result": "time", "opp": "?"}
    try:
        lines = run_resim(str(Path(replay).resolve()))
    except Exception:  # noqa: BLE001 - one bad replay must not stop the batch
        return out
    header = next(r for r in lines if r["type"] == "header")
    d = game_arrays(lines)
    races = {p["slot"]: RACES.get(p["race"]) for p in header["players"]}
    if d is None or sorted(races.get(s) for s in d["slots"]) != sorted(matchup.split("v")):
        return out
    slot = [s for s in d["slots"] if races[s] == race][nth]
    ts = np.arange(0, len(d["frames"]), GARY_EVERY_S)
    if len(ts) < 6:
        return out
    if (race != a_race) or nth == 1:                  # (the round's result is side A's)
        result = {"won": "lost", "lost": "won"}.get(result, result)
    return {"game": game, "x": view_rows(d, d["slots"].index(slot), FeatureSpec(**spec_d), race, matchup, ts)
            .astype(np.float16), "frames": d["frames"][ts].tolist(), "events": deaths(lines, slot),
            "result": result if result in ("won", "lost") else "time",
            "opp": next((p["name"].split()[-1] for p in header["players"] if p["slot"] != slot), "?")}


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
    ap.add_argument("--rounds", nargs="*", default=[], help="league rounds (globs) whose games teach the net heads")
    ap.add_argument("--since", help="every league round saved since this date (YYYY-MM-DD)")
    ap.add_argument("--max-games", type=int, default=2000)
    ap.add_argument("--pro-games", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--parallel", type=int, default=14)
    ap.add_argument("--reuse", help="a run folder whose collected data to train on again")
    ap.add_argument("--no-extras", action="store_true", help="ablation: the extras zeroed (the macro model's view alone)")
    ap.add_argument("--race", default="T", choices=["T", "Z", "P"], help="whose value (the league's games read from "
                    "that race's side, both sides in a mirror)")
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--manifest", help="the Gary whose macro model's view the critic reads (default: the race's dev "
                    "build, gt1dev, gz1dev or gp1dev, where the critic is used: the pairs, tests/test_value.py)")
    from train.select import GameFilter
    GameFilter.add_args(ap)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    keep = GameFilter.from_args(args)
    from gary.agent.manifests import manifest_of, model_path
    from gary.policy.macro import MacroModel
    manifest = args.manifest or {"T": "gt1dev", "Z": "gz1dev", "P": "gp1dev"}[args.race]
    spec = MacroModel.load(str(model_path(manifest_of(manifest)["macro"]["model"], args.matchup, args.race))).spec
    print(f"the view of {manifest}'s macro model: {spec.size} inputs", flush=True)
    run = REPO_ROOT / "runs" / "value" / f"{args.matchup}_{args.race}_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    src = Path(args.reuse) if args.reuse else run
    if not args.reuse:
        jobs = pro_jobs(args.race, args.matchup, keep)
        print(f"{len(jobs)} pro player-games ({keep.describe()})", flush=True)
        random.Random(0).shuffle(jobs)
        jobs = jobs[:args.pro_games]
        gjobs = league_jobs(args.rounds, args.since, args.max_games, args.matchup)
        print(f"{len(gjobs)} league games", flush=True)
        mirror = len(set(args.matchup.split("v"))) == 1
        gjobs = [(j, nth) for j in gjobs for nth in ((0, 1) if mirror else (0,))]
        with ProcessPoolExecutor(args.parallel) as pool:
            pro = list(monitor.track(pool.map(pro_samples, [(j, spec.to_dict(), args.race, args.matchup) for j in jobs], chunksize=16), len(jobs),
                                     "pro games"))
            np.savez_compressed(src / "pro.npz", x=np.concatenate([p[0] for p in pro]), win=np.concatenate([p[1] for p in pro]),
                                game=np.concatenate([[p[2]] * len(p[1]) for p in pro]))
            gary = list(monitor.track(pool.map(league_game, [(j, spec.to_dict(), args.race, args.matchup, nth) for j, nth in gjobs]),
                                      len(gjobs), "league games"))
        with open(src / "gary.jsonl", "w", encoding="utf-8") as f:
            for g in gary:
                f.write(json.dumps({**g, "x": g["x"].astype(np.float32).round(3).tolist()}) + "\n")
    pro = np.load(src / "pro.npz")
    gary = [g for g in (json.loads(l) for l in open(src / "gary.jsonl", encoding="utf-8")) if len(g["x"])]
    # held out: a tenth of the pro games (as the other models), Gary's games of every fifth seed
    # (all the styles of a seed together: they share the game's start)
    n0 = len(pro["win"])
    pro_test = np.array([int(s[:8], 16) % 10 == 0 for s in pro["game"]])
    pro_val = np.array([int(s[:8], 16) % 10 == 1 for s in pro["game"]])     # (early stopping: neither trained nor tested)
    gx, gwin, gnet, gtest, gnext, gend, gmin, gopp, gval = [], [], [], [], [], [], [], [], []
    k = n0
    for g in gary:
        w, n, fr = gary_targets(g)
        gx.append(np.array(g["x"], np.float32))
        gwin.append(w)
        gnet.append(n)
        gtest.append(np.full(len(w), int(g["game"].split("/")[-1]) % 5 == 0))
        gval.append(np.full(len(w), int(g["game"].split("/")[-1]) % 5 == 1))
        nxt = np.arange(k + 1, k + len(w) + 1)             # the next sample of the same game, -1 after its last
        nxt[-1] = -1
        gnext.append(nxt)
        gend.append(np.full(len(w), g["result"] != "time"))
        gmin.append(fr / 1440)
        gopp.append(np.array([g.get("opp", "?")] * len(w), object))
        k += len(w)
    gnet = np.concatenate(gnet) if gnet else np.zeros((0, 3), np.float32)
    X = np.concatenate([pro["x"].astype(np.float32)] + gx)
    W = np.concatenate([pro["win"]] + gwin)
    N = np.concatenate([np.full((n0, 2), np.nan, np.float32), gnet[:, :2]])
    R = np.concatenate([np.full(n0, np.nan, np.float32), gnet[:, 2]])          # each step's net value
    NXT = np.concatenate([np.full(n0, -1)] + gnext)
    ENDED = np.concatenate([np.zeros(n0, bool)] + gend)
    test = np.concatenate([pro_test] + gtest)
    val = np.concatenate([pro_val] + gval)
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
    tr = np.where(~test & ~val)[0]
    va = torch.tensor(np.where(val)[0])
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

    def val_loss() -> float:
        """The win and net heads' loss on the validation games (the bootstrapped head's targets move with the net)."""
        net.eval()
        with torch.no_grad():
            lw, ln = net(Xn[va])
        net.train()
        wm, nm = ~torch.isnan(Wt[va]), ~torch.isnan(Nt[va])
        return float(F.binary_cross_entropy_with_logits(lw[wm], Wt[va][wm]) + F.smooth_l1_loss(ln[:, :2][nm], Nt[va][nm]))

    best, best_state, history = float("inf"), None, []
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
        v = val_loss()
        history.append(round(v, 4))
        if v < best:                               # (early stopping: the epoch the validation games liked best, #49)
            best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"validation loss by epoch {history}: kept epoch {history.index(min(history)) + 1}", flush=True)
    net.eval()
    with torch.no_grad():
        lw, ln = net(Xn[torch.tensor(np.where(test)[0])])
    pw, pn = torch.sigmoid(lw).numpy(), ln.numpy()
    Wte, Nte, gte, tte, ote = W[test], N[test], is_gary[test], t_min[test], opp[test]
    print(f"held out: win AUC pro {auc(Wte[~gte], pw[~gte]):.3f}, Gary {auc(Wte[gte], pw[gte]):.3f}")
    for o in sorted(set(ote[gte])):
        s = gte & (ote == o)
        print(f"  vs {o:8s} " + ", ".join(f"{lo}-{hi} min {auc(Wte[s & (tte >= lo) & (tte < hi)], pw[s & (tte >= lo) & (tte < hi)]):.2f}"
                                          for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 99))))
    for j, h in enumerate((60, 180)):
        y, p = Nte[gte, j], pn[gte, j]
        m = ~np.isnan(y)
        r2 = 1 - np.mean((y[m] - p[m]) ** 2) / np.var(y[m]) if m.any() else float("nan")
        print(f"  net{h}: sd {np.std(y[m]) * 1000:.0f}, explained {r2:.1%} (the noise left: sd {np.sqrt(np.mean((y[m] - p[m]) ** 2)) * 1000:.0f})")
        parts = []
        for lo, hi in ((0, 6), (6, 12), (12, 18), (18, 99)):          # (by game time: a head can rank well and be off in scale)
            w_ = m & (tte[gte] >= lo) & (tte[gte] < hi)
            if w_.sum() > 50:
                r2w = 1 - np.mean((y[w_] - p[w_]) ** 2) / np.var(y[w_])
                parts.append(f"{lo}-{hi} min R2 {r2w:+.0%} corr {np.corrcoef(y[w_], p[w_])[0, 1]:+.2f} "
                             f"(sd {np.std(y[w_]) * 1000:.0f}, predicted {np.std(p[w_]) * 1000:.0f})")
        print("    " + "; ".join(parts))
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
