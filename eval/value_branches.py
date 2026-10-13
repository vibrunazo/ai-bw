"""Does the value network see what a decision changed sooner, and with less chaos, than the score?

The test of train/value.py as a critic (docs/CONTROL.md step 7). At moments of Gary's games (the
macro's decisions, as train/macro_rl.py picks them) the same game is played on three times from
the start (games are deterministic per seed: identical up to the moment): as it was ("same"), with
another of the macro's options taken there ("alt", drawn from the model's own odds), and with Gary
skipping one turn there, half a second of hands ("placebo": nothing that matters changed; chaos
alone). Each run records the score's net value (eval/score.py) and Gary's view (the macro model's
features, what the value network reads) LOOKS seconds after the moment.

Each way of judging a run is compared between the pairs: alt - same (a decision's effect plus
chaos) against placebo - same (chaos alone). A judge that reads decisions well spreads the alt
pairs wider than the placebo pairs (the ratio of variances, above 1); the raw score over 3 minutes
is the judge to beat. Judges:

  net180         the score over the 180 s after the moment
  v@T            the score over T s, plus the value network's net180 head at T (bootstrapped)
  win@T          the value network's win head at T (logit)

    python -m eval.value_branches --seeds 9 --points 4    # -> data/interim/value/branches.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority
from eval.block_branch import MAPS, Watch, _match, _step

REPO_ROOT = Path(__file__).resolve().parent.parent
OPPONENTS = ("dummy", "v2", "replayed")
VERSION = "v019"
LOOKS = (30, 60, 180)                  # seconds after the moment: Gary's view recorded
HORIZON = 180 * 24


def out_path() -> Path:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "ingest"))
    from inventory import data_root
    return data_root() / "interim" / "value" / "branches.jsonl"


def _view(m) -> list | None:
    """Gary's view as the macro model sees it (train/value.py gary_game), or None once it's out."""
    g = m.gary
    try:
        obs = g.hi.observe()
    except StopIteration:
        return None
    mine = [u for u in obs["units"] if u["owner"] == g.slot]
    tracker = copy.copy(g.macro.tracker)
    tracker.seen_max = g.macro.tracker.seen_max.copy()
    return tracker.features(obs, g.macro.in_production(mine))[0].astype(np.float32).round(4).tolist()


def run(args: tuple) -> dict:
    """One way of playing on from the point: "same", "alt" or "placebo"."""
    point, how = args
    m = _match(point["opponent"], point["map"], point["seed"], point["version"])
    m.gary.macro.log = True                  # (as the game the point came from: the same work, the same game)
    w = Watch(m)
    w.look()
    f0 = point["frame"]
    if how == "alt":
        m.gary.macro.force = (f0, point["alt"])
    while m.game.frame < f0 and _step(m, w):
        pass
    if m.game.frame != f0:
        m.game.close()
        return {"how": how, "error": f"at {m.game.frame}, not {f0}"}
    if how == "placebo":
        act = m.gary.act
        m.gary.act = lambda: None
        _step(m, w)
        m.gary.act = act
    views, last = {}, None
    looks = sorted(LOOKS)
    while m.game.frame < f0 + HORIZON and _step(m, w):
        while looks and m.game.frame >= f0 + looks[0] * 24:
            last = _view(m) or last
            views[looks.pop(0)] = last
    for t in looks:                           # the game ended first: the last view stands
        views[t] = last
    end = m.game.frame
    out = {"how": how, "end": end, "views": views,
           "net": {t: w.window(f0, min(end, f0 + t * 24))["net"] for t in LOOKS},
           "forced": how != "alt" or m.gary.macro.force is None}
    m.game.close()
    return out


def judge(rows: list[dict], model_path: Path) -> None:
    import torch
    from train.value import ValueNet
    ck = torch.load(model_path)
    net = ValueNet(ck["n_in"])
    net.load_state_dict(ck["state"])
    net.eval()
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)

    def value(view):
        with torch.no_grad():
            lw, ln = net(torch.tensor(((np.array(view, np.float32) - mean) / std)[None]))
        return float(lw[0]), float(ln[0, 1]) * 1000

    def judges(r):
        out = {"net180": r["net"]["180"]}
        for t in (30, 60):
            v = r["views"].get(str(t))
            if v is not None:
                lw, n180 = value(v)
                out[f"v@{t}"] = r["net"][str(t)] + n180
                out[f"win@{t}"] = lw
        v = r["views"].get("180")
        if v is not None:
            out["win@180"] = value(v)[0]
        return out

    names = ["net180", "v@30", "v@60", "win@30", "win@60", "win@180"]
    d = {k: {"alt": [], "placebo": []} for k in names}
    for p in rows:
        js = {h: judges(p[h]) for h in ("same", "alt", "placebo") if h in p}
        for h in ("alt", "placebo"):
            if h in js and "same" in js:
                for k in names:
                    if k in js[h] and k in js["same"]:
                        d[k][h].append(js[h][k] - js["same"][k])
    print(f"{len(rows)} moments; per judge, the spread of the pairs (sd) and how much wider a decision's "
          f"is than chaos's (variance ratio):")
    for k in names:
        a, pl = np.array(d[k]["alt"]), np.array(d[k]["placebo"])
        if len(a) and len(pl):
            print(f"  {k:8s} alt sd {a.std():8.2f}  placebo sd {pl.std():8.2f}  ratio {a.var() / max(pl.var(), 1e-9):5.2f}  "
                  f"changed at all: alt {np.mean(a != 0):.0%}, placebo {np.mean(pl != 0):.0%}")
    # does a judge's early read foretell how the pair ends? (the critic reads the decisions taken
    # among its inputs, so a wider spread alone could be the label, not its consequence)
    for h in ("alt", "placebo"):
        n = np.array(d["net180"][h])
        cs = [f"{k} {np.corrcoef(np.array(d[k][h]), n)[0, 1]:+.2f}" for k in names[1:]
              if len(d[k][h]) == len(n) > 2 and np.std(d[k][h]) > 0 and np.std(n) > 0]
        print(f"  {h} pairs, correlation with net180: {', '.join(cs)}")
    same = [p["same"]["net"]["180"] == p["base"]["net"] for p in rows if "base" in p]
    print(f"  the same game played again matched its first play over 180 s in {np.mean(same):.0%} of moments")


def main() -> None:
    from train.macro_rl import base_game
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--points", type=int, default=4, help="decision moments per game")
    ap.add_argument("--version", default=VERSION)
    ap.add_argument("--model", help="the value network (default: the newest in runs/value)")
    ap.add_argument("--data", help="branches played before (JSON lines): judge them, playing none")
    ap.add_argument("--parallel", type=int, default=14)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    if args.data:
        rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    else:
        jobs = [(o, mp, s, args.points, args.version) for o in OPPONENTS for mp in MAPS for s in range(1, args.seeds + 1)]
        with ProcessPoolExecutor(args.parallel) as pool:
            points = [p for ps in monitor.track(pool.map(base_game, jobs), len(jobs), "games to find moments") for p in ps]
            print(f"{len(points)} decision moments in {len(jobs)} games", flush=True)
            runs = list(monitor.track(pool.map(run, [(p, h) for p in points for h in ("same", "alt", "placebo")]),
                                      3 * len(points), "branches (same, alt, placebo)"))
        rows = []
        for i, p in enumerate(points):
            got = {r["how"]: r for r in runs[3 * i:3 * i + 3] if not r.get("error") and r["forced"]}
            if "same" in got:
                rows.append({**{k: v for k, v in p.items() if k != "x"}, **got})
        bad = len(points) - len(rows)
        if bad:
            print(f"{bad} moments dropped (a run diverged before its moment)")
        out_path().parent.mkdir(parents=True, exist_ok=True)
        out_path().write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"-> {out_path()}")
        rows = [json.loads(json.dumps(r)) for r in rows]          # (keys as read back: strings)
    model = Path(args.model) if args.model else max((REPO_ROOT / "runs" / "value").glob("*/model.pt"))
    judge(rows, model)


if __name__ == "__main__":
    main()
