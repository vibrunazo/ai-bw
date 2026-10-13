"""Does the value network see what a decision changed sooner, and with less chaos, than the score?

The test of train/value.py as a critic (docs/CONTROL.md step 7). At moments of Gary's games (the
macro's decisions, as train/macro_rl.py picks them) the same game is played on three times from
the start (games are deterministic per seed: identical up to the moment): as it was ("same"), with
another of the macro's options taken there ("alt", drawn from the model's own odds), and with Gary
skipping one turn there, half a second of hands ("placebo": nothing that matters changed; chaos
alone). Each run records the score's net value (eval/score.py) and the critic's view of the moment
(gary/policy/value.py ValueView, looking once a second from the game's start) LOOKS seconds after.

Each way of judging a run is compared between the pairs: alt - same (a decision's effect plus
chaos) against placebo - same (chaos alone). A judge that reads decisions well spreads the alt
pairs wider than the placebo pairs (the ratio of variances, above 1); the raw score over 3 minutes
is the judge to beat. Judges:

  net180         the score over the 180 s after the moment
  v@T            the score over T s, plus the value network's net180 head at T (bootstrapped)
  d@T            the score over T s, plus its discounted head at T (discounted by T s)
  win@T          the value network's win head at T (logit)

And foresight: the critic's read alone at T against the score after T (on either kind of pair:
once a game has changed, how the rest goes can be foreseen).

    python -m eval.value_branches --seeds 9 --points 4    # -> data/interim/value/branches.jsonl
"""

from __future__ import annotations

import argparse
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


_belief = None


def _belief_model():
    global _belief
    if _belief is None:
        from gary.agent.manifests import model_path
        from gary.policy.belief import BeliefModel
        _belief = BeliefModel.load(model_path("belief/latest"))
    return _belief


def _advance(m, w, view) -> bool:
    """One step of the game, the critic's view looking once a second (as train/value.py's games)."""
    if not _step(m, w):
        return False
    if m.game.frame % 24 < 2:
        try:
            view.look(m.gary.hi.observe())
        except StopIteration:                 # Gary is out
            pass
    return True


def run(args: tuple) -> dict:
    """One way of playing on from the point: "same", "alt" or "placebo"."""
    from gary.policy.value import ValueView
    point, how = args
    m = _match(point["opponent"], point["map"], point["seed"], point["version"])
    m.gary.macro.log = True                  # (as the game the point came from: the same work, the same game)
    w = Watch(m)
    w.look()
    view = ValueView(m.gary, _belief_model())
    f0 = point["frame"]
    if how == "alt":
        m.gary.macro.force = (f0, point["alt"])
    while m.game.frame < f0 and _advance(m, w, view):
        pass
    if m.game.frame != f0:
        m.game.close()
        return {"how": how, "error": f"at {m.game.frame}, not {f0}"}
    if how == "placebo":
        act = m.gary.act
        m.gary.act = lambda: None
        _advance(m, w, view)
        m.gary.act = act
    views, last = {}, None
    looks = sorted(LOOKS)
    while m.game.frame < f0 + HORIZON and _advance(m, w, view):
        while looks and m.game.frame >= f0 + looks[0] * 24:
            try:
                obs = m.gary.hi.observe()
                view.look(obs)
                last = view.features(obs)[0].astype(np.float32).round(4).tolist()
            except StopIteration:
                pass
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
    from gary.policy.value import GAMMA_S, ValueModel
    model = ValueModel.load(model_path)
    disc = "disc" in model.ckpt.get("heads", [])

    def judges(r):
        out = {"net180": r["net"]["180"]}
        for t in (30, 60, 180):
            v = r["views"].get(str(t))
            if v is None:
                continue
            p = model.predict(np.array(v, np.float32))
            out[f"win@{t}"] = float(p["win"][0])
            if t < 180:
                out[f"v@{t}"] = r["net"][str(t)] + float(p["net180"][0])
                out[f"c@{t}"] = float(p["net180"][0])                # the critic alone (what it foresees)
                if disc:
                    out[f"d@{t}"] = r["net"][str(t)] + GAMMA_S ** t * float(p["disc"][0])
                    out[f"cd@{t}"] = float(p["disc"][0])
        out["rest30"] = r["net"]["180"] - r["net"]["30"]           # the score after each look
        out["rest60"] = r["net"]["180"] - r["net"]["60"]
        return out

    names = ["net180", "v@30", "v@60"] + (["d@30", "d@60"] if disc else []) + ["win@30", "win@60", "win@180"]
    fore = ["c@30", "c@60"] + (["cd@30", "cd@60"] if disc else [])
    d = {k: {"alt": [], "placebo": []} for k in names + fore + ["rest30", "rest60"]}
    for p in rows:
        js = {h: judges(p[h]) for h in ("same", "alt", "placebo") if h in p}
        if not all(k in js.get("same", {}) for k in d):
            continue
        for h in ("alt", "placebo"):
            if h in js and all(k in js[h] for k in d):
                for k in d:
                    d[k][h].append(js[h][k] - js["same"][k])
    print(f"{len(rows)} moments; per judge, the spread of the pairs (sd) and how much wider a decision's "
          f"is than chaos's (variance ratio):")
    for k in names:
        a, pl = np.array(d[k]["alt"]), np.array(d[k]["placebo"])
        if len(a) and len(pl):
            print(f"  {k:8s} alt sd {a.std():8.2f}  placebo sd {pl.std():8.2f}  ratio {a.var() / max(pl.var(), 1e-9):5.2f}  "
                  f"changed at all: alt {np.mean(a != 0):.0%}, placebo {np.mean(pl != 0):.0%}")
    # foresight: does the critic's read at 30 or 60 s foretell the score after it? (the critic
    # reads the decisions taken among its inputs, so a wider spread alone could be the label)
    c = lambda a, b: np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    for h in ("alt", "placebo"):
        r = {k: np.array(v[h]) for k, v in d.items()}
        parts = [f"{k} {c(r[k], r['rest' + k.split('@')[1]]):+.2f}" for k in fore]
        print(f"  {h} pairs, the critic's read against the score after it: {', '.join(parts)}")
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
