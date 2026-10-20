"""Learn when to go into the early-pool defense (gary/policy/switch.py), from paired games.

Two-thirds of v0.19's losses (eval/losses.py) are hatch firsts with a pool at ~2:02 whose first
zerglings reach Gary at ~3:30 while its posture still plays the usual plan (zerglings count as an
early pool only while one hatchery is seen: calling every hatch first's zerglings one held Gary's
tech back a minute). When to switch is a big, rare decision: learned here from game outcomes.

collect  every training game (the opponent pool on its maps, seeds 1..--seeds) played as v0.19's
         rules play it, then again with the rush defense forced at each of FORCE_S (games are
         deterministic: the same game up to the moment, the switch the only difference); a moment
         at or after the rules' own switch isn't played (the same game). Each game's outcome: won
         1, lost 0, at the time limit the value network's chance to win at its end
         (gary/policy/value.py). The facts Gary knew at each moment (gary/policy/plan.py facts)
         come from the rules' game.
learn    a small regression tree (depth DEPTH) predicting a moment's gain (the forced game's
         outcome minus the rules' game's) from those facts, judged across seeds first (folds by
         seed) against never switching; saved as JSON for the posture.
test     held-out games (seeds 101..) played by the rules and with the learned switch, paired:
         losses and outcomes.

    python -m train.early_switch --seeds 48            # -> runs/early_switch/<stamp>/switch.json
    python -m train.early_switch --test runs/early_switch/<stamp>/switch.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
SETS = [("gz1", "standard_ozp3w"), ("gz1", "python_gg13676"), ("gz1", "heartbreak_9rg84")]   # (the Zerg Gary; the
                                             # scripted Zergs were deleted in docs/REFACTOR.md Phase 3)
VERSION = "v019"                              # (--version: the Gary that plays them)
FORCE_S = (120, 150, 180, 210, 240)          # forced switch moments (seconds)
FACTS_EVERY = 10 * 24                        # the facts recorded this often, FACTS_FROM to FACTS_UNTIL
FACTS_FROM, FACTS_UNTIL = 100 * 24, 260 * 24
MINUTES = 12
EARLY_UNTIL = 300 * 24                        # zerglings at Gary's bases by 5:00: an early attack (not a false alarm)
HOME_PX = 20 * 32                             # ...this close to one of its Command Centers
EARLY_LINGS = 4
DEPTH = 3
MIN_LEAF = 20
SIZES = ((1, 40), (2, 30), (2, 60), (3, 20), (3, 40))   # tree sizes tried (depth, smallest leaf)
REPEATS = 3                                    # reshufflings of the 4 folds by seed


def _critic():
    from gary.agent.manifests import model_for
    from gary.policy.belief import BeliefModel
    from gary.policy.value import ValueModel
    newest = max((REPO_ROOT / "runs" / "value").glob("*/model.pt"))
    return ValueModel.load(newest), BeliefModel.load(model_for("belief", "TvZ", "T"))


def _zerglings_home(m) -> bool:
    """EARLY_LINGS or more zerglings by one of Gary's Command Centers (the true game state: this
    judges the alarm, it isn't Gary's input)."""
    o = m.game.observe()
    ccs = [(u["x"], u["y"]) for u in o["units"] if u["owner"] == m.slot["T"] and u["type"] == 106]
    lings = [(u["x"], u["y"]) for u in o["units"] if u["owner"] == m.slot["Z"] and u["type"] == 37]
    return sum(1 for z in lings if any((z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2 <= HOME_PX ** 2 for c in ccs)) >= EARLY_LINGS


def play(args: tuple) -> dict:
    """One game: as the rules play it (force None), forced at force seconds, or with a learned switch;
    a sixth element: the Gary that plays (default VERSION)."""
    opponent, mp, seed, force, switch, *rest = args
    version = rest[0] if rest else VERSION
    import torch
    from eval.match import Match
    from gary.policy.army import supply_x2
    from gary.policy.switch import facts
    from gary.policy.value import ValueView
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, version, MAPS[mp], opponent)
    minutes = MINUTES
    g = m.gary
    p = g.posture
    if force is not None:
        p.force_at = force * 24
    if switch is not None:
        from gary.policy.switch import Switch
        p.switch = Switch.load(switch)
    critic, belief = _critic()
    view = ValueView(g, belief)
    fx, rush_at, last_x = {}, None, None
    zerg_home, at6 = None, None            # first second 4+ zerglings were by Gary's bases; Gary at 6:00
    quiet = contextlib.redirect_stdout(io.StringIO())
    quiet.__enter__()
    try:
        while m.game.frame < minutes * 60 * 24 and not m.done:
            m.step()
            f = m.game.frame
            if f % 24 >= 2:
                continue
            try:
                obs = g.hi.observe()
            except StopIteration:
                break
            view.look(obs)
            if rush_at is None and p.rush:
                rush_at = f
            if zerg_home is None and f <= EARLY_UNTIL and f % 48 < 2:
                zerg_home = f if _zerglings_home(m) else None
            if at6 is None and f >= 6 * 60 * 24:
                mine = [u for u in obs["units"] if u["owner"] == g.slot]
                at6 = {"scvs": sum(1 for u in mine if u["type"] == 7),
                       "bases": sum(1 for u in mine if u["type"] == 106),
                       "army": sum(supply_x2(u["type"]) for u in mine if u["type"] in g.army_types and u["completed"]) / 2}
            if FACTS_FROM <= f <= FACTS_UNTIL and f % FACTS_EVERY < 2:
                fx[f // 24] = facts(p, obs)
            if f % (30 * 24) < 2:
                last_x = view.features(obs)
    finally:
        quiet.__exit__(None, None, None)
    players = {q["slot"] for q in m.game.observe()["players"]}
    result = "won" if m.slot["T"] in players and m.slot["Z"] not in players else \
        "lost" if m.slot["T"] not in players else "time"
    chance = None
    if result == "time" and last_x is not None:
        chance = float(1 / (1 + np.exp(-critic.predict(last_x)["win"][0])))
    grade = {"won": 1.0, "lost": 0.0}.get(result, chance if chance is not None else 0.5)
    m.game.close()
    return {"opponent": opponent, "map": mp, "seed": seed, "force": force, "switch": switch is not None,
            "result": result, "grade": grade,
            "rush_at_s": rush_at // 24 if rush_at is not None else None, "facts": fx,
            "zerg_home_s": zerg_home // 24 if zerg_home is not None else None, "at6": at6}


def collect(seeds: list[int], parallel: int, out: Path, sets=SETS, version: str = VERSION) -> list[dict]:
    base_jobs = [(o, mp, s, None, None, version) for o, mp in sets for s in seeds]
    with ProcessPoolExecutor(parallel) as pool:
        base = list(monitor.track(pool.map(play, base_jobs), len(base_jobs), "games as the rules play them"))
        forced_jobs = [(b["opponent"], b["map"], b["seed"], t, None, version) for b in base for t in FORCE_S
                       if b["rush_at_s"] is None or t < b["rush_at_s"]]
        forced = list(monitor.track(pool.map(play, forced_jobs), len(forced_jobs), "the same games, switch forced"))
    rows = base + forced
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def moments(rows: list[dict]) -> list[dict]:
    """Per forced game: the facts Gary knew at that moment (from the rules' game) and the gain (the
    scout's facts too where the games have them)."""
    from gary.policy.switch import SCOUT, names
    sample = next((fx for r in rows for fx in r["facts"].values()), {})
    FACTS = names() if all(k in sample for k in SCOUT) else [k for k in names() if k not in SCOUT]
    base = {(r["opponent"], r["map"], r["seed"]): r for r in rows if r["force"] is None and not r["switch"]}
    out = []
    for r in rows:
        b = base.get((r["opponent"], r["map"], r["seed"]))
        if r["force"] is None or b is None:
            continue
        fx = b["facts"].get(str(r["force"])) or b["facts"].get(r["force"])
        if fx is None:
            continue
        out.append({"key": (r["opponent"], r["map"], r["seed"]), "force": r["force"], "names": FACTS,
                    "x": [fx[k] for k in FACTS], "gain": r["grade"] - b["grade"],
                    "lost": (r["result"] == "lost") - (b["result"] == "lost")})
    return out


def fit_tree(ms: list[dict], depth: int = DEPTH, leaf: int = MIN_LEAF):
    from sklearn.tree import DecisionTreeRegressor
    t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0)
    t.fit(np.array([m["x"] for m in ms]), np.array([m["gain"] for m in ms]))
    return t


def as_json(t, FACTS) -> dict:
    tr = t.tree_
    return {"left": tr.children_left.tolist(), "right": tr.children_right.tolist(),
            "feature": [FACTS[i] if i >= 0 else None for i in tr.feature],
            "threshold": tr.threshold.tolist(), "value": tr.value[:, 0, 0].tolist(),
            "n": tr.n_node_samples.tolist()}


def policy_gain(ms: list[dict], decide) -> float:
    """What a switch policy gains per game on these moments: in each game, the first moment it
    switches at (the game's gain then), or nothing."""
    by_game: dict = {}
    for m in sorted(ms, key=lambda m: m["force"]):
        by_game.setdefault(m["key"], []).append(m)
    total = 0.0
    for key, gm in by_game.items():
        first = next((m for m in gm if decide(m)), None)
        total += first["gain"] if first else 0.0
    return total / max(1, len(by_game))


def learn(rows: list[dict], out: Path) -> Path:
    from gary.policy.switch import MARGIN, Switch
    ms = moments(rows)
    keys = sorted({m["key"] for m in ms})
    print(f"{len(ms)} forced moments in {len(keys)} games; mean gain {st.mean(m['gain'] for m in ms):+.3f}, "
          f"losses {sum(m['lost'] for m in ms):+d} (forced minus the rules)")
    for t in FORCE_S:
        sub = [m for m in ms if m["force"] == t]
        if sub:
            print(f"  forced at {t // 60}:{t % 60:02d}: {len(sub)} games, gain {st.mean(m['gain'] for m in sub):+.3f}, "
                  f"losses {sum(m['lost'] for m in sub):+d}")
    # across seeds (folds by seed: a seed's games on every map together), REPEATS reshufflings of 4
    # folds for each tree size in SIZES; the best size is kept
    seeds = sorted({k[2] for k in keys})
    results = []
    for depth, leaf in SIZES:
        cv = []
        for rep in range(REPEATS):
            perm = list(np.random.default_rng(rep).permutation(seeds))
            for i in range(4):
                fold = set(perm[i::4])
                train = [m for m in ms if m["key"][2] not in fold]
                test = [m for m in ms if m["key"][2] in fold]
                t = fit_tree(train, depth, leaf)
                pred = dict(zip(map(id, test), t.predict(np.array([m["x"] for m in test]))))
                cv.append(policy_gain(test, lambda m: pred[id(m)] > MARGIN))
        se = st.stdev(cv) / len(cv) ** 0.5
        results.append((st.mean(cv), se, depth, leaf, sum(c > 0 for c in cv), len(cv)))
        print(f"  tree depth {depth}, leaves of {leaf}+: across seeds {st.mean(cv):+.3f} a game (± {se:.3f}; "
              f"positive in {sum(c > 0 for c in cv)} of {len(cv)} folds)")
    best = max(results)
    print(f"the learned switch across seeds: {best[0]:+.3f} ± {best[1]:.3f} a game (depth {best[2]}, leaves of {best[3]}+); "
          f"never switching: 0; hindsight best per game: {policy_gain(ms, lambda m: m['gain'] > 0):+.3f}")
    t = fit_tree(ms, best[2], best[3])
    tree = as_json(t, ms[0]["names"])
    for line in Switch.describe(tree):
        print("  " + line)
    path = out / "switch.json"
    path.write_text(json.dumps({"tree": tree, "margin": MARGIN, "cv_gain": best[0], "cv_se": best[1]}, indent=1),
                    encoding="utf-8")
    print(f"-> {path}")
    return path


def test(switch: str, seeds: list[int], parallel: int, sets=SETS, out: Path | None = None, version: str = VERSION) -> None:
    jobs = [(o, mp, s, None, sw, version) for o, mp in sets for s in seeds for sw in (None, switch)]
    with ProcessPoolExecutor(parallel) as pool:
        rows = list(monitor.track(pool.map(play, jobs), len(jobs), "held-out games, rules and learned switch"))
    if out is not None:
        out.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "facts"}) + "\n" for r in rows),
                       encoding="utf-8")
    report(rows, sets)


def report(rows: list[dict], sets=SETS) -> None:
    """Paired: losses, wins, the outcome; how often the defense went on by 4:30 and how often with no
    early zerglings at Gary's bases (false alarms); SCVs, bases and army at 6:00."""
    def fired(r):
        return r["rush_at_s"] is not None and r["rush_at_s"] <= 270

    def false_alarm(r):
        return fired(r) and r.get("zerg_home_s") is None

    def at6(rs, k):
        v = [r["at6"][k] for r in rs if r.get("at6")]
        return st.mean(v) if v else float("nan")
    print("  (rules -> learned switch)            losses   wins   outcome          defense by 4:30  "
          "false alarms / quiet games   at 6:00: SCVs, bases, army")
    for o, mp in list(sets) + [("all", "")]:
        def pick(sw):
            return {(r["opponent"], r["map"], r["seed"]): r for r in rows
                    if r["switch"] == sw and (o == "all" or (r["opponent"], r["map"]) == (o, mp))}
        a, b = pick(False), pick(True)
        both = [k for k in a if k in b]
        if not both:
            continue
        A, B = [a[k] for k in both], [b[k] for k in both]
        d = [y["grade"] - x["grade"] for x, y in zip(A, B)]
        n = lambda rs, f: sum(1 for r in rs if f(r))
        quiet = lambda rs: sum(1 for r in rs if r.get("zerg_home_s") is None)     # no early zerglings (each arm's own)
        print(f"  {o:8s} {mp:16s} {len(both):3d}  {n(A, lambda r: r['result'] == 'lost'):2d} -> {n(B, lambda r: r['result'] == 'lost'):2d}"
              f"  {n(A, lambda r: r['result'] == 'won'):2d} -> {n(B, lambda r: r['result'] == 'won'):2d}"
              f"  {st.mean(d):+.3f} ± {st.stdev(d) / len(d) ** .5 if len(d) > 1 else 0:.3f}"
              f"  {n(A, fired):3d} -> {n(B, fired):3d}"
              f"   {n(A, false_alarm):3d}/{quiet(A):<3d} -> {n(B, false_alarm):3d}/{quiet(B):<3d}"
              f"   {at6(A, 'scvs'):.1f} -> {at6(B, 'scvs'):.1f}, {at6(A, 'bases'):.2f} -> {at6(B, 'bases'):.2f},"
              f" {at6(A, 'army'):.1f} -> {at6(B, 'army'):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--version", default=VERSION, help="the Gary that plays (e.g. v019+watch: the touring scout)")
    ap.add_argument("--data", help="collected games (JSON lines): learn from them, playing none")
    ap.add_argument("--test", help="a learned switch (switch.json): play the held-out games with it")
    ap.add_argument("--test-seeds", type=int, nargs=2, default=[101, 148])
    ap.add_argument("--sets", nargs="+", help="only these opponent:map sets (e.g. v2:python_gg13676)")
    ap.add_argument("--report", nargs="+", help="report held-out games played before (test_*.jsonl), playing none")
    ap.add_argument("--parallel", type=int, default=14)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    sets = [tuple(x.split(":")) for x in args.sets] if args.sets else SETS
    if args.report:
        rows = [json.loads(line) for f in args.report for line in open(f, encoding="utf-8")]
        report(rows, [s for s in SETS if any((r["opponent"], r["map"]) == s for r in rows)])
        return
    if args.test:
        out = Path(args.test).with_name(f"test_{args.test_seeds[0]}-{args.test_seeds[1]}.jsonl")
        test(args.test, list(range(args.test_seeds[0], args.test_seeds[1] + 1)), args.parallel, sets, out, args.version)
        return
    out = REPO_ROOT / "runs" / "early_switch" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    if args.data:                          # (with --sets: those sets' games collected, added to these)
        rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
        if args.sets:
            rows += collect(list(range(1, args.seeds + 1)), args.parallel, out / "games_added.jsonl", sets, args.version)
            (out / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    else:
        rows = collect(list(range(1, args.seeds + 1)), args.parallel, out / "games.jsonl", sets, args.version)
    learn(rows, out)


if __name__ == "__main__":
    main()
