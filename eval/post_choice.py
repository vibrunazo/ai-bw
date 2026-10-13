"""Where the army waits at home, chosen per situation: learned on the training home defenses, judged
on the held-out ones (docs/CONTROL.md step 7: big, rare decisions are the ones to learn).

The post spots are gary/agent/roles.py's post option (manifests' post, post0, post6, post_minerals,
post_ramp, post_bunker); eval/episodes.py --out plays each on every home defense, several draws of
Gary's hands, and records each episode's situation at its start (minute, Command Centers, finished
bunkers, army and zerg in sight, whether the fight is at the main). A situation is a cell of a few
of those, coarsely binned; each cell's spot is the one with the best mean net value on the
training defenses there (a cell with few defenses leans on all of them: BORROW, as the strategy
selector). Which situation features make the cells is chosen on the training defenses alone
(halves of them, crossed); the chosen rule is then judged once on the held-out defenses, paired
with today's spot (3 tiles from the Command Center toward the zerg), against each fixed spot too.

    python -m eval.post_choice runs/posts/train.json runs/posts/test.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st

BASE = "v019"
BORROW = 10                     # a cell with fewer training defenses than this leans on all of them
BINS = {
    "at_main": lambda c: {True: "main", False: "other", None: "?"}[c.get("at_main")],
    "bases": lambda c: "1 base" if c["ccs"] <= 1 else "2+ bases",
    "bunker": lambda c: "bunker" if c["bunkers"] else "no bunker",
    "army": lambda c: "army <=8" if c["army"] <= 8 else "army <=20" if c["army"] <= 20 else "army >20",
    "zerg": lambda c: "no zerg seen" if c["zerg"] == 0 else "zerg <=8" if c["zerg"] <= 8 else "zerg >8",
    "minute": lambda c: "<4:00" if c["minute"] < 4 else "4-6:00" if c["minute"] < 6 else "6:00+",
}


def episodes(path: str) -> dict:
    """{episode: {"context": ..., "net": {spot: mean net over its draws}}}."""
    out: dict = {}
    for r in json.load(open(path, encoding="utf-8")):
        if r["error"] or r["net"] is None or r.get("context") is None:
            continue
        e = out.setdefault(r["sha1"], {"context": r["context"], "net": {}})
        e["net"].setdefault(r["who"], []).append(r["net"])
    spots = set.intersection(*(set(e["net"]) for e in out.values())) if out else set()
    return {k: {"context": e["context"], "net": {s: st.mean(e["net"][s]) for s in spots}}
            for k, e in out.items() if spots <= set(e["net"])}


def cell(ctx: dict, feats: tuple) -> tuple:
    return tuple(BINS[f](ctx) for f in feats)


def fit(eps: dict, feats: tuple) -> tuple[dict, str]:
    """(cell -> spot, the spot for unseen cells): each cell's best mean, leaning on all defenses."""
    spots = sorted(next(iter(eps.values()))["net"])
    overall = {s: st.mean(e["net"][s] for e in eps.values()) for s in spots}
    cells: dict = {}
    for e in eps.values():
        cells.setdefault(cell(e["context"], feats), []).append(e)
    rule = {}
    for c, es in cells.items():
        n = len(es)
        w = max(0, BORROW - n)
        score = {s: (sum(e["net"][s] for e in es) + w * overall[s]) / (n + w) for s in spots}
        rule[c] = max(score, key=score.get)
    return rule, max(overall, key=overall.get)


def gain(eps: dict, choose) -> tuple[float, float, int]:
    """Mean paired gain over today's spot (net per defense), its standard error, defenses."""
    d = [e["net"][choose(e)] - e["net"][BASE] for e in eps.values()]
    return st.mean(d), (st.stdev(d) / len(d) ** .5 if len(d) > 1 else 0.0), len(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--max-features", type=int, default=2, help="situation features in a cell, at most")
    args = ap.parse_args()
    train, test = episodes(args.train), episodes(args.test)
    spots = sorted(next(iter(train.values()))["net"])
    print(f"{len(train)} training defenses, {len(test)} held out; spots: {', '.join(spots)}")
    # which features make the cells: crossed halves of the training defenses
    keys = sorted(train)
    halves = [{k: train[k] for k in keys[i::2]} for i in range(2)]
    options = [()] + [f for n in range(1, args.max_features + 1) for f in itertools.combinations(BINS, n)]
    scored = []
    for feats in options:
        g = []
        for a, b in ((0, 1), (1, 0)):
            rule, default = fit(halves[a], feats)
            g.append(gain(halves[b], lambda e: rule.get(cell(e["context"], feats), default))[0])
        scored.append((st.mean(g), feats))
    scored.sort(reverse=True)
    print("situation features, by the gain their rule makes on the other half of the training defenses:")
    for s, feats in scored[:6]:
        print(f"  {s:+6.1f}  {', '.join(feats) or '(one spot for all)'}")
    best = scored[0][1]
    rule, default = fit(train, best)
    print(f"the rule (from all training defenses, cells by {', '.join(best) or 'nothing'}):")
    counts = {}
    for e in train.values():
        counts[cell(e["context"], best)] = counts.get(cell(e["context"], best), 0) + 1
    for c in sorted(rule, key=lambda c: -counts[c]):
        print(f"  {' / '.join(c) or 'all':40s} {counts[c]:4d} defenses -> {rule[c]}")
    print("held out, gain over today's spot (net per defense, paired):")
    m, se, n = gain(test, lambda e: rule.get(cell(e["context"], best), default))
    print(f"  the rule                 {m:+6.1f} ± {se:.1f}  ({n} defenses)")
    for s in spots:
        if s != BASE:
            m, se, _ = gain(test, lambda e, s=s: s)
            print(f"  always {s:18s} {m:+6.1f} ± {se:.1f}")
    m, se, _ = gain(test, lambda e: max(e["net"], key=e["net"].get))
    print(f"  (hindsight: each defense's best spot {m:+.1f}; partly luck of the draws)")


if __name__ == "__main__":
    main()
