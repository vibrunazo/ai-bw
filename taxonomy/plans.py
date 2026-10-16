"""Zerg plans: a categorical build label for every pro game (docs/SELF_PLAY.md, steering Gary-Z).

taxonomy/discover.py's clusters on item timings left two thirds of the Zerg games in no cluster
(one cluster of 24%, four of about 2%), and their count swings with HDBSCAN's settings (2 to 23).
Zerg builds vary smoothly in timing but commit to a few discrete choices, so the label is those:

  opening    pool first or hatchery first (which of the Spawning Pool and the 2nd Hatchery came first)
  hatcheries 2 or 3 by HATCH_BY (the 3rd Hatchery before it, or not)
  tech       which came first by TECH_BY: the Spire (mutalisks), Lurker Aspect (lurkers); without
             either, PRESSURE_LINGS zerglings or more by 4:00 (zergling pressure), a Hydralisk Den
             (hydralisks), none of them in a game that went to TECH_BY (no tech), or a game that ended
             before it without a choice (undecided)

Every game gets one, from what it showed by its end (short games too: left out, "no style" meant
"a short game", and early-pool rushes were in no plan); plans under MIN_SHARE of the games are
merged into "other". Output, like
discover.py's: data/interim/taxonomy/<matchup>_Z_plan.json (assignments 'rel_path|player' -> plan
id) and .md (one card a plan: share, win rate, median timings), for review.

    python taxonomy/plans.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from discover import data_root, first_times, load_rows, mmss

HATCH_BY = 5 * 60
TECH_BY = 9 * 60
PRESSURE_BY, PRESSURE_LINGS = 4 * 60, 10
MIN_SHARE = 0.01
CARD_ITEMS = ["pool", "hatch2", "hatch3", "gas1", "lair", "spire", "muta", "den", "lurker_aspect", "hive"]


def lings_by(r: dict, when: float) -> int:
    return sum(1 for i in r["items"] if i["name"] == "Zergling" and i["t"] <= when and i["conf"] in ("confirmed", "ordered"))


def plan_of(t: dict[str, float], lings: int = 0, duration: float = TECH_BY) -> tuple[str, str, str]:
    never = 9e9
    opening = "pool first" if t.get("pool", never) < t.get("hatch2", never) else "hatchery first"
    hatcheries = "3 hatcheries" if t.get("hatch3", never) <= HATCH_BY else "2 hatcheries"
    spire, lurker, den = t.get("spire", never), t.get("lurker_aspect", never), t.get("den", never)
    if min(spire, lurker) < never:
        tech = "spire first" if spire <= lurker else "lurker first"
    elif lings >= PRESSURE_LINGS:
        tech = "zergling pressure"
    elif den < never:
        tech = "hydralisks"
    else:
        tech = "no tech" if duration >= TECH_BY else "undecided"
    return opening, hatcheries, tech


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    args = ap.parse_args()
    rows = load_rows(args.matchup, "Z")
    times = [first_times(r["items"], "Z", TECH_BY) for r in rows]
    plans = [plan_of(t, lings_by(r, PRESSURE_BY), r["duration_s"]) for r, t in zip(rows, times)]
    counts = Counter(plans)
    kept = [p for p, n in counts.most_common() if n / len(rows) >= MIN_SHARE]
    ids = {p: i for i, p in enumerate(kept)}
    other = len(kept)
    members = defaultdict(list)
    assignments = {}
    for r, t, p in zip(rows, times, plans):
        k = ids.get(p, other)
        members[k].append((r, t))
        assignments[f"{r['rel_path']}|{r['player']}"] = k
    names = [", ".join(p) for p in kept] + ["other"]
    md = [f"# {args.matchup} Z plans (by {mmss(TECH_BY)}; hatcheries by {mmss(HATCH_BY)})", "",
          f"{len(rows)} Zerg player-games of 4:00 or more; every one has a plan.", ""]
    for k in range(len(names)):
        ms = members.get(k, [])
        if not ms:
            continue
        won = [r.get("won") for r, _ in ms if r.get("won") is not None]
        md += [f"## {k}: {names[k]} ({len(ms)} games, {len(ms) / len(rows):.1%}"
               + (f", win rate {np.mean(won):.0%})" if won else ")"), ""]
        line = []
        for item in CARD_ITEMS:
            vals = [t[item] for _, t in ms if item in t]
            if len(vals) >= 0.2 * len(ms):
                line.append(f"{item} {mmss(float(np.median(vals)))} ({len(vals) / len(ms):.0%})")
        md += ["median timings (share of games): " + ", ".join(line), ""]
    out = data_root() / "interim" / "taxonomy"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.matchup}_Z_plan"
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(
        {"matchup": args.matchup, "race": "Z", "level": "plan", "names": names, "assignments": assignments},
        ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} games, {len(names)} plans; wrote {out / (stem + '.md')}")
    for k, n in enumerate(names):
        print(f"  {k:2d} {n:45s} {len(members.get(k, [])) / len(rows):6.1%}")


if __name__ == "__main__":
    main()
