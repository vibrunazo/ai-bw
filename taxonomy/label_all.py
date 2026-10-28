"""Every game of both eras labelled with its build style (issue #54), into the style catalogue
data/interim/taxonomy/v2/<matchup>_<race>.json (gary/styles.py reads it):

  TvZ Terran  the taxonomy's tech clusters (discover.py: 9,211 old games clustered, the rest none) extended to every game
              by a nearest-neighbour classifier on the same features (each key item's first time in 8 minutes; 5-fold
              accuracy 1.00 on the clustered games)
  TvZ Zerg    its plans (plans.py's rules: every game has one), with the ids and names of the plan taxonomy the macro
              models were trained with (a combination it didn't have is "other")
  PvT, PvZ,   Protoss plans by the same kind of rules: the opening (which of the first Gateway, the second Nexus and
  PvP         the Forge came first) and the first tech by P_TECH_BY (Robotics Facility, Citadel of Adun or Stargate);
              a combination under MIN_SHARE of the modern games is "other"
  PvT, TvT    Terran plans: the opening (which of the second Command Center, the first Factory and the second Barracks
  Terran      came first) and the tech by T_PLAN_BY (a Starport before the second Factory, else the Factories, else bio)
  PvZ, ZvZ    the TvZ Zerg's plan rules (plans.py), ZvZ's pool-first openings split by the pool's time (ZVZ_NINE_POOL:
  Zerg        the 9-pool and the 12-pool or overpool, two humps in the modern games)

A race's matchups take their own id ranges (MATCHUP_FROM), so the shared macro model's style input means one build per
id across them.

The macro dataset's index can then be restyled from it (train/macro.py --styles v2) and the
macro models trained on every game's style. The v1 files (data/interim/taxonomy/*.json) are left as they are:
gz1's roll and per-style history read them.

    python taxonomy/label_all.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover import KEY_ITEMS, data_root, first_times   # noqa: E402
from plans import PRESSURE_BY, TECH_BY, lings_by, plan_of   # noqa: E402

WINDOW, NEVER = 480, 600.0
T_NAMES = {1: "standard bio", 2: "Factory and vultures", 3: "Factory and Starport", 4: "two-Barracks marines"}
P_TECH_BY = 8 * 60
MIN_SHARE = 0.01
# each race's matchups' first style id (the TvZ ids, from the taxonomies the models were trained with, start at 0)
MATCHUP_FROM = {("PvT", "P"): 0, ("PvZ", "P"): 20, ("PvP", "P"): 40, ("PvT", "T"): 20, ("TvT", "T"): 40,
                ("PvZ", "Z"): 20, ("ZvZ", "Z"): 40}
PVZ_FROM = MATCHUP_FROM[("PvZ", "P")]
T_PLAN_BY = 8 * 60
T_PLAN_ITEMS = {("Barracks", 1): "rax1", ("Barracks", 2): "rax2", ("Barracks", 3): "rax3", ("Command Center", 1): "cc2",
                ("Factory", 1): "fact1", ("Factory", 2): "fact2", ("Factory", 3): "fact3", ("Starport", 1): "port",
                ("Academy", 1): "academy"}
T_OPENINGS = (("cc2", "expansion first"), ("fact1", "factory first"), ("rax2", "two barracks"))
ZVZ_NINE_POOL = 72                       # s: a pool-first ZvZ pool before this is a 9-pool
P_OPENINGS = (("gate1", "gateway first"), ("nexus2", "nexus first"), ("forge", "forge first"))
P_TECHS = (("robo", "robotics first"), ("citadel", "citadel first"), ("stargate", "stargate first"))


def is_modern(r: dict) -> bool:
    return r.get("source") == "cwal" or "cwal" in str(r.get("rel_path"))


def terran_vec(items: list[dict]) -> np.ndarray:
    t = first_times(items, "T", WINDOW)
    return np.array([min(t.get(f, NEVER), NEVER) for f, _, _ in KEY_ITEMS["T"]], np.float32) / 60.0


def label_terran(rows: list[dict], tax: Path) -> tuple[dict, dict]:
    from sklearn.neighbors import KNeighborsClassifier
    old = json.loads((tax / "TvZ_T_tech.json").read_text(encoding="utf-8"))["assignments"]
    key = lambda r: f"{r['rel_path']}|{r['player']}"
    train = [(terran_vec(r["items"]), old[key(r)]) for r in rows if old.get(key(r), -1) >= 0]
    knn = KNeighborsClassifier(15).fit(np.array([v for v, _ in train]), np.array([k for _, k in train]))
    labels = knn.predict(np.array([terran_vec(r["items"]) for r in rows]))
    return {key(r): int(k) for r, k in zip(rows, labels)}, dict(T_NAMES)


def label_zerg(rows: list[dict], tax: Path) -> tuple[dict, dict]:
    old = json.loads((tax / "TvZ_Z_plan.json").read_text(encoding="utf-8"))
    names = old["names"]
    ids = {n: i for i, n in enumerate(names)}
    other = ids.get("other", len(names) - 1)
    out = {}
    for r in rows:
        t = first_times(r["items"], "Z", TECH_BY)
        name = ", ".join(plan_of(t, lings_by(r, PRESSURE_BY), r["duration_s"]))
        out[f"{r['rel_path']}|{r['player']}"] = ids.get(name, other)
    return out, {i: n for i, n in enumerate(names)}


def protoss_plan(t: dict[str, float], duration: float) -> str:
    never = 9e9
    opening = min(P_OPENINGS, key=lambda o: t.get(o[0], never))[1]
    techs = [(t[k], name) for k, name in P_TECHS if k in t and t[k] <= P_TECH_BY]
    tech = min(techs)[1] if techs else "no tech" if duration >= P_TECH_BY else "undecided"
    return f"{opening}, {tech}"


def terran_plan(r: dict) -> str:
    seen, t, never = Counter(), {}, 9e9
    for i in r["items"]:
        if i["conf"] in ("confirmed", "ordered") and i["t"] <= T_PLAN_BY:
            seen[i["name"]] += 1
            feat = T_PLAN_ITEMS.get((i["name"], seen[i["name"]]))
            if feat and feat not in t:
                t[feat] = i["t"]
    opening = min(T_OPENINGS, key=lambda o: t.get(o[0], never))[1]
    factories = sum(k in t for k in ("fact1", "fact2", "fact3"))
    if t.get("port", never) < t.get("fact2", never):
        tech = "starport first"
    elif factories:
        tech = ("one factory", "two factories", "three factories")[factories - 1]
    elif "rax3" in t or "academy" in t:
        tech = "bio"
    else:
        tech = "no tech" if r["duration_s"] >= T_PLAN_BY else "undecided"
    return f"{opening}, {tech}"


def zerg_plan(r: dict, matchup: str) -> str:
    t = first_times(r["items"], "Z", TECH_BY)
    plan = list(plan_of(t, lings_by(r, PRESSURE_BY), r["duration_s"]))
    if matchup == "ZvZ" and plan[0] == "pool first":
        plan[0] = "9 pool" if t["pool"] < ZVZ_NINE_POOL else "12 pool"
    return ", ".join(plan)


def label_plans(rows: list[dict], plan_of_row, first: int) -> tuple[dict, dict]:
    """Each game's plan by the rules; the plans of at least MIN_SHARE of the modern games named from first on, the
    rest "other"."""
    plan = {f"{r['rel_path']}|{r['player']}": plan_of_row(r) for r in rows}
    modern = Counter(plan[f"{r['rel_path']}|{r['player']}"] for r in rows if is_modern(r))
    n = max(1, sum(modern.values()))
    kept = [p for p, c in modern.most_common() if c / n >= MIN_SHARE]
    names = {first + i: p for i, p in enumerate(kept)}
    names[first + len(kept)] = "other"
    ids = {p: k for k, p in names.items()}
    return {k: ids.get(p, first + len(kept)) for k, p in plan.items()}, names


def label_protoss(rows: list[dict], tax: Path, first: int = 0) -> tuple[dict, dict]:
    return label_plans(rows, lambda r: protoss_plan(first_times(r["items"], "P", P_TECH_BY), r["duration_s"]), first)


LABELLERS = {
    ("TvZ", "T"): label_terran, ("TvZ", "Z"): label_zerg,
    ("PvT", "P"): lambda rs, tax: label_protoss(rs, tax, MATCHUP_FROM[("PvT", "P")]),
    ("PvZ", "P"): lambda rs, tax: label_protoss(rs, tax, MATCHUP_FROM[("PvZ", "P")]),
    ("PvP", "P"): lambda rs, tax: label_protoss(rs, tax, MATCHUP_FROM[("PvP", "P")]),
    ("PvT", "T"): lambda rs, tax: label_plans(rs, terran_plan, MATCHUP_FROM[("PvT", "T")]),
    ("TvT", "T"): lambda rs, tax: label_plans(rs, terran_plan, MATCHUP_FROM[("TvT", "T")]),
    ("PvZ", "Z"): lambda rs, tax: label_plans(rs, lambda r: zerg_plan(r, "PvZ"), MATCHUP_FROM[("PvZ", "Z")]),
    ("ZvZ", "Z"): lambda rs, tax: label_plans(rs, lambda r: zerg_plan(r, "ZvZ"), MATCHUP_FROM[("ZvZ", "Z")]),
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="MATCHUP_RACE", help="only these catalogues, e.g. TvT_T ZvZ_Z")
    args = ap.parse_args()
    root = data_root() / "interim"
    tax = root / "taxonomy"
    out_dir = tax / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(root / "build_orders.jsonl", encoding="utf-8")]
    for (matchup, race), labeller in LABELLERS.items():
        if args.only and f"{matchup}_{race}" not in args.only:
            continue
        rs = [r for r in rows if r.get("matchup") == matchup and r.get("race") == race]
        assignments, names = labeller(rs, tax)
        share = lambda sel: {k: v / max(1, len(sel)) for k, v in Counter(assignments[f"{r['rel_path']}|{r['player']}"] for r in sel).items()}
        modern = [r for r in rs if is_modern(r)]
        d = {"matchup": matchup, "race": race, "version": "v2", "names": names, "shares": share(modern),
             "shares_all": share(rs), "games": len(rs), "modern_games": len(modern), "assignments": assignments}
        (out_dir / f"{matchup}_{race}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        print(f"{matchup} {race}: {len(rs)} games ({len(modern)} modern) labelled -> {out_dir / f'{matchup}_{race}.json'}")
        for k in sorted(names):
            print(f"  {k:3d} {names[k]:50s} modern {d['shares'].get(k, 0):5.1%}  all {d['shares_all'].get(k, 0):5.1%}")


if __name__ == "__main__":
    main()
