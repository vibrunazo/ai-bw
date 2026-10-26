"""Every game of both eras labelled with its build style (issue #54), into the style catalogue
data/interim/taxonomy/v2/<matchup>_<race>.json (gary/styles.py reads it):

  TvZ Terran  the taxonomy's tech clusters (discover.py: 9,211 old games clustered, the rest none) extended to every game
              by a nearest-neighbour classifier on the same features (each key item's first time in 8 minutes; 5-fold
              accuracy 1.00 on the clustered games)
  TvZ Zerg    its plans (plans.py's rules: every game has one), with the ids and names of the plan taxonomy the macro
              models were trained with (a combination it didn't have is "other")

The macro dataset's index can then be restyled from it (ingest/macro_dataset.py --restyle-version v2) and the
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


def main() -> None:
    root = data_root() / "interim"
    tax = root / "taxonomy"
    out_dir = tax / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(root / "build_orders.jsonl", encoding="utf-8")]
    for matchup, race, labeller in (("TvZ", "T", label_terran), ("TvZ", "Z", label_zerg)):
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
