"""Gary's build judged per style: told style X, does Gary play X, and like the pros who play X on the same maps (an
overall comparison mixes styles and map pools: told style 4, gt1dev played standard bio in 89% of its games). The Terran styles are the taxonomy's tech clusters (taxonomy/discover.py, 9,221 labelled player-games, none of
them modern); a nearest-neighbour classifier on the same features (each key item's first time in 8 minutes) labels
the modern pros' games and Gary's. The Zerg styles are its plans (taxonomy/plans.py: rules on the opening, the
hatcheries and the first tech), which label Gary-Z's games as they label the pros'. Then per style: the label of
Gary's games told that style (does it play it), and the key items' timings against the modern pros of that style on
the same ladder maps.

    python -m eval.style_adherence SR144a,SR144b [UC143a,UC143b]    # rounds of a Terran manifest, comma-joined
    python -m eval.style_adherence --race Z ZG153a,ZG153b             # the rounds' Zerg
    python -m eval.style_adherence --race P --matchup PvT PV160a        # the Protoss (its plans: taxonomy/label_all.py)"""
import glob, hashlib, json, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ingest")); sys.path.insert(0, str(REPO / "taxonomy"))
from inventory import data_root, find_screp
from discover import KEY_ITEMS, first_times
from plans import PRESSURE_BY, TECH_BY, lings_by, plan_of
from build_orders import extract_player
from cmdcache import P, load_game

WINDOW, NEVER = 480, 600.0
RACE = sys.argv[sys.argv.index("--race") + 1] if "--race" in sys.argv else "T"
MATCHUP = sys.argv[sys.argv.index("--matchup") + 1] if "--matchup" in sys.argv else "TvZ"
FEATS = [f for f, _, _ in KEY_ITEMS[RACE]]
SHOW = {"T": ["gas1", "cc2", "factory", "shop", "starport", "academy", "rax2", "bunker"],
        "Z": ["pool", "hatch2", "gas1", "hatch3", "lair", "spire", "muta", "den", "lurker_aspect", "gas2"],
        "P": ["gate1", "forge", "nexus2", "gas1", "core", "gate2", "robo", "citadel", "stargate", "reaver", "dt"]}[RACE]


def vec(items):
    t = first_times(items, RACE, WINDOW)
    return np.array([min(t.get(f, NEVER), NEVER) for f in FEATS], np.float32) / 60.0, t


def zerg_plan(items, ids: dict[str, int]) -> int:
    """A Zerg game's plan id in the style catalogue (taxonomy/plans.py's rules; "other" for the plans it merged)."""
    t = first_times(items, "Z", TECH_BY)
    name = ", ".join(plan_of(t, lings_by({"items": items}, PRESSURE_BY), max((i["t"] for i in items), default=0)))
    return ids.get(name, ids["other"])


def told_style(rep: str, seed: int) -> int | None:
    """The style the game's Gary was told: its records' "style" decision (since 2026-10-26), else rebuilt from its
    manifest, map and seed (right only while the manifest's definition is the one the game played)."""
    rec = Path(rep.replace(".rep", ".records.jsonl"))
    if rec.exists():
        for line in open(rec, encoding="utf-8"):
            if '"layer": "style"' in line:
                r = json.loads(line)
                if r.get("side") == RACE:
                    return int(r["choice"])
            elif '"f": ' in line and not line.startswith('{"f": 0'):
                break
    if RACE != "T":
        return None
    m = _match(rep, seed)
    return int(m.gary.macro.tracker.style) if m is not None and m.gary.macro.tracker.style is not None else None


def _match(rep: str, seed: int):
    """The style the game's Gary was told (built again from its manifest, map and seed: the strategy selector or the
    roll decides it at the start), or None."""
    import contextlib, io
    from eval.match import Match
    stem = Path(rep).stem
    manifest = stem.rsplit("_seed", 1)[0]
    from eval.league import ladder_maps
    path = ladder_maps().get(Path(rep).parent.name)
    if path is None:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, manifest, path, opponent="gz1")
    m.game.close()
    return m


def gary_items(rep: str, cache: Path):
    game = load_game(find_screp(None), Path(rep), hashlib.sha1(rep.encode()).hexdigest(), cache)
    t = next((p for p in game["players"] if not p["observer"] and p["race"] == RACE), None)
    if t is None:
        return None
    return extract_player([c for c in game["cmds"] if c[P] == t["id"]], RACE)[0]


def main():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from gary import styles
    NAMES = styles.catalog(MATCHUP, RACE)["names"]
    rows = [json.loads(l) for l in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("matchup") == MATCHUP and r.get("race") == RACE]
    if RACE == "P":
        sys.path.insert(0, str(REPO / "taxonomy"))
        from label_all import P_TECH_BY, protoss_plan
        ids = {n: k for k, n in NAMES.items()}
        label = lambda items: ids.get(protoss_plan(first_times(items, "P", P_TECH_BY), max((i["t"] for i in items), default=0)),
                                      max(NAMES))
    elif RACE == "Z":
        ids = {n: k for k, n in NAMES.items()}
        label = lambda items: zerg_plan(items, ids)
    else:
        lab = json.loads((data_root() / "interim" / "taxonomy" / "TvZ_T_tech.json").read_text(encoding="utf-8"))["assignments"]
        X, y = [], []
        for r in rows:
            k = lab.get(f"{r['rel_path']}|{r['player']}")
            if k is not None and k >= 0:
                X.append(vec(r["items"])[0]); y.append(k)
        X, y = np.array(X), np.array(y)
        knn = KNeighborsClassifier(15).fit(X, y)
        print(f"classifier: {len(y)} labelled games, styles {dict(Counter(y.tolist()))}, 5-fold accuracy "
              f"{cross_val_score(KNeighborsClassifier(15), X, y, cv=5).mean():.2f}")
        label = lambda items: int(knn.predict(vec(items)[0][None])[0])
    from eval.league import ladder_maps
    ladder = set(ladder_maps())
    modern = [r for r in rows if r.get("source") == "cwal" or "cwal" in str(r.get("rel_path"))]
    import re
    base = {re.sub(r"[^a-z]", "", m.lower().split(" ")[0]) for m in ladder}      # (attitude, backrooms, ...)
    norm = lambda name: re.sub(r"[^a-z]", "", str(name).lower())
    on_ladder = [r for r in modern if any(b_ in norm(r.get("map", "")) for b_ in base)]
    pros = defaultdict(list)
    for r in on_ladder:
        pros[label(r["items"])].append(vec(r["items"])[1])
    print(f"modern pros on the ladder maps: {len(on_ladder)} games; by the {'plans' if RACE == 'Z' else 'classifier'}: "
          + ", ".join(f"style {k} {len(v) / len(on_ladder):.0%}" for k, v in sorted(pros.items())))
    cache = Path(tempfile.mkdtemp())

    def timings(ts):
        out = []
        for f in SHOW:
            v = [t[f] for t in ts if f in t]
            out.append(f"{f} {np.median(v) / 60 if v else float('nan'):.1f} ({len(v) / max(1, len(ts)):.0%})")
        return ", ".join(out)
    if "--write-shares" in sys.argv:            # (the roll's weights: gary/agent/manifests.py rolled_style)
        n = sum(len(v) for v in pros.values())
        out = data_root() / "interim" / "taxonomy" / f"TvZ_{RACE}_modern_shares.json"
        out.write_text(json.dumps({"source": "modern pros on the ladder maps, eval/style_adherence.py", "games": n,
                                   "shares": {str(k): round(len(v) / n, 4) for k, v in sorted(pros.items())}}), encoding="utf-8")
        print(f"wrote {out}")
    for arg in [a for a in sys.argv[1:] if not a.startswith("--") and a not in (RACE, MATCHUP)]:
        reps = [p for r in arg.split(",") for p in sorted(glob.glob(str(REPO / "runs" / "league" / r / "*" / "*.rep")))]
        told = defaultdict(list)
        for rep in reps:
            seed = int(Path(rep).stem.split("seed")[-1])
            style = told_style(rep, seed)
            items = gary_items(rep, cache)
            if items is None:
                continue
            told[style].append((label(items), vec(items)[1]))
        print(f"== {arg}")
        for s, g in sorted(told.items()):
            plays = Counter(k for k, _ in g)
            print(f"  told style {s} ({NAMES.get(s, '?')}), {len(g)} games: plays as " +
                  ", ".join(f"style {k} {n / len(g):.0%}" for k, n in plays.most_common()))
            print(f"    Gary's first times (median minute, share of games by 8:00): {timings([t for _, t in g])}")
            if s in pros:
                print(f"    modern pros of style {s} on these maps ({len(pros[s])}):  {timings(pros[s])}")


if __name__ == "__main__":
    main()
