"""The humanlikeness classifier (docs/SELF_PLAY.md, "Before any RL"): can a model tell Gary's games
from the pros', minute by minute, from one player's side only? Its score is the AUC per game minute
(0.5: it can't tell), its top features are the next work items, and the raw pro-distance table keeps
it honest (a retrained classifier finds the next tell and hides progress).

Features per player-minute, from the replay re-simulated in OpenBW (the full state, the player's side
only: never the opponent's play, or it learns to spot Gary-Z), at the decision level, not clicks
(raw commands give it away on interface fingerprints). Sampled every 2 s:

  economy    workers, town halls, supply used and max, the share of the minute supply-blocked, the
             bank (minerals, gas; means), idle workers, workers on gas
  army       army supply, the count of each fighter type, the share of the army standing (no order),
             orders changed per fighter per minute, the share at home / in the field / at an enemy
             base, the share standing alone (no fighter of its own within 4 tiles), its spread, the
             army supply gone in the minute (dead, or in a bunker or a transport)
  buildings  the count of each production, tech and defense building at the minute's end (build
             timings: the minute a count first moves)
  threats    the share of the minute with enemy fighters near the player's buildings, and of those
             with its own fighters met them; the seconds from an attack starting (enemy fighters
             near its buildings after none for 10 s) to its first fighter there

Gradient-boosted trees as train/stim.py; out-of-fold scores, 5 folds by game. Pros: the race's side of
pro games of 10 minutes or more on Gary's three maps (Python, Fighting Spirit, Heartbreak Ridge),
their features cached in data/interim/humanlike/.

    python -m eval.humanlike --gary "league=runs/garyz_accept/*/*.rep" "human=runs/live/*human*.rep"
    python -m eval.humanlike --race Z --gary "runs/garyz_accept/*/*.rep"            # Gary-Z's side
    python -m eval.humanlike --gary "runs/new/*.rep" --frozen runs/humanlike/T_<first>/model.pkl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import random
import re
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority
from gary import terran as T
from gary import zerg as Z

SAMPLE = 48                     # frames between samples (2 s)
MINUTES = 20                    # the first this many minutes of a game
PRO_GAMES = 300                 # pro games, a third per map
PRO_MIN_S = 600
MAPS = ("python", "fighting spirit", "heartbreak")
IDLE = {2, 3}                   # Guard, PlayerGuard: standing
GAS = {81, 82, 83, 84}
NEUTRAL = 11
ALONE = 4 * 32
HOME = 15 * 32
THREAT = 12 * 32
MET = 8 * 32
QUIET = 5                       # samples (10 s) with no threat before a new attack starts
SIEGED = 30                     # Siege Tank in siege mode
BUCKETS = ((2, 4), (5, 7), (8, 10), (11, 14), (15, 20))

RACES = {
    "T": {"worker": T.SCV, "halls": {T.CC},
          "fighters": {T.MARINE: "marine", T.MEDIC: "medic", T.FIREBAT: "firebat", T.GHOST: "ghost",
                       T.VULTURE: "vulture", T.TANK: "tank", SIEGED: "tank", T.GOLIATH: "goliath",
                       T.WRAITH: "wraith", T.VESSEL: "vessel", T.VALKYRIE: "valkyrie", T.BC: "bc"},
          "others": {T.DROPSHIP: "dropship"},
          "buildings": {T.RAX: "barracks", T.FACTORY: "factory", T.STARPORT: "starport", T.ACADEMY: "academy",
                        T.EBAY: "ebay", T.ARMORY: "armory", T.TURRET: "turret", T.BUNKER: "bunker",
                        T.COMSAT: "comsat", T.REFINERY: "refinery", T.SCIENCE_FACILITY: "science_facility",
                        T.MACHINE_SHOP: "machine_shop", T.CONTROL_TOWER: "control_tower"},
          "supply": lambda t: T.SUPPLY.get(5 if t == SIEGED else t, 0)},
    "Z": {"worker": Z.DRONE, "halls": set(Z.TOWN_HALLS),
          "fighters": {Z.ZERGLING: "zergling", Z.HYDRALISK: "hydralisk", Z.MUTALISK: "mutalisk",
                       Z.SCOURGE: "scourge", Z.LURKER: "lurker", Z.ULTRALISK: "ultralisk", Z.DEFILER: "defiler",
                       Z.QUEEN: "queen", Z.GUARDIAN: "guardian", Z.DEVOURER: "devourer"},
          "others": {Z.OVERLORD: "overlord", Z.LARVA: "larva"},
          "buildings": {Z.POOL: "pool", Z.EXTRACTOR: "extractor", Z.DEN: "hydralisk_den", Z.SPIRE: "spire",
                        Z.EVO: "evo", Z.CREEP: "creep", Z.SUNKEN: "sunken", Z.SPORE: "spore",
                        Z.QUEENS_NEST: "queens_nest", Z.DEFILER_MOUND: "defiler_mound",
                        Z.ULTRA_CAVERN: "ultralisk_cavern", Z.GREATER_SPIRE: "greater_spire"},
          "supply": lambda t: Z.SUPPLY.get(t, 0) / (2 if t in Z.PAIRS else 1)},
}
FIGHTERS_ANY = set(RACES["T"]["fighters"]) | set(RACES["Z"]["fighters"])


def _names(race: str) -> list[str]:
    r = RACES[race]
    kinds = sorted(set(r["fighters"].values()) | set(r["others"].values()))
    return (["minute", "workers", "halls", "supply_used", "supply_max", "blocked", "minerals", "gas",
             "workers_idle", "workers_gas", "army_supply", "army_standing", "orders_per_fighter",
             "army_home", "army_field", "army_enemy", "army_alone", "army_spread", "army_gone",
             "threatened", "threat_met", "response_s"]
            + [f"n_{k}" for k in kinds] + [f"b_{k}" for k in sorted(set(r["buildings"].values()))])


def gary_end(path: str) -> int | None:
    """The frame a Gary game ended: the last time in its log (eval/vs_rush.py writes one beside each
    replay). OpenBW plays on past a replay's last command, the players idle: minutes 13-20 of a
    12-minute game banked 8,000 minerals no one spent."""
    log = Path(path).with_suffix(".log")
    if not log.exists():
        return None
    times = re.findall(r"^\s*(\d+):(\d\d)\s", log.read_text(encoding="utf-8", errors="replace"), re.M)
    return max((int(m) * 60 + int(s)) * 24 for m, s in times) if times else None


def features(path: str, race: str, minutes: int = MINUTES, end: int | None = None) -> list[dict]:
    """One row per full minute the player is in the game (from minute 1), up to the replay's end."""
    from gary.env import Game
    R = RACES[race]
    names = _names(race)
    rows: list[dict] = []
    g = Game.scenario(path)
    try:
        slot = None
        prev_order: dict[int, int] = {}
        prev_fight: dict[int, float] = {}
        acc = _fresh()
        threat_run, quiet, start, met_yet = False, QUIET, 0, False
        last = min(minutes * 1440, end if end is not None else minutes * 1440)
        while g.frame + SAMPLE <= last:
            try:
                g.step(SAMPLE)
            except Exception:
                break
            o = g.observe()
            units = o["units"]
            if slot is None:
                slot = next((u["owner"] for u in units if u["type"] in R["halls"]), None)
                if slot is None:
                    continue
            me = next((p for p in o["players"] if p["slot"] == slot), None)
            mine = [u for u in units if u["owner"] == slot]
            if me is None or not any(u["type"] in R["halls"] or 106 <= u["type"] <= 175 for u in mine):
                break                                          # (the player is out)
            foes = [u for u in units if u["owner"] not in (slot, NEUTRAL)]
            workers = [u for u in mine if u["type"] == R["worker"] and u["completed"]]
            fight = [u for u in mine if u["type"] in R["fighters"] and u["completed"]]
            buildings = [u for u in mine if 106 <= u["type"] <= 175 or u["type"] in R["halls"]]
            acc["n"] += 1
            acc["blocked"] += me["supply_used"] >= me["supply_max"] and me["supply_max"] < 200
            acc["minerals"] += me["minerals"]
            acc["gas"] += me["gas"]
            acc["workers_idle"] += sum(1 for u in workers if u["order"] in IDLE) / max(1, len(workers))
            acc["workers_gas"] += sum(1 for u in workers if u["order"] in GAS) / max(1, len(workers))
            # the army
            now = {u["tag"]: R["supply"](u["type"]) for u in fight}
            acc["army_gone"] += sum(s for t, s in prev_fight.items() if t not in now)
            prev_fight = now
            if fight:
                acc["fight_n"] += 1
                acc["army_standing"] += sum(1 for u in fight if u["order"] in IDLE) / len(fight)
                same = [u for u in fight if u["tag"] in prev_order]
                acc["changes"] += sum(1 for u in same if prev_order[u["tag"]] != u["order"])
                acc["fighter_samples"] += len(same)
                xy = np.array([(u["x"], u["y"]) for u in fight], np.float32)
                own = np.array([(u["x"], u["y"]) for u in mine if u["type"] in R["halls"]] or [(1e9, 1e9)], np.float32)
                enemy = np.array([(u["x"], u["y"]) for u in foes if u["type"] in (RACES["T"]["halls"] | RACES["Z"]["halls"])]
                                 or [(1e9, 1e9)], np.float32)
                d_own = np.sqrt(((xy[:, None] - own[None]) ** 2).sum(-1)).min(1)
                d_enemy = np.sqrt(((xy[:, None] - enemy[None]) ** 2).sum(-1)).min(1)
                home = d_own < HOME
                at_enemy = ~home & (d_enemy < HOME)
                acc["army_home"] += home.mean()
                acc["army_enemy"] += at_enemy.mean()
                acc["army_field"] += (~home & ~at_enemy).mean()
                d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
                np.fill_diagonal(d, 1e9)
                acc["army_alone"] += (d.min(1) > ALONE).mean() if len(fight) > 1 else 1.0
                acc["army_spread"] += float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1)).mean()) / 32
            prev_order = {u["tag"]: u["order"] for u in fight}
            # threats: enemy fighters near its buildings, and its answer
            bxy = np.array([(u["x"], u["y"]) for u in buildings] or [(1e9, 1e9)], np.float32)
            raiders = [u for u in foes if u["type"] in FIGHTERS_ANY]
            near = []
            if raiders:
                rxy = np.array([(u["x"], u["y"]) for u in raiders], np.float32)
                near_mask = np.sqrt(((rxy[:, None] - bxy[None]) ** 2).sum(-1)).min(1) < THREAT
                near = rxy[near_mask]
            if len(near):
                acc["threatened"] += 1
                met = bool(fight) and bool((np.sqrt(((np.array([(u["x"], u["y"]) for u in fight], np.float32)[:, None]
                                                     - near[None]) ** 2).sum(-1)).min() < MET))
                acc["met"] += met
                if not threat_run and quiet >= QUIET:
                    threat_run, start, met_yet = True, g.frame, False
                if threat_run and met and not met_yet:
                    met_yet = True
                    acc["responses"].append((g.frame - start) / 24)
                quiet = 0
            else:
                quiet += 1
                if threat_run and quiet >= QUIET:
                    if not met_yet:                            # (never met: as long as it lasted)
                        acc["responses"].append((g.frame - start) / 24)
                    threat_run = False
            if g.frame % 1440 == 0:                            # the minute's row
                rows.append(_row(g.frame // 1440, acc, me, mine, workers, fight, R, names))
                acc = _fresh()
    finally:
        g.close()
    return rows


def _fresh() -> dict:
    return {"n": 0, "fight_n": 0, "blocked": 0, "minerals": 0, "gas": 0, "workers_idle": 0, "workers_gas": 0,
            "army_gone": 0, "army_standing": 0, "changes": 0, "fighter_samples": 0, "army_home": 0,
            "army_enemy": 0, "army_field": 0, "army_alone": 0, "army_spread": 0, "threatened": 0, "met": 0,
            "responses": []}


def _row(minute: int, acc: dict, me: dict, mine: list, workers: list, fight: list, R: dict, names: list) -> dict:
    n, fn = max(1, acc["n"]), acc["fight_n"]
    nan = float("nan")
    row = {k: 0.0 for k in names}
    row.update(minute=minute, workers=len(workers), halls=sum(1 for u in mine if u["type"] in R["halls"]),
               supply_used=me["supply_used"], supply_max=me["supply_max"], blocked=acc["blocked"] / n,
               minerals=acc["minerals"] / n, gas=acc["gas"] / n, workers_idle=acc["workers_idle"] / n,
               workers_gas=acc["workers_gas"] / n, army_supply=sum(R["supply"](u["type"]) for u in fight),
               army_gone=acc["army_gone"],
               army_standing=acc["army_standing"] / fn if fn else nan,
               orders_per_fighter=acc["changes"] / acc["fighter_samples"] * (1440 / SAMPLE) if acc["fighter_samples"] else nan,
               army_home=acc["army_home"] / fn if fn else nan, army_field=acc["army_field"] / fn if fn else nan,
               army_enemy=acc["army_enemy"] / fn if fn else nan, army_alone=acc["army_alone"] / fn if fn else nan,
               army_spread=acc["army_spread"] / fn if fn else nan, threatened=acc["threatened"] / n,
               threat_met=acc["met"] / acc["threatened"] if acc["threatened"] else nan,
               response_s=st.mean(acc["responses"]) if acc["responses"] else nan)
    for u in mine:
        if u["type"] in R["fighters"] and u["completed"]:
            row[f"n_{R['fighters'][u['type']]}"] += 1
        elif u["type"] in R["others"]:
            row[f"n_{R['others'][u['type']]}"] += 1
        elif u["type"] in R["buildings"]:
            row[f"b_{R['buildings'][u['type']]}"] += 1
    return row


def _job(args: tuple) -> tuple[str, list[dict] | None]:
    path, race, end = args
    try:
        return path, features(path, race, end=end)
    except Exception:
        return path, None


def clean_replays() -> dict[str, int]:
    """Pro replays that play the same in OpenBW (resim_index.jsonl: ran, no desync, the frames match;
    as ingest/macro_dataset.py): a desynced one goes another way (a pro with 18 SCVs, no army and no
    gas at 10:00). {rel_path: the frame it ends}"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    from inventory import data_root
    out = {}
    for line in open(data_root() / "interim" / "resim_index.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("ok") and r.get("desync_minute") is None and r.get("frames_match") is not False:
            out[r["rel_path"]] = r.get("end_frame") or 0
    return out


def pro_rows(race: str, games: int, workers: int) -> dict[str, list[dict]]:
    """{rel_path: rows} for pro games on Gary's maps (cached per game)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    from inventory import data_root
    cache = data_root() / "interim" / "humanlike" / f"pro_{race}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            r = json.loads(line)
            have[r["rel_path"]] = r["rows"]
    clean = clean_replays()
    recs = [json.loads(line) for line in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["rel_path"] in clean]
    by_map: dict[str, list] = {m: [] for m in MAPS}
    for r in recs:
        if r["matchup"] != "TvZ" or r["race"] != race or r.get("duration_s", 0) < PRO_MIN_S:
            continue
        name = re.sub(r"[^a-z ]", "", r.get("map", "").lower())
        for m in MAPS:
            if m in name:
                by_map[m].append(r["rel_path"])
    chosen = []
    for m in MAPS:
        random.Random(0).shuffle(by_map[m])
        chosen += by_map[m][:games // len(MAPS)]
    todo = [p for p in chosen if p not in have]
    if todo:
        with ProcessPoolExecutor(workers) as pool, open(cache, "a", encoding="utf-8") as out:
            jobs = [(str(data_root() / "raw" / p), race, clean[p] or None) for p in todo]
            for (path, rows), rel in zip(monitor.track(pool.map(_job, jobs), len(jobs), f"pro {race} games"), todo):
                rows = rows or []
                have[rel] = rows
                out.write(json.dumps({"rel_path": rel, "rows": rows}) + "\n")
    return {p: have[p] for p in chosen if have.get(p)}


def gap_score(Xs, Xp, cap: float = 5.0) -> list:
    """Per minute range (BUCKETS, then all minutes): (the mean standardized gap, the share of features
    with Gary's median inside the pros' interquartile range), or None where there are too few rows.
    A feature's spread: the pros' interquartile range, else their standard deviation, else 1."""
    out = []
    for a, b in list(BUCKETS) + [(0, 99)]:
        g = Xs[(Xs[:, 0] >= a) & (Xs[:, 0] <= b)]
        q = Xp[(Xp[:, 0] >= a) & (Xp[:, 0] <= b)]
        if len(g) < 5 or len(q) < 20:
            out.append(None)
            continue
        zs, inside = [], []
        for k in range(1, Xs.shape[1]):
            gv, qv = g[:, k][~np.isnan(g[:, k])], q[:, k][~np.isnan(q[:, k])]
            if len(gv) < 3 or len(qv) < 10:
                continue
            lo, hi = np.percentile(qv, [25, 75])
            spread = (hi - lo) or qv.std() or 1.0
            zs.append(min(cap, abs(gv.mean() - qv.mean()) / spread))
            inside.append(lo <= np.median(gv) <= hi)
        out.append((float(np.mean(zs)), float(np.mean(inside))) if zs else None)
    return out


def matrix(games: dict[str, list[dict]], names: list[str]):
    X, gid, keys = [], [], list(games)
    for k, key in enumerate(keys):
        for r in games[key]:
            X.append([r.get(n, float("nan")) for n in names])
            gid.append(k)
    return np.array(X, np.float64), np.array(gid), keys


def auc_by_minute(y, p, minute) -> dict[int, float]:
    from sklearn.metrics import roc_auc_score
    out = {}
    for m in sorted(set(minute.tolist())):
        sel = minute == m
        if len(set(y[sel].tolist())) == 2 and min(y[sel].sum(), (1 - y[sel]).sum()) >= 5:
            out[int(m)] = float(roc_auc_score(y[sel], p[sel]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", choices=("T", "Z"), default="T", help="whose side is judged")
    ap.add_argument("--gary", nargs="+", required=True,
                    help="globs of Gary's replays, each optionally labeled by its opponent: label=glob "
                         "(league, replayed, human; unlabeled: league)")
    ap.add_argument("--pros", type=int, default=PRO_GAMES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frozen", help="also score Gary's games with this earlier model.pkl (did the old tells go?)")
    ap.add_argument("--tells", type=int, default=12, help="tells found by dropping the top one in turn")
    ap.add_argument("--out", default="runs/humanlike")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    names = _names(args.race)
    label_of = {}
    for pat in args.gary:
        label, _, pat = pat.rpartition("=") if "=" in pat else ("league", "", pat)
        for p in glob.glob(pat):
            label_of.setdefault(p, label)
    paths = sorted(label_of)
    with ProcessPoolExecutor(args.workers) as pool:
        got = list(monitor.track(pool.map(_job, [(p, args.race, gary_end(p)) for p in paths]), len(paths),
                                 f"Gary {args.race} games"))
    gary = {p: rows for p, rows in got if rows}
    pros = pro_rows(args.race, args.pros, args.workers)
    print(f"{len(gary)} Gary games ({sum(map(len, gary.values()))} minutes), "
          f"{len(pros)} pro games ({sum(map(len, pros.values()))} minutes), side {args.race}")
    Xg, gg, gkeys = matrix(gary, names)
    labels = np.array([label_of[gkeys[k]] for k in gg])
    Xp, gp, _ = matrix(pros, names)
    score = {lab: gap_score(Xg[labels == lab], Xp) for lab in sorted(set(labels.tolist()))}
    score["all"] = gap_score(Xg, Xp)
    X = np.vstack([Xg, Xp])
    y = np.r_[np.ones(len(Xg)), np.zeros(len(Xp))]
    group = np.r_[gg, gp + gg.max() + 1]
    minute = X[:, 0]

    clf = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                 early_stopping=False, random_state=0)
    oof = np.zeros(len(y))
    imp = np.zeros(len(names))
    for train, test in GroupKFold(5).split(X, y, group):
        m = clf().fit(X[train], y[train])
        oof[test] = m.predict_proba(X[test])[:, 1]
        r = permutation_importance(m, X[test], y[test], scoring="roc_auc", n_repeats=3, random_state=0)
        imp += r.importances_mean / 5
    auc = float(roc_auc_score(y, oof))
    per_min = auc_by_minute(y, oof, minute)
    model = clf().fit(X, y)
    # the tells in order: shuffling one feature hides little while others tell as well (bunkers took
    # all the importance at AUC 0.999), so the top one is dropped and the model trained again, and so
    # on: each step's feature, and the AUC left without it and the ones before it
    tells, keep = [], [k for k in range(len(names)) if names[k] != "minute"]
    quick = lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31,
                                                   early_stopping=False, random_state=0)
    for _ in monitor.track(range(args.tells), args.tells, "tells"):
        cols = [0] + keep
        p_, imp_ = np.zeros(len(y)), np.zeros(len(cols))
        for train, test in GroupKFold(3).split(X, y, group):
            m = quick().fit(X[train][:, cols], y[train])
            p_[test] = m.predict_proba(X[test][:, cols])[:, 1]
            imp_ += permutation_importance(m, X[test][:, cols], y[test], scoring="roc_auc", n_repeats=2,
                                           random_state=0).importances_mean
        top = cols[1 + int(np.argmax(imp_[1:]))]
        tells.append((names[top], float(roc_auc_score(y, p_))))
        keep.remove(top)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"{args.race}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "names": names, "race": args.race, "auc": auc, "per_minute": per_min,
                     "gap_score": score}, f)

    lines = [f"# Humanlikeness: Gary's {args.race} side against the pros'", "",
             f"{len(gary)} Gary games, {len(pros)} pro games (TvZ, 10 min or more, Python / Fighting Spirit / "
             f"Heartbreak Ridge); out-of-fold, 5 folds by game.", "",
             f"**AUC {auc:.3f}** (0.5: it can't tell Gary from the pros)", "",
             "| minute | " + " | ".join(str(m) for m in per_min) + " |",
             "|---|" + "---|" * len(per_min),
             "| AUC | " + " | ".join(f"{v:.2f}" for v in per_min.values()) + " |", "",
             "## The gap score (moves while the AUC saturates)", "",
             "Mean standardized gap: each feature's Gary mean less the pros', over the pros' spread (their "
             "interquartile range; capped at 5), averaged over the features: 0 is the pros. Inside: the share "
             "of features whose Gary median lies within the pros' interquartile range. By opponent: the "
             "pros faced pros; human games are the only rows where Gary's opponent is real.", "",
             "| Gary's games | games | minutes | " + " | ".join(f"{a}-{b} min" for a, b in BUCKETS) + " | all |",
             "|---|---|---|" + "---|" * (len(BUCKETS) + 1)]
    for lab, sc in score.items():
        n_games = len(gary) if lab == "all" else sum(1 for p in gary if label_of[p] == lab)
        n_rows = len(Xg) if lab == "all" else int((labels == lab).sum())
        lines.append(f"| {lab} | {n_games} | {n_rows} | " + " | ".join(
            "-" if v is None else f"{v[0]:.2f} ({v[1]:.0%} inside)" for v in sc) + " |")
    lines += [""]
    if args.frozen:
        old = pickle.load(open(args.frozen, "rb"))
        ok = old["names"] == names
        if ok:
            pf = old["model"].predict_proba(X)[:, 1]
            lines += [f"Frozen classifier ({args.frozen}, AUC then {old['auc']:.3f}): AUC now "
                      f"{roc_auc_score(y, pf):.3f} on these games (its pros were in its training)", ""]
        else:
            lines += ["Frozen classifier: its features differ, not scored", ""]
    lines += ["## The tells in order (each dropped in turn; the AUC with it and every later one still in)", "",
              "| step | feature | AUC with it |", "|---|---|---|"]
    lines += [f"| {k + 1} | {n} | {a:.3f} |" for k, (n, a) in enumerate(tells)]
    lines += [""]
    # each feature alone, per minute range: how far apart Gary's values and the pros' are (the AUC of
    # the feature as a score, folded: 0 the same spread, 1 no overlap). Not hidden by the others as the
    # combined classifier's importance is: the gaps to close, biggest first
    def sep(k, a, b):
        g = Xg[(Xg[:, 0] >= a) & (Xg[:, 0] <= b), k]
        q = Xp[(Xp[:, 0] >= a) & (Xp[:, 0] <= b), k]
        g, q = g[~np.isnan(g)], q[~np.isnan(q)]
        if len(g) < 10 or len(q) < 10:
            return float("nan")
        return abs(2 * roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(q))], np.r_[g, q]) - 1)

    def med(Xs, k, a, b):
        v = Xs[(Xs[:, 0] >= a) & (Xs[:, 0] <= b), k]
        v = v[~np.isnan(v)]
        return f"{np.median(v):.3g}" if len(v) else "-"
    seps = {k: [sep(k, a, b) for a, b in BUCKETS] for k in range(1, len(names))}
    order = sorted(seps, key=lambda k: -np.nanmax(seps[k]) if not np.all(np.isnan(seps[k])) else 0)
    lines += ["## Gary against the pros, feature by feature (biggest gap first)", "",
              "Medians, Gary / pros, per minute range; in brackets how far apart (0: the same spread, 1: no overlap).", "",
              "| feature | " + " | ".join(f"{a}-{b} min" for a, b in BUCKETS) + " |",
              "|---|" + "---|" * len(BUCKETS)]
    for k in order:
        lines.append(f"| {names[k]} | " + " | ".join(
            f"{med(Xg, k, a, b)} / {med(Xp, k, a, b)}" + ("" if np.isnan(v) else f" ({v:.2f})")
            for (a, b), v in zip(BUCKETS, seps[k])) + " |")
    lines += ["", "Medians per minute range; every feature, with means, in table.json."]
    table = {names[k]: {f"{a}-{b}": {"gary_median": med(Xg, k, a, b), "pro_median": med(Xp, k, a, b),
                                     "gary_mean": float(np.nanmean(Xg[(Xg[:, 0] >= a) & (Xg[:, 0] <= b), k]))
                                     if ((Xg[:, 0] >= a) & (Xg[:, 0] <= b)).any() else None,
                                     "pro_mean": float(np.nanmean(Xp[(Xp[:, 0] >= a) & (Xp[:, 0] <= b), k]))
                                     if ((Xp[:, 0] >= a) & (Xp[:, 0] <= b)).any() else None}
                         for a, b in BUCKETS} | {"importance": float(imp[k]), "apart": seps.get(k)}
             for k in range(len(names))}
    (out / "table.json").write_text(json.dumps(table, indent=1))
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
