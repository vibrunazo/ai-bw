"""The cast dataset: when and where the pros cast a spell (gary/policy/cast.py, trained by train/cast.py).

For each game, the spell's casts come from two passes over the replay:
  - the re-simulator's unit commands (ingest/hotkey_dataset.py's pass): every order of the spell's order id, with
    the units selected and the point clicked (pros click a cast several times: the first click of a burst is the
    decision, the last is where it landed)
  - the game stepped in OpenBW every half second (as ingest/stim_dataset.py): a caster that lost the spell's energy
    in the half second, with an order of the spell on it in the seconds before, cast it (EMP and Feedback drain
    energy too, without the order)
From a minute before the player's first cast to 30 seconds after its last, each caster with the energy, in a fight
(a visible enemy unit within 12 tiles), is a sample of the when model: label 1 when it cast within the next second
(samples while a cast is under way are left out). The sample just before each cast gives the where model its
candidate spots, the one nearest the point cast at (within 2 tiles) labelled.

    python -m ingest.cast_dataset --spell storm --matchups PvZ PvT PvP --games 400 --parallel 4
    python -m ingest.cast_dataset --matchups PvZ --games 200 --report runs/league/ST184a   # Gary's against the pros'
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))

from eval import monitor, priority  # noqa: E402
from gary.policy.cast import (CAST_NAMES, CAST_WITHIN as AHEAD, RECENT, SAMPLE_EVERY, SPELLS, SPOT_NAMES, cast_features,  # noqa: E402
                              enemy_units, in_fight, spots)

BURST = 5 * 24                   # an order's clicks this close together are one cast
LEAD, TAIL = 60 * 24, 30 * 24    # samples from a minute before the first cast to 30 s after the last
UNTIL = 30 * 1440
SPOT_NEAR = 64                   # the spot nearest the cast point labelled, if this near


def _orders(path: str, spell) -> dict[int, list[tuple[int, set, int, int]]]:
    """slot -> the spell's orders: (frame, units selected, x, y)."""
    from hotkey_dataset import _resim
    lines = _resim(Path(path)) or []
    out: dict[int, list] = {}
    for r in lines:
        if r.get("type") == "ucmd" and r.get("order") == spell.order:
            out.setdefault(r["slot"], []).append((r["frame"], set(r["units"]), r["x"], r["y"]))
    return out


def _job(job: tuple) -> dict:
    path, end, spell_name = job
    spell = SPELLS[spell_name]
    empty = {"error": None, "X": [], "y": [], "S": [], "s_game": [], "s_label": [], "casts": 0, "cast_at": []}
    orders = _orders(path, spell)
    if not orders:
        return empty
    from gary.env import Game
    try:
        g = Game.scenario(path)
    except Exception as e:
        return {**empty, "error": str(e)}
    first = min(o[0] for os_ in orders.values() for o in os_)
    last = max(o[0] for os_ in orders.values() for o in os_)
    stop = min(end - AHEAD - SAMPLE_EVERY, last + TAIL, UNTIL)
    samples = []                 # (slot, tag, frame, features, spot points, spot rows)
    casts = []                   # (slot, tag, decided, landed, x, y)
    energy: dict[int, tuple[int, int]] = {}      # tag -> (energy, hp) at the last sample
    try:
        g.step(max(1, first - LEAD) - g.frame)
        while g.frame < stop:
            g.step(SAMPLE_EVERY)
            obs = g.observe()
            f = obs["frame"]
            for slot, os_ in orders.items():
                own = [u for u in obs["units"] if u["owner"] == slot and u["completed"]]
                enemies = enemy_units(obs["units"], slot)
                for u in own:
                    if u["type"] != spell.caster:
                        continue
                    e0, _ = energy.get(u["tag"], (u.get("energy") or 0, 0))
                    if e0 - (u.get("energy") or 0) >= spell.energy - 5:
                        mine = [o for o in os_ if u["tag"] in o[1] and f - SAMPLE_EVERY - 4 * 24 < o[0] <= f]
                        if mine:
                            burst = [o for o in os_ if u["tag"] in o[1] and mine[-1][0] - BURST <= o[0] <= mine[-1][0]]
                            casts.append((slot, u["tag"], burst[0][0], f, mine[-1][2], mine[-1][3]))
                recent = [(c[4], c[5]) for c in casts if c[0] == slot and f - RECENT <= c[3] <= f]
                for u in own:
                    if u["type"] != spell.caster or (u.get("energy") or 0) < spell.energy or not in_fight(u, enemies):
                        continue
                    before = energy.get(u["tag"])
                    x = cast_features(spell, u, own, enemies, f, recent, before[1] if before else None)
                    pts, rows = spots(spell, u, own, enemies, recent)
                    samples.append((slot, u["tag"], f, x, pts, rows))
                for u in own:
                    if u["type"] == spell.caster:
                        energy[u["tag"]] = (u.get("energy") or 0, u["hp"] + u.get("shields", 0))
    except Exception as e:
        return {**empty, "error": str(e)}
    finally:
        g.close()
    X, y, S, s_label = [], [], [], []
    by_tag: dict[int, list] = {}
    for c in casts:
        by_tag.setdefault(c[1], []).append(c)
    for slot, tag, f, x, pts, rows in samples:
        cs = by_tag.get(tag, [])
        if any(c[2] <= f < c[3] for c in cs):                 # a cast under way
            continue
        hit = next((c for c in cs if f < c[2] <= f + AHEAD), None)
        X.append(x)
        y.append(1 if hit else 0)
        if hit is not None and len(pts) and hit[2] - f <= SAMPLE_EVERY:      # (the sample just before it)
            d = [np.hypot(px - hit[4], py - hit[5]) for px, py in pts]
            j = int(np.argmin(d))
            if d[j] < SPOT_NEAR:
                S.append(rows)
                s_label.append(j)
    return {"error": None, "X": X, "y": y, "S": S, "s_label": s_label, "casts": len(casts),
            "cast_at": [c[2] for c in casts]}


def report(results: list[dict], name: str, n_games: int) -> None:
    """Casts a game (begun by 10, 15 and 20 minutes) and what their spots hit."""
    with_casts = [r for r in results if r["casts"]]
    line = f"{name}: {n_games} games, a cast in {len(with_casts) / max(1, n_games):.0%}"
    for m in (10, 15, 20):
        line += f"; by {m}:00 {sum(sum(1 for f in r['cast_at'] if f < m * 1440) for r in results) / max(1, n_games):.1f}"
    rows = [s[j] for r in results for s, j in zip(r["S"], r["s_label"])]
    if rows:
        mean = np.mean(rows, 0)
        line += " a game; at the spot " + ", ".join(f"{k} {mean[SPOT_NAMES.index(k)]:.2f}" for k in
                                                     ("dist", "enemies", "enemy_supply", "own", "recent_cast"))
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spell", default="storm", choices=sorted(SPELLS))
    ap.add_argument("--matchups", nargs="+", default=["PvZ", "PvT", "PvP"])
    ap.add_argument("--games", type=int, default=400, help="per matchup")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", help="default: data/interim/cast/<spell>_v1.npz")
    ap.add_argument("--report", nargs="*", help="league round folders: their games' casts against the pros' (--games of "
                                                "each matchup, no dataset written)")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from inventory import data_root
    from macro_dataset import select
    root = data_root()
    jobs = []
    for m in args.matchups:
        games = [g for g in select(m) if g["end_frame"] > 10 * 1440]
        games = games[::max(1, len(games) // args.games)][:args.games]   # spread over the sources
        for g in games:
            p = next((q for q in (root / "raw" / g["rel_path"], root / g["rel_path"]) if q.exists()), None)
            if p is not None:
                jobs.append((str(p), g["end_frame"], args.spell))
    with ProcessPoolExecutor(args.parallel) as pool:
        results = list(monitor.track(pool.map(_job, jobs, chunksize=2), len(jobs), f"{args.spell} games"))
        if args.report is not None:
            report(results, "pros " + " ".join(args.matchups), len(jobs))
            for folder in args.report:
                reps = sorted(str(q) for q in Path(folder).glob("*/*.rep"))
                report(list(pool.map(_job, [(q, UNTIL, args.spell) for q in reps])), f"Gary {Path(folder).name}",
                       len(reps))
            return
    X = np.array([x for r in results for x in r["X"]], np.float32).reshape(-1, len(CAST_NAMES))
    y = np.array([v for r in results for v in r["y"]], np.int8)
    game = np.array([i for i, r in enumerate(results) for _ in r["y"]], np.int32)
    rows = [s for r in results for s in r["S"]]
    s_start = np.cumsum([0] + [len(s) for s in rows]).astype(np.int64)
    S = np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, len(SPOT_NAMES)), np.float32)
    s_label = np.array([v for r in results for v in r["s_label"]], np.int32)
    s_game = np.array([i for i, r in enumerate(results) for _ in r["s_label"]], np.int32)
    out = Path(args.out) if args.out else root / "interim" / "cast" / f"{args.spell}_v1.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, game=game, S=S, s_start=s_start, s_label=s_label, s_game=s_game,
                        names=np.array(CAST_NAMES), spot_names=np.array(SPOT_NAMES), spell=args.spell)
    print(f"{len(jobs)} games ({sum(1 for r in results if r['error'])} errors, {sum(1 for r in results if r['casts'])} "
          f"with casts, {sum(r['casts'] for r in results)} casts), {len(y)} samples, cast {y.mean() if len(y) else 0:.1%}, "
          f"{len(s_label)} spots labelled -> {out}")


if __name__ == "__main__":
    main()
