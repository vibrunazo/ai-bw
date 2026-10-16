"""The stim dataset: when the pros stim (for gary/policy/stim.py, trained by train/stim.py).

Re-simulates the pros' TvZ games in OpenBW and, every half second from 4:00, takes each of the
Terran's marines and firebats in a fight (a zerg fighter within 10 tiles, as the Terran sees them):
its view of the fight (gary/policy/stim.py features) and whether the Terran stimmed it in the next
half second (OpenBW's stim timer went up). Only from the Terran's first stim on (before it, stim
isn't researched: no decision to learn).

    python -m ingest.stim_dataset --games 1500 --parallel 3
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority
from gary.policy.stim import BIO, NAMES, SAMPLE_EVERY, enemy_fighters, features, in_fight, recall

AHEAD = 2                        # the label: stimmed within this many samples (1 s): the moment is noisy

START = 4 * 1440                 # nobody stims before 4:00
UNTIL = 20 * 1440                # enough fights by then


def _job(job: tuple[str, int]) -> dict:
    path, end = job
    from gary.env import Game
    X, y, frames = [], [], []
    first_stim = None
    try:
        g = Game.scenario(path)
    except Exception as e:
        return {"error": str(e), "X": X, "y": y}
    try:
        obs = g.observe()
        slot = next((u["owner"] for u in obs["units"] if u["type"] == 106), None)
        if slot is None:
            return {"error": "no Terran", "X": X, "y": y}
        g.step(START - g.frame)
        waiting: list[list] = []                 # [tag, features, stim then, frame, steps left, stimmed]
        last: dict[int, tuple] = {}              # tag -> recall() half a second ago
        stop = min(end - (AHEAD + 1) * SAMPLE_EVERY, UNTIL)
        while g.frame < stop:
            g.step(SAMPLE_EVERY)
            obs = g.observe()
            stim = {u["tag"]: u.get("stim", 0) for u in obs["units"] if u["owner"] == slot}
            for w in waiting:
                tag = w[0]
                if tag in stim and stim[tag] > w[2]:
                    w[5] = 1
                w[2] = stim.get(tag, w[2])
                w[4] -= 1
            for w in [w for w in waiting if w[4] <= 0 and w[0] in stim]:
                X.append(w[1])
                y.append(w[5])
                frames.append(w[3])
                if w[5] and first_stim is None:
                    first_stim = w[3]
            waiting = [w for w in waiting if w[4] > 0 and w[0] in stim]
            own = [u for u in obs["units"] if u["owner"] == slot and u["completed"]]
            enemies = enemy_fighters(obs["units"], slot)
            bio = [u for u in own if u["type"] in BIO and in_fight(u, enemies)]
            for u in bio:
                f = features(u, own, enemies, u.get("stim", 0), obs["frame"], last.get(u["tag"]), lambda t: stim.get(t, 0))
                waiting.append([u["tag"], f, u.get("stim", 0), obs["frame"], AHEAD, 0])
            last = {u["tag"]: recall(u, enemies) for u in own if u["type"] in BIO}
    except Exception as e:
        return {"error": str(e), "X": X, "y": y}
    finally:
        g.close()
    if first_stim is None:
        return {"error": None, "X": [], "y": []}
    keep = [i for i, fr in enumerate(frames) if fr >= first_stim]
    return {"error": None, "X": [X[i] for i in keep], "y": [y[i] for i in keep]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--out", help="default: data/interim/stim/stim_v1.npz")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))      # (macro_dataset imports its siblings by name)
    from inventory import data_root
    from macro_dataset import select
    root = data_root()
    games = [g for g in select("TvZ") if g["end_frame"] > 9 * 1440]
    games = games[::max(1, len(games) // args.games)][:args.games]   # spread over the sources
    jobs = []
    for g in games:
        p = next((q for q in (root / "raw" / g["rel_path"], root / g["rel_path"]) if q.exists()), None)
        if p is not None:
            jobs.append((str(p), g["end_frame"]))
    with ProcessPoolExecutor(args.parallel) as pool:
        results = list(monitor.track(pool.map(_job, jobs, chunksize=4), len(jobs), "stim games"))
    X = np.array([x for r in results for x in r["X"]], np.float32)
    y = np.array([v for r in results for v in r["y"]], np.int8)
    game = np.array([i for i, r in enumerate(results) for _ in r["y"]], np.int32)
    out = Path(args.out) if args.out else root / "interim" / "stim" / "stim_v2.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, game=game, names=np.array(NAMES))
    used = sum(1 for r in results if r["y"])
    print(f"{len(jobs)} games ({sum(1 for r in results if r['error'])} errors, {used} with stims), "
          f"{len(y)} samples, stimmed {y.mean():.1%} -> {out}")


if __name__ == "__main__":
    main()
