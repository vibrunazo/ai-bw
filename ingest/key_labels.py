"""The pros' control groups, added to a response dataset (ingest/response_dataset.py): for each labeled unit at each
moment, the key its player had it on (for the role model's key input: units on one key act as one group, as the
pros' do; docs/PROGRESS.md 2026-10-22, recalls need decisions by key).

From the replay's own commands (ingest/cmdcache.py, ingest/hotkey_dataset.py key_events: selections and hotkeys),
the Terran player's keys over time; a unit's key at a moment is the key holding it that was assigned or recalled
last (a unit on two keys: the one in use), -1 if no key holds it, -2 if the replay's commands couldn't be read.
Replay tags and the dataset's (OpenBW's unit IDs) meet on OpenBW's table index (hotkey_dataset.index_mapper; the
ID's low 11 bits).

    python -m ingest.key_labels --data data/interim/roles/response_T_att_again.npz
Output: the same name with _keys (every array of the input, plus key).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))
sys.path.insert(0, str(REPO / "resim"))

from eval import monitor, priority  # noqa: E402

RACE = {"T": "T", "Z": "Z", "P": "P"}


def game(job: tuple) -> tuple[int, dict | None]:
    """For one game: {(tag, frame): key} over the asked units and frames, or None if its commands couldn't be read."""
    gi, path, race, asks = job
    from cmdcache import load_game
    from hotkey_dataset import index_mapper, key_events
    from inventory import data_root, find_screp
    raw = Path(path).read_bytes()
    try:
        cg = load_game(find_screp(None), Path(path), hashlib.sha1(raw).hexdigest(), data_root() / "interim" / "cmdcache" / "v1")
    except Exception:
        return gi, None
    players = [p for p in cg["players"] if not p["observer"] and p["race"] == race]
    if len(players) != 1:
        return gi, None
    index_of = index_mapper(raw)
    # per key, its contents over time: [(frame, set of indexes)], and when it was last used (assigned or recalled)
    contents: dict[int, list] = {}
    used: dict[int, list] = {}
    for f, kind, key, units in key_events(cg, players[0]["id"]):
        if kind is None:
            continue
        if kind in ("assign", "add"):
            contents.setdefault(key, []).append((f, {index_of(u) for u in units}))
        used.setdefault(key, []).append(f)
    frames = {k: [f for f, _ in v] for k, v in contents.items()}
    out = {}
    for tag, f in asks:
        idx = tag & 0x7FF
        best = (-1, -1)                              # (last use, key)
        for k, v in contents.items():
            i = bisect.bisect_right(frames[k], f) - 1
            if i < 0 or idx not in v[i][1]:
                continue
            u = used[k][bisect.bisect_right(used[k], f) - 1]
            if u > best[0]:
                best = (u, k)
        out[(tag, f)] = best[1]
    return gi, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/interim/roles/response_T_att_again.npz")
    ap.add_argument("--race", default="T")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", type=int, help="only the first N games (a check)")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from inventory import data_root
    d = dict(np.load(args.data))
    recs = {}
    for line in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8"):
        r = json.loads(line)
        recs.setdefault(r["sha1"], r)
    n_games = len(d["sha1"]) if args.games is None else args.games
    frames = np.rint(d["time"] * 1000 / 42).astype(int)
    labeled = d["labels"] >= 0
    jobs = []
    for gi in range(n_games):
        rec = recs.get(str(d["sha1"][gi]))
        if rec is None:
            continue
        idx = np.nonzero(d["game"] == gi)[0]
        asks = [(int(d["tags"][i, j]), int(frames[i])) for i in idx for j in np.nonzero(labeled[i])[0]]
        jobs.append((gi, str(data_root() / "raw" / rec["rel_path"]), RACE[args.race], asks))
    key = np.full(d["labels"].shape, -2, np.int8)
    read = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for gi, res in monitor.track(pool.map(game, jobs), len(jobs), "games"):
            if res is None:
                continue
            read += 1
            for i in np.nonzero(d["game"] == gi)[0]:
                for j in np.nonzero(labeled[i])[0]:
                    key[i, j] = res.get((int(d["tags"][i, j]), int(frames[i])), -2)
    keep = d["game"] < n_games
    d = {k: (v[keep] if k != "sha1" and len(v) == len(keep) else v) for k, v in d.items()}
    d["key"] = key[keep]
    out = Path(args.data).with_name(Path(args.data).stem + "_keys.npz")
    np.savez_compressed(out, **d)
    k = d["key"][d["labels"] >= 0]
    print(f"{read} of {len(jobs)} games read; of the labeled units {np.mean(k >= -1):.0%} known, "
          f"{np.mean(k[k >= -1] >= 0):.0%} of those on a key -> {out}")


if __name__ == "__main__":
    main()
