"""The pros' re-issues, added to a response dataset (ingest/response_dataset.py): for the role model's "again" decision
and its offset head (docs/PROGRESS.md, 2026-10-22: the fight roles' pipe is capped by what the decisions mean, since
keep gives no order, while half the pros' fight commands give a unit the same order again with its target moved).

Per labeled unit at each moment, from the re-simulator's unit commands (its --fights output; OpenBW's unit IDs, as the
dataset's tags, the same lining-up as its attended array):
  commanded  1 if a unit command of the player included the unit in the AGAIN_WINDOW frames from the moment, 0 if
             not, -1 if the replay's commands couldn't be read
  cmd_to     the first such command's target point (right-click or order) relative to the unit, in tiles, mirrored
             as the rows; NaN without a point or without the unit's position (from the re-simulator's fight
             snapshots, FIGHT_EVERY frames apart: the nearest within NEAR_FRAMES)
train/roles.py --again turns "keep" with commanded = 1 into "again" and trains the offset head on cmd_to.

    python -m ingest.reissue_labels --data data/interim/roles/response_T_att.npz
Output: the same name with _again (every array of the input, plus commanded and cmd_to).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))
sys.path.insert(0, str(REPO / "resim"))

from eval import monitor, priority  # noqa: E402

AGAIN_WINDOW = 12                     # frames: a command this soon after the moment is the moment's decision
NEAR_FRAMES = 12                      # a fight snapshot this near the moment gives the unit's position
MAX_MIN = 30
TERRAN = 1                            # (the header's race ids: 0 Zerg, 1 Terran, 2 Protoss)
RACE_ID = {"Z": 0, "T": 1, "P": 2}


def _resim(path: Path) -> list[dict] | None:
    import scr_format
    from inventory import data_root
    from macro_dataset import resim_exe
    data = path.read_bytes()
    args = [str(resim_exe()), "--data", str(data_root() / "gamedata" / "scr"), "--replay", "-", "--every", "0",
            "--fights", str(MAX_MIN * 60)]
    if scr_format.replay_format(data) != "legacy":
        args += ["--flat", "--unit-limit", str(scr_format.unit_limit(data))]
        data = scr_format.to_flat(data)
    try:
        proc = subprocess.run(args, input=data, capture_output=True, timeout=1800)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return [json.loads(l) for l in proc.stdout.decode("utf-8", "replace").splitlines()
            if '"ucmd"' in l or '"fight"' in l or '"header"' in l or ('"event"' in l and '"done"' in l)]


def game(job: tuple) -> tuple[int, dict | None]:
    """For one game: {(tag, frame): (commanded, dx, dy)} over the frames asked, or None if it couldn't be read."""
    gi, path, race, asks = job
    lines = _resim(Path(path))
    if lines is None:
        return gi, None
    head = next((r for r in lines if r["type"] == "header"), None)
    if head is None:
        return gi, None
    slots = [p["slot"] for p in head["players"] if p["race"] == RACE_ID[race]]
    if len(slots) != 1:
        return gi, None
    slot = slots[0]
    hall = next((r for r in lines if r["type"] == "event" and r["slot"] == slot and r["frame"] <= 1 and "x" in r), None)
    if hall is None:
        return gi, None
    fx = -1 if hall["x"] > head["map_w"] / 2 else 1
    fy = -1 if hall["y"] > head["map_h"] / 2 else 1
    import bisect
    cmds = sorted(((r["frame"], r) for r in lines if r["type"] == "ucmd" and r["slot"] == slot), key=lambda x: x[0])
    cf = [f for f, _ in cmds]
    snaps = sorted(((r["frame"], r) for r in lines if r["type"] == "fight" and r["slot"] == slot), key=lambda x: x[0])
    sf = [f for f, _ in snaps]
    pos_cache: dict[int, dict] = {}

    def position(tag: int, f: int):
        k = bisect.bisect_left(sf, f - NEAR_FRAMES)
        best = None
        while k < len(sf) and sf[k] <= f + NEAR_FRAMES:
            if k not in pos_cache:
                pos_cache[k] = {u[0]: (u[3], u[4]) for u in snaps[k][1]["u"] if u[2] == 1}
            p = pos_cache[k].get(tag)
            if p is not None and (best is None or abs(sf[k] - f) < best[0]):
                best = (abs(sf[k] - f), p)
            k += 1
        return best[1] if best else None
    out = {}
    for tag, f in asks:
        k = bisect.bisect_left(cf, f)
        hit = None
        while k < len(cf) and cf[k] < f + AGAIN_WINDOW:
            r = cmds[k][1]
            if tag in r["units"]:
                hit = r
                break
            k += 1
        if hit is None:
            out[(tag, f)] = (0, np.nan, np.nan)
            continue
        dx = dy = np.nan
        if hit["cmd"] in ("rclick", "order") and hit.get("x") is not None:
            p = position(tag, f)
            if p is not None:
                dx, dy = fx * (hit["x"] - p[0]) / 32, fy * (hit["y"] - p[1]) / 32
        out[(tag, f)] = (1, dx, dy)
    return gi, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/interim/roles/response_T_att.npz")
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
        jobs.append((gi, str(data_root() / "raw" / rec["rel_path"]), args.race, asks))
    commanded = np.full(d["labels"].shape, -1, np.int8)
    cmd_to = np.full(d["labels"].shape + (2,), np.nan, np.float16)
    read = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for gi, res in monitor.track(pool.map(game, jobs), len(jobs), "games"):
            if res is None:
                continue
            read += 1
            for i in np.nonzero(d["game"] == gi)[0]:
                for j in np.nonzero(labeled[i])[0]:
                    c, dx, dy = res.get((int(d["tags"][i, j]), int(frames[i])), (-1, np.nan, np.nan))
                    commanded[i, j] = c
                    cmd_to[i, j] = (dx, dy)
    keep = d["game"] < n_games
    d = {k: (v[keep] if k != "sha1" and len(v) == len(keep) else v) for k, v in d.items()}
    d["commanded"], d["cmd_to"] = commanded[keep], cmd_to[keep]
    out = Path(args.data).with_name(Path(args.data).stem + "_again.npz")
    np.savez_compressed(out, **d)
    known = d["commanded"][d["labels"] >= 0]
    pts = np.isfinite(d["cmd_to"][..., 0])[d["commanded"] == 1]
    print(f"{read} of {len(jobs)} games read; of the labeled units, {np.mean(known >= 0):.0%} known, "
          f"{np.mean(known[known >= 0] == 1):.1%} commanded in the next {AGAIN_WINDOW} frames "
          f"({np.mean(pts):.0%} of those with a point and a position) -> {out}")


if __name__ == "__main__":
    main()
