"""A stop-gap's decisions as labels on Gary's own states, for a commit model to imitate the rule where the pros'
replays don't show the behaviour (AGENTS.md, stop-gaps; the hunt: pros leave at GG, so their replays hardly show a
game finished). The rule runs in advice mode in Gary's games (option huntadvice: what it would do, in the records,
play unchanged); its labels are joined to the game's commitment ticks, built from the replay as the pros' are
(ingest/army_dataset.py, the same channels and mirroring).

Each "hunt advice" record (a squad, the place it stands, its target) labels the tick cluster nearest its squad, at the
latest tick at most TICK seconds before it, as engage to the target's cell. The numbers' "seen lately" memory comes
from the ticks themselves (the most seen at once over MEMORY_S, at 5-s ticks rather than the pros' 1-s snapshots).

    from train.rule_labels import rule_samples
    train, test = rule_samples(["HL90a", "HL90b"], spec, workers=6)
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ingest"))
sys.path.insert(0, str(REPO))

from inventory import data_root  # noqa: E402
from gary.policy.army import ARMY_SUPPLY, MEMORY_S, ArmySpec, commit_inputs, encode_global  # noqa: E402

TICK = 5                        # seconds between ticks (ingest/army_dataset.py TICK)
NEAR_CLUSTER = 2                # an advised squad labels a cluster at most this many cells from where it stands
ENGAGE = 1


def _out(round_name: str) -> Path:
    import army_dataset as ad
    return data_root() / "interim" / "army_gary" / ad.VERSION / round_name


def _build(job: tuple) -> dict | None:
    """One of Gary's replays built into commitment ticks (cached), with its records' path."""
    import army_dataset as ad
    round_name, rep = job
    out = _out(round_name)
    ad.out_dir = lambda: out                                  # (Gary's games kept out of the pros' dataset)
    sha = hashlib.sha1(rep.encode()).hexdigest()
    path = out / sha[:2] / f"{sha}.npz"
    meta = path.with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    row = ad.build_game({"sha1": sha, "rel_path": rep, "matchup": "TvZ"})   # (absolute: data/raw / it is it)
    if not row.get("ok"):
        return None
    i = next((k for k, p in enumerate(row["players"]) if p["race"] == "T"), None)
    if i is None:
        return None
    info = {"npz": str(path), "i": i, "records": rep.replace(".rep", ".records.jsonl"), "held": int(sha[:8], 16) % 10 == 0}
    meta.write_text(json.dumps(info), encoding="utf-8")
    return info


def _cell(tx: int, ty: int, flip: np.ndarray) -> int:
    from army_dataset import cell_of
    fx, fy, mw, mh = (int(v) for v in flip)
    return cell_of(tx * 32 + 16, ty * 32 + 16, mw, mh, bool(fx), bool(fy))


def _samples(info: dict, spec: ArmySpec) -> tuple | None:
    d = np.load(info["npz"])
    i = info["i"]
    frames, grid9 = d[f"p{i}_tick_frame"], d[f"p{i}_tick_grid"]
    if not len(frames) or not Path(info["records"]).exists():
        return None
    advice = []
    with open(info["records"], encoding="utf-8") as f:
        for line in f:
            if '"hunt advice"' in line:
                r = json.loads(line)
                if r.get("side") == "T" and r["in"].get("kind") != "SCAN":
                    advice.append(r)
    picked, targets = [], []
    for r in advice:
        k = int(np.searchsorted(frames, r["f"], side="right")) - 1
        if k < 0 or r["f"] - frames[k] > TICK * 24:
            continue
        same = np.nonzero(frames == frames[k])[0]
        at = _cell(*r["in"]["at"], d[f"p{i}_flip"])
        dist = [max(abs(int(grid9[j, 8].argmax()) // 16 - at // 16), abs(int(grid9[j, 8].argmax()) % 16 - at % 16))
                for j in same]
        j = int(same[int(np.argmin(dist))])
        if min(dist) <= NEAR_CLUSTER and j not in picked:
            picked.append(j)
            targets.append(_cell(*r["in"]["to"], d[f"p{i}_flip"]))
    if not picked:
        return None
    idx = np.array(picked, np.int64)
    grid = grid9[idx]
    if spec.losses:
        grid = np.concatenate([grid, d[f"p{i}_tick_loss"][idx][:, None]], 1)
    for name in ("static", "known", "unseen"):
        if getattr(spec, name):
            grid = np.concatenate([grid, d[f"p{i}_tick_{name}"][idx][:, None]], 1)
    seen = d[f"p{i}_tick_seen"].astype(np.float32)
    seen_mem = np.stack([seen[(frames <= frames[j]) & (frames >= frames[j] - MEMORY_S * 24)].max(0) for j in idx])
    units = d[f"p{i}_tick_units"][idx].astype(np.float32)
    commit = d[f"p{i}_tick_commit"][idx]
    glob = encode_global(spec, frames[idx].astype(np.float32), d[f"p{i}_tick_eco"][idx].astype(np.float32),
                         grid[:, 8].sum(1).astype(np.float32), units, seen[idx], seen_mem, units @ ARMY_SUPPLY,
                         seen_mem @ ARMY_SUPPLY, commit_inputs(commit, d[f"p{i}_tick_age"][idx]) if spec.commit else None)
    n = len(idx)
    return (grid, glob.astype(np.float16), np.full(n, ENGAGE, np.int64), np.array(targets, np.int64),
            commit.astype(np.int64), frames[idx])


def rule_samples(rounds: list[str], spec: ArmySpec, workers: int = 6) -> dict:
    """{"train": arrays, "test": arrays} as train/commit.py's load(): the advice-labelled ticks of these league
    rounds' games, split by game (a tenth held out)."""
    league = REPO / "runs" / "league"
    jobs = [(r, str(p.resolve())) for r in rounds for p in sorted((league / r).glob("*/*.rep"))]
    with ProcessPoolExecutor(workers) as pool:
        infos = [x for x in pool.map(_build, jobs) if x]
    parts = {False: [], True: []}
    for info in infos:
        s = _samples(info, spec)
        if s is not None:
            parts[info["held"]].append(s)
    empty = lambda: tuple(np.zeros((0,) + shape, dt) for shape, dt in
                          (((spec.channels, 256), np.uint8), ((spec.n_global,), np.float16), ((), np.int64),
                           ((), np.int64), ((), np.int64), ((), np.int32)))
    return {("test" if held else "train"): tuple(np.concatenate([p[k] for p in ps]) for k in range(6)) if ps else empty()
            for held, ps in parts.items()}
