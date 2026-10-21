"""Scenarios by state, from whole pro games (for eval/scenarios.py --suite kinds): the situations where the pros'
fight commands go, which the early home-defense suite doesn't sample (half to three quarters of their fight
commands re-issue moves across the map, in attacks, pokes and retreats: docs/PROGRESS.md 2026-10-21).

Each held-out (or training) modern TvZ game re-simulated with the fight output to 30 minutes; every Terran fight
snapshot from 4:00 is classified, the first kind that fits (the Terran's view: its army units and the zerg fighters
in the snapshot, the town halls standing then):
  mixed        a big fight: 20+ supply of Terran army, 2+ fighter types each side
  attack       contact nearer a Zerg town hall than a Terran one, 8+ supply of Terran army, 3+ zerg fighters
  outnumbered  the zerg fighters there outweigh the Terran's 1.6 to 1 or more (2+ supply of Terran army)
  harass       from 6:00, 4+ zerg fighters by the Terran's buildings, its army there weaker than theirs
Per game and kind at most PER_KIND, LEAD frames before the moment, MIN_GAP apart.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ingest"))
sys.path.insert(0, str(REPO / "resim"))

FROM = 4 * 60 * 24
HARASS_FROM = 6 * 60 * 24
LEAD = 24
MIN_GAP = 60 * 24
PER_KIND = 2
KINDS = ("mixed", "attack", "outnumbered", "harass")
T_HALLS, Z_HALLS = {106}, {131, 132, 133}


def _lines(path: Path) -> list[dict] | None:
    import scr_format
    from inventory import data_root
    from macro_dataset import resim_exe
    data = path.read_bytes()
    args = [str(resim_exe()), "--data", str(data_root() / "gamedata" / "scr"), "--replay", "-", "--every", "0",
            "--fights", "1800"]
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
            if '"fight"' in l or '"event"' in l]


def kind_of(snap: dict, halls: dict[str, list[tuple[int, int]]], frame: int) -> str | None:
    from gary.policy.army import is_building, is_worker, supply_x2
    own = [u for u in snap["u"] if u[2] == 1]
    foe = [u for u in snap["u"] if u[2] == 0]
    army = [u for u in own if supply_x2(u[1]) > 0 and not is_worker(u[1]) and not is_building(u[1])]
    zerg = [u for u in foe if supply_x2(u[1]) > 0 and not is_worker(u[1]) and not is_building(u[1])]
    ours, theirs = sum(supply_x2(u[1]) for u in army), sum(supply_x2(u[1]) for u in zerg)
    if not army or not zerg:
        return None
    if ours >= 40 and len({u[1] for u in army}) >= 2 and len({u[1] for u in zerg}) >= 2:
        return "mixed"
    c = (snap["cx"], snap["cy"])
    near = lambda hs: min((math.dist(c, h) for h in hs), default=1e9)
    if ours >= 16 and len(zerg) >= 3 and near(halls["Z"]) < near(halls["T"]):
        return "attack"
    if ours >= 4 and theirs >= 1.6 * ours:
        return "outnumbered"
    buildings = [u for u in own if is_building(u[1])]
    if frame >= HARASS_FROM and len(zerg) >= 4 and buildings and ours < theirs:
        return "harass"
    return None


def pick_game(job: tuple) -> list[dict]:
    """The scenarios of one game."""
    rec, root = job
    path = Path(root) / rec["rel_path"]
    lines = _lines(path)
    if not lines:
        return []
    race_of, halls_at = {}, []                        # (frame, slot-race, tag, x, y, alive?)
    for r in lines:
        if r["type"] == "event" and r["frame"] <= 1 and r.get("unit") in T_HALLS | Z_HALLS and r["slot"] >= 0:
            race_of[r["slot"]] = "T" if r["unit"] in T_HALLS else "Z"
    tslot = next((s for s, rc in race_of.items() if rc == "T"), None)
    if tslot is None:
        return []
    halls = {"T": {}, "Z": {}}
    out, last = [], {k: -10 ** 9 for k in KINDS}
    count = {k: 0 for k in KINDS}
    for r in lines:
        if r["type"] == "event" and r.get("unit") in T_HALLS | Z_HALLS and "x" in r:
            side = "T" if r["unit"] in T_HALLS else "Z"
            if r["ev"] == "done":
                halls[side][r["tag"]] = (r["x"], r["y"])
            elif r["ev"] == "gone":
                halls[side].pop(r["tag"], None)
        elif r["type"] == "fight" and r["slot"] == tslot and r["frame"] >= FROM:
            k = kind_of(r, {s: list(h.values()) for s, h in halls.items()}, r["frame"])
            if k is None or count[k] >= PER_KIND or r["frame"] - last[k] < MIN_GAP:
                continue
            last[k], count[k] = r["frame"], count[k] + 1
            out.append({"sha1": rec["sha1"], "rel_path": rec["rel_path"], "map": rec.get("map"), "terran_slot": tslot,
                        "terran": rec.get("player"), "frame": max(0, r["frame"] - LEAD), "kind": k,
                        "cx": r["cx"], "cy": r["cy"],
                        "zerg_fighters": sum(1 for u in r["u"] if u[2] == 0), "own_units": sum(1 for u in r["u"] if u[2] == 1)})
    return out


def pick(n: int, split: str, workers: int = 6) -> list[dict]:
    """n candidate scenarios at most, from modern TvZ games of the split (test: held out of training)."""
    from concurrent.futures import ProcessPoolExecutor
    from eval import monitor
    from eval.humanlike import clean_replays
    from inventory import data_root
    from train.macro import is_test
    clean = clean_replays()
    recs = [json.loads(l) for l in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r.get("matchup") == "TvZ" and r.get("race") == "T" and r.get("source") == "cwal"
            and r["rel_path"] in clean and r.get("duration_s", 0) >= 600 and is_test(r["sha1"]) == (split == "test")]
    recs.sort(key=lambda r: r["sha1"])
    out = []
    with ProcessPoolExecutor(workers) as pool:
        for scs in monitor.track(pool.map(pick_game, [(r, str(data_root() / "raw")) for r in recs]), len(recs), "games"):
            out += scs
            if len(out) >= n:
                break
    return out[:n]
