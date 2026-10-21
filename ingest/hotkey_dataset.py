"""The pros' control groups, as data (for the grouping slot's learned provider, gary/agent/grouping.py, and as
humanlike features): every hotkey command of a player, with the units on the key and their types at that moment.

Per replay (clean, modern by default), per player: its selections tracked through Select / Select Add / Select
Remove / Hotkey (as ingest/build_orders.py does; a key holds at most 12 units), each Hotkey command an event:
  {"f": frame, "kind": "assign" | "add" | "recall", "key": n, "units": [tags], "types": [unit type ids]}
with the units' types from the re-simulator's events (a unit's type: its last "done" before the moment), plus the
player's selection and unit-order counts (orders: the re-simulator's unit commands, right-clicks included). A group
is army when most of its units are fighters (not workers, not buildings).

    python -m ingest.hotkey_dataset --matchup TvZ --games 200
    python -m ingest.hotkey_dataset --report                         # the pros' statistics
    python -m ingest.hotkey_dataset --report --gary runs/league/L40a  # and Gary's, the same way

Output: data/interim/hotkeys/<matchup>.jsonl (a line per player-game).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import random
import statistics as st
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))
sys.path.insert(0, str(REPO / "resim"))

from eval import monitor, priority  # noqa: E402
from gary.policy.army import is_building, is_worker, supply_x2  # noqa: E402

LATE = 300 * 24                       # statistics from 5:00 (openings hotkey buildings, not armies)
SELECTS = {"Select", "Select Add", "Select Remove"}
RACE_HALL = {106: "T", 131: "Z", 154: "P"}


def _resim(path: Path) -> list[dict] | None:
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
            if '"event"' in l or '"ucmd"' in l or '"header"' in l]


def game(job: tuple) -> list[dict]:
    """The player-game lines for one replay."""
    path, matchup = job
    from cmdcache import F, G, H, P, T, TAGS, load_game
    from inventory import data_root, find_screp
    path = Path(path)
    lines = _resim(path)
    if lines is None:
        return []
    raw = path.read_bytes()
    try:
        cg = load_game(find_screp(None), path, hashlib.sha1(raw).hexdigest(), data_root() / "interim" / "cmdcache" / "v1")
    except Exception:
        return []
    import scr_format
    limit = scr_format.unit_limit(raw) if scr_format.replay_format(raw) != "legacy" else 1700
    shift = limit - 1700

    def index_of(tag: int) -> int:
        """A replay command's unit tag -> OpenBW's table index (resim/scr_replay.h translate_unit_id): a 3400-slot
        table has a 13-bit index, its top half shifted by the extra slots; the generation bits are left out."""
        if limit <= 1700:
            return tag & 0x7FF
        i = tag & 0x1FFF
        return i if i <= shift else i - shift
    types: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)    # OpenBW index -> [(frame, type)]
    race_of_slot = {}
    for r in lines:
        if r.get("type") == "event" and r.get("ev") in ("done", "gone"):     # (gone: None from then on)
            types[r["tag"] & 0x7FF].append((r["frame"], r["unit"] if r["ev"] == "done" else None))
            if r["frame"] <= 1 and r["unit"] in RACE_HALL and r["slot"] >= 0:
                race_of_slot[r["slot"]] = RACE_HALL[r["unit"]]
    orders = collections.Counter()
    for r in lines:
        if r.get("type") == "ucmd" and r["frame"] >= LATE:
            orders[race_of_slot.get(r["slot"])] += 1
    end = max((r["frame"] for r in lines if "frame" in r), default=0)

    def type_at(tag: int, f: int) -> int | None:
        last = None
        for fr, t in types.get(index_of(tag), []):
            if fr <= f:
                last = t
        return last
    out = []
    for p in cg["players"]:
        if p["observer"] or p["race"] not in ("T", "Z", "P"):
            continue
        sel: list[int] = []
        keys: dict[int, list[int]] = {}
        events, selections = [], 0
        for c in cg["cmds"]:
            if c[P] != p["id"]:
                continue
            t = c[T]
            if t == "Select":
                sel = list(dict.fromkeys(c[TAGS] or []))[:12]
            elif t == "Select Add":
                sel = list(dict.fromkeys(sel + list(c[TAGS] or [])))[:12]
            elif t == "Select Remove":
                gone = set(c[TAGS] or [])
                sel = [u for u in sel if u not in gone]
            elif t == "Hotkey" and c[G] is not None:
                kind = {"Assign": "assign", "Add": "add", "Select": "recall"}.get(c[H])
                if kind == "assign":
                    keys[c[G]] = list(sel)
                elif kind == "add":
                    keys[c[G]] = list(dict.fromkeys(keys.get(c[G], []) + sel))[:12]
                elif kind == "recall":
                    sel = list(keys.get(c[G], []))
                if kind:
                    units = keys.get(c[G], [])
                    events.append({"f": c[F], "kind": kind, "key": c[G], "units": units,
                                   "types": [type_at(u, c[F]) for u in units]})
            else:
                continue
            if c[F] >= LATE and (t in SELECTS or (t == "Hotkey" and c[H] == "Select")):
                selections += 1
        out.append({"rel": str(path), "matchup": matchup, "race": p["race"], "minutes": end / 1440,
                    "selections": selections, "orders": orders.get(p["race"], 0), "events": events})
    return out


def is_army(types: list) -> bool:
    known = [t for t in types if t is not None]
    fighters = [t for t in known if supply_x2(t) > 0 and not is_worker(t) and not is_building(t)]
    return bool(known) and len(fighters) > len(known) / 2


def report(rows: list[dict], name: str) -> None:
    from gary.interface import unit_name
    by_race = collections.defaultdict(list)
    for r in rows:
        by_race[r["race"]].append(r)
    for race, rs in sorted(by_race.items()):
        stat = collections.defaultdict(list)
        mixes = collections.Counter()
        sizes = []
        known = total = 0
        for r in rs:
            mins = max(1.0, r["minutes"] - LATE / 1440)
            late = [e for e in r["events"] if e["f"] >= LATE]
            army = [e for e in late if is_army(e["types"])]
            stat["army keys in use"].append(len({e["key"] for e in army if e["kind"] == "recall"}))
            for kind in ("assign", "add", "recall"):
                stat[f"army {kind}s a minute"].append(sum(1 for e in army if e["kind"] == kind) / mins)
            stat["selections per order"].append(r["selections"] / max(1, r["orders"]))
            for e in army:
                known += sum(t is not None for t in e["types"])
                total += len(e["types"])
                if e["kind"] == "assign":
                    sizes.append(len(e["units"]))
                    mixes[" + ".join(sorted({unit_name(t) for t in e["types"] if t is not None}))] += 1
        print(f"{name} {race}: {len(rs)} player-games (unit types known for {known / max(1, total):.0%} of grouped units)")
        for k, v in stat.items():
            print(f"  {k:28s} median {st.median(v):6.2f}  quartiles {[round(x, 2) for x in st.quantiles(v, n=4)] if len(v) > 1 else v}")
        if sizes:
            print(f"  army group size at assign: median {st.median(sizes)}")
        n = sum(mixes.values()) or 1
        print("  army groups' unit types (assigns): " + "; ".join(f"{m} {c / n:.0%}" for m, c in mixes.most_common(6)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--gary", nargs="*", default=[], help="league round folders: Gary's games the same way")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from inventory import data_root
    out = data_root() / "interim" / "hotkeys" / f"{args.matchup}.jsonl"
    if not args.report:
        from eval.humanlike import clean_replays
        clean = clean_replays()
        recs = [json.loads(l) for l in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
        rels = sorted({r["rel_path"] for r in recs if r.get("matchup") == args.matchup and r.get("source") == "cwal"
                       and r["rel_path"] in clean and r.get("duration_s", 0) >= 600})
        random.Random(0).shuffle(rels)
        jobs = [(str(data_root() / "raw" / p), args.matchup) for p in rels[:args.games]]
        out.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.workers) as pool, open(out, "w", encoding="utf-8") as f:
            for rows in monitor.track(pool.map(game, jobs), len(jobs), "replays"):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    report(rows, "pros")
    for folder in args.gary:
        jobs = [(p, args.matchup) for p in sorted(glob.glob(f"{folder}/*/*.rep"))]
        with ProcessPoolExecutor(args.workers) as pool:
            gary_rows = [r for rows in pool.map(game, jobs) for r in rows]
        report(gary_rows, f"Gary {Path(folder).name}")


if __name__ == "__main__":
    main()
