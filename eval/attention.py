"""Attention in fights, Gary's and the pros' the same way (the roles RL's gate 2, docs/SELF_PLAY.md and
docs/PROGRESS.md): each replay re-simulated with the fight output (resim --fights: a player's fight snapshots every
FIGHT_EVERY frames while its units have visible enemy fighters near, and every unit command with its units), per
player and race, within a window of the game:

  fight minutes    the snapshots' time
  into the fight   unit commands including one of the player's army units in a snapshot within NEAR frames, a
                   fight minute
  attended         the share of snapshots where such a command in the ATTEND frames before (to NEAR after) reached
                   one of its army units there
  reached          the share of its army units in snapshots that such a command reached
  the hands        during fight moments, the player's actions into the fight (as above), to its other units
                   (unit commands to none of the fight's army units) and macro (Train, Build, Unit and Building
                   Morph, Tech, Upgrade: the replay's own commands, ingest/cmdcache.py); and macro actions a minute
                   in fights and out of them

    python -m eval.attention --gary runs/league/R50a runs/league/D50a --pros 100
    python -m eval.attention --gary runs/league/R50a --until 600          # the first 10 minutes only
    python -m eval.attention --augment R50a D50a                          # into the rounds' feature rows
    python -m eval.attention --augment-pros                               # into the pros' cached rows

--augment writes, per side and game minute, fight_attended (the share of fight moments attended) and fight_reached
(the share of army units reached) into rows.json.gz (eval/league.py), the humanlike gap's features (NaN in a minute
without fight moments); --augment-pros the same into eval/humanlike.py's cache of the pros' rows.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))
sys.path.insert(0, str(REPO / "resim"))

from eval import monitor, priority  # noqa: E402

FIGHT_EVERY = 12                       # frames between a player's fight snapshots (resim/gary_resim.cpp)
NEAR = 12
ATTEND = 48
HALLS = {106: "T", 131: "Z", 154: "P"}
MACRO = {"Train", "Build", "Unit Morph", "Building Morph", "Tech", "Upgrade"}


def one(job: tuple) -> dict | None:
    """{race: Counter} for one replay, commands and snapshots up to `until` seconds."""
    path, until = job
    import bisect
    import scr_format
    from inventory import data_root
    from macro_dataset import resim_exe
    from gary.policy.army import is_building, is_worker
    data = Path(path).read_bytes()
    args = [str(resim_exe()), "--data", str(data_root() / "gamedata" / "scr"), "--replay", "-", "--every", "0",
            "--fights", str(until)]
    if scr_format.replay_format(data) != "legacy":
        args += ["--flat", "--unit-limit", str(scr_format.unit_limit(data))]
        data = scr_format.to_flat(data)
    try:
        proc = subprocess.run(args, input=data, capture_output=True, timeout=1800)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    lines = [json.loads(l) for l in proc.stdout.decode("utf-8", "replace").splitlines()
             if '"fight"' in l or '"ucmd"' in l or '"event"' in l]
    race_of = {}
    for r in lines:
        if r["type"] == "event" and r["frame"] <= 1 and r.get("unit") in HALLS and r["slot"] >= 0:
            race_of[r["slot"]] = HALLS[r["unit"]]
    cmds = collections.defaultdict(list)
    snaps = collections.defaultdict(list)
    for r in lines:
        if r["type"] == "ucmd" and r["frame"] <= until * 24:
            cmds[r["slot"]].append((r["frame"], set(r["units"])))
        elif r["type"] == "fight" and r["slot"] in race_of and r["frame"] <= until * 24:
            army = {u[0] for u in r["u"] if u[2] == 1 and not is_worker(u[1]) and not is_building(u[1])}
            if army:
                snaps[r["slot"]].append((r["frame"], army))
    macro = collections.defaultdict(list)          # race -> the frames of its player's macro commands
    try:
        import hashlib
        from cmdcache import F, P, T, load_game
        from inventory import find_screp
        raw = Path(path).read_bytes()
        cg = load_game(find_screp(None), Path(path), hashlib.sha1(raw).hexdigest(), data_root() / "interim" / "cmdcache" / "v1")
        race_of_pid = {pl["id"]: pl["race"] for pl in cg["players"] if not pl["observer"]}
        for cmd in cg["cmds"]:
            if cmd[T] in MACRO and cmd[F] <= until * 24 and cmd[P] in race_of_pid:
                macro[race_of_pid[cmd[P]]].append(cmd[F])
    except Exception:
        pass
    last = max((r["frame"] for r in lines if "frame" in r), default=0)
    out = {}
    for slot, race in race_of.items():
        sl = snaps[slot]
        if not sl:
            continue
        c = collections.Counter()
        c["snaps"] = len(sl)
        per_min = collections.defaultdict(lambda: [0, 0, 0, 0])   # game minute (1-based) -> snaps, attended, units, reached
        frames = [f for f, _ in sl]
        cf = [f for f, _ in cmds[slot]]
        in_fight = lambda f: (lambda k: k < len(sl) and abs(sl[k][0] - f) <= NEAR)(bisect.bisect_left(frames, f - NEAR))
        for f, us in cmds[slot]:                    # (commands into the fight: a snapshot within NEAR)
            k = bisect.bisect_left(frames, f - NEAR)
            if k < len(sl) and abs(sl[k][0] - f) <= NEAR:
                c["into" if us & sl[k][1] else "elsewhere"] += 1
        for f in macro.get(race, []):              # (macro actions, in fights or out of them)
            c["macro in fights" if in_fight(f) else "macro out of fights"] += 1
        c["game minutes"] = min(until * 24, last) / 1440
        for f, army in sl:
            reached = set()
            k = bisect.bisect_left(cf, f - ATTEND)
            while k < len(cf) and cf[k] <= f + NEAR:
                reached |= cmds[slot][k][1] & army
                k += 1
            c["attended"] += bool(reached)
            c["units"] += len(army)
            c["reached"] += len(reached)
            pm = per_min[f // 1440 + 1]
            pm[0] += 1; pm[1] += bool(reached); pm[2] += len(army); pm[3] += len(reached)
        c["_per_min"] = dict(per_min)
        out[race] = c
    return out


def show(name: str, paths: list[str], until: int, workers: int) -> None:
    tot = collections.defaultdict(collections.Counter)
    games = collections.Counter()
    with ProcessPoolExecutor(workers) as pool:
        for res in monitor.track(pool.map(one, [(p, until) for p in paths]), len(paths), name):
            for race, c in (res or {}).items():
                c.pop("_per_min", None)
                tot[race] += c
                games[race] += 1
    for race, c in sorted(tot.items()):
        minutes = c["snaps"] * FIGHT_EVERY / 24 / 60
        print(f"{name} {race} ({games[race]} games, to {until // 60} min): {minutes / max(1, games[race]):.1f} fight "
              f"minutes a game; commands into the fight {c['into'] / max(1e-9, minutes):.0f} a fight minute; attended "
              f"{c['attended'] / max(1, c['snaps']):.0%} of fight moments; army units reached {c['reached'] / max(1, c['units']):.0%}",
              flush=True)
        hands = c["into"] + c["elsewhere"] + c["macro in fights"]
        out_min = max(1e-9, c["game minutes"] - minutes)
        print(f"  the hands in fights: into the fight {c['into'] / max(1, hands):.0%}, other units "
              f"{c['elsewhere'] / max(1, hands):.0%}, macro {c['macro in fights'] / max(1, hands):.0%} "
              f"({hands / max(1e-9, minutes):.0f} actions a fight minute); macro actions a minute in fights "
              f"{c['macro in fights'] / max(1e-9, minutes):.1f}, out of them {c['macro out of fights'] / out_min:.1f}",
              flush=True)


def put(rows: list[dict], per_min: dict) -> None:
    """A side's per-minute rows given fight_attended and fight_reached (NaN in a minute without fight moments)."""
    nan = float("nan")
    for r in rows:
        snaps, att, units, reached = per_min.get(int(r["minute"]), (0, 0, 0, 0))
        r["fight_attended"] = att / snaps if snaps else nan
        r["fight_reached"] = reached / units if units else nan


def augment_round(folder: Path, workers: int) -> None:
    """A league round's rows.json.gz given the attention features, both sides."""
    import gzip
    path = folder / "rows.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        saved = json.load(f)
    rels = sorted(set().union(*(set(v) for v in saved.values())))      # (the sides by race: T, Z, P)
    with ProcessPoolExecutor(workers) as pool:
        for rel, res in zip(rels, monitor.track(pool.map(one, [(str(folder / rel), 1800) for rel in rels]), len(rels),
                                                folder.name)):
            for side in saved:
                if rel in saved.get(side, {}):
                    put(saved[side][rel], (res or {}).get(side, {}).get("_per_min", {}))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(saved, f)
    print(f"{folder.name}: attention features in {len(rels)} games' rows", flush=True)


def augment_pros(workers: int) -> None:
    """eval/humanlike.py's cached modern pro rows (both races; what eval/gap_windows.py compares with) given the attention features."""
    from inventory import data_root
    for cache in sorted((data_root() / "interim" / "humanlike").glob("pro_*_modern_*min.jsonl")):
        race = cache.name.split("_")[1]
        lines = [json.loads(l) for l in open(cache, encoding="utf-8")]
        todo = [r for r in lines if not any("fight_attended" in row for row in r["rows"])]
        jobs = [(str(data_root() / "raw" / r["rel_path"]), 1800) for r in todo]
        with ProcessPoolExecutor(workers) as pool:
            for r, res in zip(todo, monitor.track(pool.map(one, jobs), len(jobs), cache.stem)):
                put(r["rows"], (res or {}).get(race, {}).get("_per_min", {}))
        with open(cache, "w", encoding="utf-8") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        print(f"{cache.name}: attention features in {len(todo)} games' rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gary", nargs="*", default=[], help="league round folders")
    ap.add_argument("--pros", type=int, default=0, help="this many modern pro TvZ games (clean, 10+ minutes)")
    ap.add_argument("--until", type=int, default=1800, help="seconds of each game")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--augment", nargs="*", default=[], help="league rounds (names): their rows given the features")
    ap.add_argument("--augment-pros", action="store_true", help="the pros' cached rows given the features")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    for name in args.augment:
        augment_round(REPO / "runs" / "league" / name, args.workers)
    if args.augment_pros:
        augment_pros(args.workers)
    for folder in args.gary:
        show(Path(folder).name, sorted(glob.glob(f"{folder}/*/*.rep")), args.until, args.workers)
    if args.pros:
        from inventory import data_root
        from eval.humanlike import clean_replays
        clean = clean_replays()
        recs = [json.loads(l) for l in open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8")]
        rels = sorted({r["rel_path"] for r in recs if r.get("matchup") == "TvZ" and r.get("source") == "cwal"
                       and r["rel_path"] in clean and r.get("duration_s", 0) >= 600})
        random.Random(0).shuffle(rels)
        show("pros", [str(data_root() / "raw" / p) for p in rels[:args.pros]], args.until, args.workers)


if __name__ == "__main__":
    main()
