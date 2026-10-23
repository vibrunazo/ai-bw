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

    python -m eval.attention --gary runs/league/R50a runs/league/D50a --pros 100
    python -m eval.attention --gary runs/league/R50a --until 600          # the first 10 minutes only
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
    out = {}
    for slot, race in race_of.items():
        sl = snaps[slot]
        if not sl:
            continue
        c = collections.Counter()
        c["snaps"] = len(sl)
        frames = [f for f, _ in sl]
        cf = [f for f, _ in cmds[slot]]
        for f, us in cmds[slot]:                    # (commands into the fight: a snapshot within NEAR)
            k = bisect.bisect_left(frames, f - NEAR)
            if k < len(sl) and abs(sl[k][0] - f) <= NEAR and us & sl[k][1]:
                c["into"] += 1
        for f, army in sl:
            reached = set()
            k = bisect.bisect_left(cf, f - ATTEND)
            while k < len(cf) and cf[k] <= f + NEAR:
                reached |= cmds[slot][k][1] & army
                k += 1
            c["attended"] += bool(reached)
            c["units"] += len(army)
            c["reached"] += len(reached)
        out[race] = c
    return out


def show(name: str, paths: list[str], until: int, workers: int) -> None:
    tot = collections.defaultdict(collections.Counter)
    games = collections.Counter()
    with ProcessPoolExecutor(workers) as pool:
        for res in monitor.track(pool.map(one, [(p, until) for p in paths]), len(paths), name):
            for race, c in (res or {}).items():
                tot[race] += c
                games[race] += 1
    for race, c in sorted(tot.items()):
        minutes = c["snaps"] * FIGHT_EVERY / 24 / 60
        print(f"{name} {race} ({games[race]} games, to {until // 60} min): {minutes / max(1, games[race]):.1f} fight "
              f"minutes a game; commands into the fight {c['into'] / max(1e-9, minutes):.0f} a fight minute; attended "
              f"{c['attended'] / max(1, c['snaps']):.0%} of fight moments; army units reached {c['reached'] / max(1, c['units']):.0%}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gary", nargs="*", default=[], help="league round folders")
    ap.add_argument("--pros", type=int, default=0, help="this many modern pro TvZ games (clean, 10+ minutes)")
    ap.add_argument("--until", type=int, default=1800, help="seconds of each game")
    ap.add_argument("--workers", type=int, default=6)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
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
