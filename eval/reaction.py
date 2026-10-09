"""How fast pros react, from replays: the delay between something happening and the player's first
command about it (docs/ARCHITECTURE.md §7.2 step 8; the interface's reaction_ms, gary/interface.py).

A replay records commands, not why they were given, so only events with an unambiguous answer
count (gary_resim --fights prints them, with every unit command and its selection):

  sight  an enemy unit comes into the player's sight (after 2 s out of it) -> the first command
         targeting that very unit (attacking it, or right-clicking it)
  hit    one of the player's units loses hit points (after 2 s without), and the player hasn't
         commanded it in the 2 s before -> the first command that includes that unit (move it
         away, stim, attack back...)

Only events within a screen of the player's last click (within 2 s) count: replays don't record
the camera, but clicks land on screen, so that's where the player was looking. Players also give
those commands anyway, unprompted; a control measures that: the same search started 3 s after each
event. The reaction is what the events add over the control, and its fast end is the perception
delay (the slow end is attention and priorities, which Gary's arbiter models apart).

    python -m eval.reaction --matchup TvZ --games 200
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))
sys.path.insert(0, str(REPO_ROOT))

FRAME_MS = 42
WINDOW = 3 * 24                 # a response this soon after the event (frames)
CONTROL_AFTER = 3 * 24          # the control: the same search, this long after the event
LOOKING = 2 * 24                # the player's last click this recent...
SCREEN = (320 + 64, 200 + 64)   # ...and within half a screen (and a bit) of the event: looking there


def events_of(lines: list[dict]):
    """Per slot: sorted commands, sights, hits."""
    cmds, sights, hits = {}, {}, {}
    for e in lines:
        t = e.get("type")
        if t == "ucmd":
            cmds.setdefault(e["slot"], []).append(e)
        elif t == "sight":
            sights.setdefault(e["slot"], []).append(e)
        elif t == "hit":
            hits.setdefault(e["slot"], []).append(e)
    return cmds, sights, hits


def looking(cmds: list[dict], frame: int, x: int, y: int) -> bool:
    """The player's last click before frame, if recent, was within a screen of (x, y)."""
    i = bisect.bisect_left(cmds, frame, key=lambda c: c["frame"]) - 1
    while i >= 0 and cmds[i]["x"] < 0:
        i -= 1
    last = cmds[i] if i >= 0 else None
    return last is not None and frame - last["frame"] <= LOOKING and \
        abs(last["x"] - x) <= SCREEN[0] and abs(last["y"] - y) <= SCREEN[1]


def commanded(cmds: list[dict], frame: int, tag: int) -> bool:
    """The player gave this unit a command in the LOOKING frames before frame (it's under control
    already: a command right after a hit is then no answer to it)."""
    i = bisect.bisect_left(cmds, frame, key=lambda c: c["frame"]) - 1
    while i >= 0 and frame - cmds[i]["frame"] <= LOOKING:
        if tag in cmds[i]["units"]:
            return True
        i -= 1
    return False


def first_response(cmds: list[dict], start: int, match) -> int | None:
    """Frames from start to the first command within WINDOW that matches, or None."""
    for c in cmds[bisect.bisect_left(cmds, start, key=lambda c: c["frame"]):]:
        if c["frame"] - start > WINDOW:
            return None
        if match(c):
            return c["frame"] - start
    return None


def measure(lines: list[dict], race: int | None = None, name: str | None = None) -> dict:
    """The delays of the players of this race (0 Zerg, 1 Terran, 2 Protoss) or whose name starts
    with name; everyone's if neither."""
    cmds, sights, hits = events_of(lines)
    players = next((e["players"] for e in lines if e.get("type") == "header"), [])
    keep = {p["slot"] for p in players if (race is None or p["race"] == race) and (name is None or p["name"].startswith(name))}
    out = {"sight": Counter(), "sight_ctl": Counter(), "hit": Counter(), "hit_ctl": Counter(),
           "n_sight": 0, "n_hit": 0}
    for slot, cs in cmds.items():
        if slot not in keep:
            continue
        for e in sights.get(slot, []):
            if not looking(cs, e["frame"], e["x"], e["y"]):
                continue
            out["n_sight"] += 1
            on_it = lambda c, tag=e["tag"]: c["target"] == tag
            d = first_response(cs, e["frame"], on_it)
            if d is not None:
                out["sight"][d] += 1
            d = first_response(cs, e["frame"] + CONTROL_AFTER, on_it)
            if d is not None:
                out["sight_ctl"][d] += 1
        for e in hits.get(slot, []):
            if not looking(cs, e["frame"], e["x"], e["y"]) or commanded(cs, e["frame"], e["tag"]):
                continue
            out["n_hit"] += 1
            with_it = lambda c, tag=e["tag"]: tag in c["units"]
            d = first_response(cs, e["frame"], with_it)
            if d is not None:
                out["hit"][d] += 1
            d = first_response(cs, e["frame"] + CONTROL_AFTER, with_it)
            if d is not None:
                out["hit_ctl"][d] += 1
    return out


def one(rec: dict, race: int | None, name: str | None) -> dict | None:
    """rec: a replay in the data (rel_path) or a file anywhere (path)."""
    from fight_dataset import run_resim
    try:
        # (run_resim reads data/raw / rel_path: an absolute path stands for itself)
        return measure(run_resim(str(Path(rec["path"]).resolve()) if "path" in rec else rec["rel_path"]), race, name)
    except Exception:
        return None


def summarize(name: str, ev: Counter, ctl: Counter, n: int) -> None:
    """The events' delays over the control's, in 2-frame bins, and the percentiles of the excess."""
    if not n:
        print(f"{name}: no events")
        return
    excess = {}
    for b in range(0, WINDOW + 1, 2):
        e = ev.get(b, 0) + ev.get(b + 1, 0)
        c = ctl.get(b, 0) + ctl.get(b + 1, 0)
        excess[b] = max(0, e - c)
    total = sum(excess.values()) or 1
    acc, pct = 0, {}
    for b in sorted(excess):
        acc += excess[b]
        for q in (0.1, 0.25, 0.5, 0.75):
            if q not in pct and acc >= q * total:
                pct[q] = b
    answered, control = sum(ev.values()), sum(ctl.values())
    print(f"{name}: {n} events, answered within {WINDOW * FRAME_MS / 1000:.0f} s: {answered} "
          f"(control {control}); reaction over the control, percentiles: " +
          ", ".join(f"{int(q * 100)}% {pct[q] * FRAME_MS} ms" for q in sorted(pct)))
    peak = max(excess, key=excess.get)
    print(f"   most common delay {peak * FRAME_MS}-{(peak + 2) * FRAME_MS} ms; by 2 frames (events - control): " +
          " ".join(f"{excess[b]}" for b in sorted(excess)[:24]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--race", choices=["Z", "T", "P"], help="only players of this race")
    ap.add_argument("--replays", nargs="+", help="these replay files instead (e.g. Gary's games from eval/vs_rush.py --out)")
    ap.add_argument("--name", help="only players whose name starts with this (e.g. Gary)")
    from eval import priority
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    if args.replays:
        games = [{"path": p} for p in args.replays]
    else:
        from macro_dataset import select
        games = select(args.matchup)
        random.Random(args.seed).shuffle(games)
        games = games[:args.games]
    race = {"Z": 0, "T": 1, "P": 2}.get(args.race)
    tot = {"sight": Counter(), "sight_ctl": Counter(), "hit": Counter(), "hit_ctl": Counter(), "n_sight": 0, "n_hit": 0}
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(one, g, race, args.name) for g in games]):
            r = f.result()
            if r is None:
                continue
            done += 1
            for k, v in r.items():
                tot[k] = tot[k] + v
    print(f"{done} of {len(games)} games" + (f", {args.race} players" if args.race else "") +
          (f", players named {args.name}*" if args.name else ""))
    summarize("enemy unit comes into sight -> command on it", tot["sight"], tot["sight_ctl"], tot["n_sight"])
    summarize("own unit hit -> command including it", tot["hit"], tot["hit_ctl"], tot["n_hit"])


if __name__ == "__main__":
    main()
