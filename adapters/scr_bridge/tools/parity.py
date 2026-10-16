#!/usr/bin/env python3
"""Is the bridge 1:1 with OpenBW? A live game's replay re-simulated in OpenBW, and what the bridge showed
Gary each second (gary.play --live --snapshots) compared with the re-simulated game at the same frames.

If OpenBW plays the replay's commands as the real game did, both are the same game, frame for frame: a
unit or field that differs is the bridge reading the game wrong (or OpenBW disagreeing with it), and a
building the re-simulation has but the live game hasn't is a command the real game refused. Needs a
replay without the computer AI (OpenBW can't re-simulate that): a LAN game against a second client.

    python adapters/scr_bridge/tools/parity.py runs/live/v025_lan1.snap.jsonl runs/live/v025_lan1.rep
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from gary.env import Game  # noqa: E402
from gary.terran import NAMES  # noqa: E402

SKIP = {"tag", "order_target"}       # the key, and tags (compared as units, not numbers)
NEAR = 2                             # px: positions this close count as equal (sub-pixel rounding)
PLAYER_FIELDS = ("minerals", "gas", "supply_used", "supply_max")


def name(t: int) -> str:
    return NAMES.get(t, str(t))


def live_key(tag: int, unit_limit: int) -> tuple[int, int]:
    """A live tag as (OpenBW's unit index + 1, generation mod 8). In a game with the extended unit table
    (3400) SC:R's ids carry a 13-bit index and units made in play sit 1700 slots above OpenBW's (the
    map's own units, at the start of the table, don't): resim/scr_replay.h translate_unit_id."""
    if unit_limit <= 1700:
        return tag & 0x7ff, (tag >> 11) % 8
    index1 = tag & 0x1fff
    return (index1 - (unit_limit - 1700) if index1 > unit_limit - 1700 else index1), tag >> 13


def sim_key(tag: int) -> tuple[int, int]:
    return tag & 0x7ff, (tag >> 11) % 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots")
    ap.add_argument("replay")
    ap.add_argument("--examples", type=int, default=4, help="examples shown per mismatching field")
    ap.add_argument("--shift", type=int, default=0, help="compare live frame F with re-simulated F+shift")
    args = ap.parse_args()

    snaps = [json.loads(l) for l in Path(args.snapshots).read_text(encoding="utf-8").splitlines() if l.strip()]
    # the replay ends when the game does for its writer (a player defeated: both LAN games ended at
    # ~9:30, when Gary killed the idle Zerg, while Gary played on): later snapshots have nothing to match
    head = json.loads(subprocess.run([str(REPO / "tools" / "bin" / "screp.exe"), args.replay], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace").stdout)["Header"]
    end = head["Frames"]
    dropped = sum(1 for s in snaps if s["frame"] + args.shift >= end)
    snaps = [s for s in snaps if s["frame"] + args.shift < end]
    game = Game.scenario(args.replay)
    sys.path.insert(0, str(REPO / "resim"))
    import scr_format
    limit = scr_format.unit_limit(Path(args.replay).read_bytes()) if scr_format.replay_format(Path(args.replay).read_bytes()) != "legacy" else 1700
    compared = Counter()                  # field -> comparisons
    differ = Counter()                    # field -> mismatches
    first_differ: dict[str, int] = {}     # field -> first frame it differed
    examples = defaultdict(list)
    only_live, only_sim = Counter(), Counter()
    first_only: list[str] = []
    keys_live, keys_sim = set(), set()
    players_differ = Counter()
    player_examples = []
    try:
        for s in snaps:
            target = s["frame"] + args.shift
            if target < game.frame:
                continue
            game.step(target - game.frame)
            sim = game.observe()
            live_u = {live_key(u["tag"], limit): u for u in s["units"]}
            sim_u = {sim_key(u["tag"]): u for u in sim["units"]}
            if live_u:
                keys_live |= set(next(iter(live_u.values())))
            if sim_u:
                keys_sim |= set(next(iter(sim_u.values())))
            for tag in live_u.keys() - sim_u.keys():
                u = live_u[tag]
                only_live[(u["owner"], name(u["type"]))] += 1
                if len(first_only) < 8:
                    first_only.append(f"  {s['frame']}: live only: {name(u['type'])} of player {u['owner']} at ({u['x']}, {u['y']})")
            for tag in sim_u.keys() - live_u.keys():
                u = sim_u[tag]
                only_sim[(u["owner"], name(u["type"]))] += 1
                if len(first_only) < 8:
                    first_only.append(f"  {s['frame']}: OpenBW only: {name(u['type'])} of player {u['owner']} at ({u['x']}, {u['y']})")
            for tag in live_u.keys() & sim_u.keys():
                a, b = live_u[tag], sim_u[tag]
                for k in (a.keys() & b.keys()) - SKIP:
                    compared[k] += 1
                    same = abs(a[k] - b[k]) <= NEAR if k in ("x", "y") and isinstance(a[k], int) else a[k] == b[k]
                    if not same:
                        differ[k] += 1
                        first_differ.setdefault(k, s["frame"])
                        if len(examples[k]) < args.examples:
                            examples[k].append(f"{s['frame']}: {name(a['type'])} {tag}: live {a[k]!r}, OpenBW {b[k]!r}")
            sim_p = {p["slot"]: p for p in sim["players"]}
            for p in s["players"]:
                q = sim_p.get(p["slot"])
                for k in PLAYER_FIELDS:
                    if q is not None and k in p and k in q and p[k] != q[k]:
                        players_differ[k] += 1
                        if len(player_examples) < 6:
                            player_examples.append(f"{s['frame']}: player {p['slot']} {k}: live {p[k]}, OpenBW {q[k]}")
    finally:
        game.close()

    print(f"{len(snaps)} snapshots, frames {snaps[0]['frame']}-{snaps[-1]['frame']} (shift {args.shift}); the replay "
          f"ends at frame {end}: {dropped} later snapshots not compared")
    print("\nunits in one game only (owner, type: snapshots):")
    for (owner, t), n in (only_live + only_sim).most_common(12):
        print(f"  player {owner} {t}: live only {only_live[(owner, t)]}, OpenBW only {only_sim[(owner, t)]}")
    print("first ones:")
    print("\n".join(first_only) or "  none")
    print("\nfields (mismatches / comparisons, first frame):")
    for k in sorted(compared, key=lambda k: -differ[k] / compared[k]):
        line = f"  {k:14s} {differ[k]:7d} / {compared[k]:7d}"
        if differ[k]:
            line += f"  first {first_differ[k]}"
        print(line)
        for e in examples[k] if differ[k] else []:
            print(f"      {e}")
    print(f"\nfields only the bridge has: {sorted(keys_live - keys_sim)}")
    print(f"fields only OpenBW has:     {sorted(keys_sim - keys_live)}")
    print("\nplayers (mismatching snapshots):", dict(players_differ) or "none")
    for e in player_examples:
        print(f"  {e}")


if __name__ == "__main__":
    main()
