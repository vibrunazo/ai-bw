"""The response dataset: what the pros' Terran units did, every half second, whenever the Terran saw
enemy fighters near them, before contact as well as in it (for the role model, gary/policy/roles.py;
docs/SELF_PLAY.md: anticipation).

The role dataset (ingest/role_dataset.py) labels units in contact, in home-defense scenarios. What made
Gary feel AFK to the user came earlier and with no damage: marines moved into the mineral line as a
dozen zerglings came (they backed off, no blow traded), marines stimmed and gathered at the sight of
mutalisks strafing at a distance, a lone marine running to the buildings from 4 zerglings, where Gary
attacked and died. So here: whole pro games (clean re-simulations, the Terran's side), the game as
the Terran sees it (enemy units it doesn't see are left out of everything, the labels' distances as
well), and every Terran fighter with a seen enemy fighter within SIGHT labeled with its role over the
next half second (fire, close in, pull back, retreat, reposition, stand, other: role_dataset's
labels), with that distance kept (near: pre-contact when it's NEAR or more).

Between moments the game steps a second at a time; a moment (a seen enemy fighter within SIGHT plus
a margin of its fighters or buildings) steps every 2 frames, as the role probe needs.

    python -m ingest.response_dataset --games 600
Output: data/interim/roles/response_T.npz (the role dataset's arrays, plus near and move: each labeled
unit's move over the half second, tiles, mirrored as the rows; attended: 1 if a unit command of the player
included the unit in the ATTEND_BEFORE frames before the moment or the ATTEND_AFTER after it, 0 if not, -1 if the
replay's commands couldn't be read (the re-simulator's unit commands): a third of the pros' fighting is
unattended, units on auto-acquire or an old order, which teaches no decision; budget: the player's unit commands
per minute over the BUDGET_S before the moment, the attention it was spending)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority
from eval.micro_roles import NEAR
from gary.policy.roles import EXTRA, MAX_UNITS, ROLES
from ingest.role_dataset import HALLS, Probe

SIGHT = 12 * 32                 # a seen enemy fighter this near: the unit is labeled
ACTIVE = SIGHT + 4 * 32         # ...this near its fighters or buildings: the game steps finely
QUIET = 48                      # frames with no moment before stepping coarsely again
ATTEND_BEFORE, ATTEND_AFTER = 48, 12   # frames: a command this near the moment reached the unit
BUDGET_S = 30
MAX_MIN = 30                    # (20 until 2026-10-20: the role models had never seen a late-game fight)
MAPS = ("python", "fighting spirit", "heartbreak")


class Seen:
    """The game as one player sees it: enemy units it doesn't see are left out of observe()."""

    def __init__(self, game, slot: int):
        self.game, self.slot = game, slot

    def observe(self) -> dict:
        o = self.game.observe()
        return {**o, "units": [u for u in o["units"] if u["owner"] in (self.slot, 11)
                               or (u.get("visible_to", 0) >> self.slot) & 1]}

    def __getattr__(self, name):
        return getattr(self.game, name)


def _moment(o: dict, slot: int) -> bool:
    from gary.policy.army import is_building, is_worker, supply_x2
    mine = [(u["x"], u["y"]) for u in o["units"] if u["owner"] == slot and (is_building(u["type"]) or supply_x2(u["type"]) > 0)]
    foes = [(u["x"], u["y"]) for u in o["units"] if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
            and not is_worker(u["type"]) and not is_building(u["type"])]
    if not mine or not foes:
        return False
    a, b = np.array(mine, np.float32), np.array(foes, np.float32)
    return bool((np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)) < ACTIVE).any())


def _commands(path: str) -> dict[int, list[tuple[int, list[int]]]] | None:
    """{slot: [(frame, unit tags)]}: every unit command of each player, from the re-simulator (its --fights output;
    tags are OpenBW's unit IDs, as the game's), or None if it couldn't run."""
    import subprocess
    import sys
    from ingest.inventory import data_root
    from ingest.macro_dataset import resim_exe
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "resim"))
    import scr_format
    data = Path(path).read_bytes()
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
    out: dict[int, list] = {}
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        if '"ucmd"' in line:
            r = json.loads(line)
            out.setdefault(r["slot"], []).append((r["frame"], r["units"]))
    return out


def _attention(cmds: list[tuple[int, list[int]]]):
    """(attended(tag, frame) -> 1/0, budget(frame) -> commands a minute over BUDGET_S) for one player's commands."""
    import bisect
    by_tag: dict[int, list[int]] = {}
    frames = [f for f, _ in cmds]
    for f, tags in cmds:
        for t in tags:
            by_tag.setdefault(t, []).append(f)

    def attended(tag: int, f: int) -> int:
        fs = by_tag.get(tag)
        if not fs:
            return 0
        k = bisect.bisect_left(fs, f - ATTEND_BEFORE)
        return int(k < len(fs) and fs[k] <= f + ATTEND_AFTER)

    def budget(f: int) -> float:
        return (bisect.bisect_right(frames, f) - bisect.bisect_left(frames, f - BUDGET_S * 24)) * 60.0 / BUDGET_S
    return attended, budget


def _job(job: tuple) -> dict:
    path, end, sha1, race, own = job
    out = {"sha1": sha1, "error": None, "rows": [], "extra": [], "labels": [], "near": [], "move": [], "time": [],
           "tags": [], "attended": [], "budget": []}
    try:
        from gary.env import Game
        g = Game.scenario(path)
        try:
            last = min(end or MAX_MIN * 1440, MAX_MIN * 1440)
            g.step(24)
            slot = next(u["owner"] for u in g.observe()["units"] if u["type"] in HALLS[race])
            seen = Seen(g, slot)
            probe = Probe(near=SIGHT, race=race, own=own)
            quiet = QUIET
            while g.frame + 24 <= last:
                o = seen.observe()
                if _moment(o, slot):
                    quiet = 0
                if quiet < QUIET:
                    probe(seen, slot)
                    g.step(2)
                    quiet += 2
                else:
                    g.step(24 - g.frame % 24 if g.frame % 24 else 24)
        finally:
            g.close()
        labels = probe.labels()
        cmds = _commands(path)
        attended, budget = _attention(cmds.get(slot, [])) if cmds is not None else (None, None)
        near_of = {}
        for tag, s in probe.samples.items():
            for f, *_rest in s:
                near_of[(tag, f)] = _rest[3]
        for f, r, e, tags in probe.snaps:
            y = np.array([ROLES.index(labels[(t, f)]) if (t, f) in labels and r[i][1] == 1 else -1
                          for i, t in enumerate(tags)], np.int8)
            if (y >= 0).any():
                out["rows"].append(r)
                out["extra"].append(e)
                out["labels"].append(y)
                out["near"].append(np.array([near_of.get((t, f), 1e4) for t in tags], np.float32))
                out["move"].append(np.array([probe.moves.get((t, f), (np.nan, np.nan)) for t in tags], np.float32))
                out["time"].append(f * 42 / 1000)
                out["tags"].append(tags)
                out["attended"].append(np.array([attended(t, f) if attended and y[i] >= 0 else -1
                                                 for i, t in enumerate(tags)], np.int8))
                out["budget"].append(budget(f) if budget else np.nan)
    except Exception as e:                         # (a replay that won't run: counted, skipped)
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def select(games: int, race: str = "T", era: str = "both") -> list[tuple]:
    """games of the race's side of TvZ, clean, 6+ minutes: era "old", the 1.16 games on MAPS; "modern", the
    ladder games (data/raw/cwal) on any map; "both", half each."""
    from eval.humanlike import clean_replays
    from ingest.inventory import data_root
    clean = clean_replays()
    sha = {}
    for line in open(data_root() / "interim" / "resim_index.jsonl", encoding="utf-8"):
        r = json.loads(line)
        sha[r["rel_path"]] = r["sha1"]
    recs = [r for r in map(json.loads, open(data_root() / "interim" / "build_orders.jsonl", encoding="utf-8"))
            if r["matchup"] == "TvZ" and r["race"] == race and r["rel_path"] in clean and r.get("duration_s", 0) >= 360]
    old = [r for r in recs if any(m in re.sub(r"[^a-z ]", "", r.get("map", "").lower()) for m in MAPS)]
    modern = [r for r in recs if r["rel_path"].startswith("cwal/")]
    random.Random(0).shuffle(old)
    random.Random(0).shuffle(modern)
    chosen = old[:games] if era == "old" else modern[:games] if era == "modern" else         old[:games // 2] + modern[:games - games // 2]
    return [(str(data_root() / "raw" / r["rel_path"]), clean[r["rel_path"]] or None, sha[r["rel_path"]], race)
            for r in chosen]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", choices=("T", "Z"), default="T", help="whose side (TvZ games)")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--era", choices=("old", "modern", "both"), default="both",
                    help="old: the 1.16 games on the three fixture maps (all until 2026-10-20); modern: ladder games")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", help="default: data/interim/roles/response_<race>.npz")
    ap.add_argument("--own-extras", action="store_true", help="fill extras 3-5 (a unit's age, its bunch: as play does)")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    from ingest.inventory import data_root
    jobs = [(*j, args.own_extras) for j in select(args.games, args.race, args.era)]
    with ProcessPoolExecutor(args.parallel) as pool:
        results = list(monitor.track(pool.map(_job, jobs, chunksize=1), len(jobs), "games"))
    n = sum(len(r["rows"]) for r in results)
    rows = np.zeros((n, MAX_UNITS, 11), np.int16)
    extra = np.zeros((n, MAX_UNITS, EXTRA), np.float32)
    labels = np.full((n, MAX_UNITS), -1, np.int8)
    near = np.full((n, MAX_UNITS), 1e4, np.float16)
    move = np.full((n, MAX_UNITS, 2), np.nan, np.float16)
    unit_tags = np.zeros((n, MAX_UNITS), np.int32)
    att = np.full((n, MAX_UNITS), -1, np.int8)
    bud = np.full(n, np.nan, np.float32)
    count = np.zeros(n, np.int16)
    time_s = np.zeros(n, np.float32)
    game = np.zeros(n, np.int32)
    i = 0
    for gi, r in enumerate(results):
        for rr, e, y, nr, mv, t, tg, a, b in zip(r["rows"], r["extra"], r["labels"], r["near"], r["move"], r["time"],
                                                 r["tags"], r["attended"], r["budget"]):
            k = min(len(rr), MAX_UNITS)
            rows[i, :k], extra[i, :k], labels[i, :k], near[i, :k], move[i, :k] = rr[:k], e[:k], y[:k], nr[:k], mv[:k]
            count[i], time_s[i], game[i] = k, t, gi
            unit_tags[i, :k] = tg[:k]
            att[i, :k], bud[i] = a[:k], b
            i += 1
    out = Path(args.out) if args.out else data_root() / "interim" / "roles" / f"response_{args.race}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=rows, extra=extra, labels=labels, near=near, move=move, count=count, time=time_s, game=game,
                        tags=unit_tags, attended=att, budget=bud, sha1=np.array([r["sha1"] for r in results]))
    y = labels[labels >= 0]
    pre = labels[(labels >= 0) & (near >= NEAR)]
    print(f"{len(results)} games ({sum(1 for r in results if r['error'])} errors), {n} snapshots, {len(y)} labeled units "
          f"({len(pre)} before contact) -> {out}")
    print("  all:            " + ", ".join(f"{ROLES[k]} {np.mean(y == k):.1%}" for k in range(len(ROLES))))
    known = att[labels >= 0]
    if (known >= 0).any():
        ya = labels[(labels >= 0) & (att == 1)]
        print(f"  attended: {np.mean(known[known >= 0] == 1):.0%} of the labeled units whose commands were read "
              f"({np.mean(known >= 0):.0%} read); budget median {np.nanmedian(bud):.0f} commands a minute")
        print("  attended only:  " + ", ".join(f"{ROLES[k]} {np.mean(ya == k):.1%}" for k in range(len(ROLES))))
    print("  before contact: " + ", ".join(f"{ROLES[k]} {np.mean(pre == k):.1%}" for k in range(len(ROLES))))


if __name__ == "__main__":
    main()
