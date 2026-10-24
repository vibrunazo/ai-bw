"""Training data for Gary's fight model: what each unit was told to do in a skirmish.

From every clean, re-simulated game, the first 10 minutes (early harass and fights): every half
second in which a player's units have visible enemy fighters within 8 tiles, a snapshot of the
units around the fight (the player's own within 24 tiles; the enemy units they can see and
mineral fields and geysers within 12), and, for each of the player's units, what the player
ordered it to do in the next half second. Also every unit command each player gave (the command
model's memory of its own recent commands: gary/policy/fight_memory.py).

Per game (one .npz), snapshots are stored flat, with offsets:
  snap_frame (S,), snap_player (S,) 0/1, snap_start (S+1,) into the unit rows
  units (U, 11) int16: type, side (1 own, 0 enemy, 2 neutral), x, y relative to the fight's
         center (pixels, mirrored so the player's main is top-left), hp, shields, weapon
         cooldown, order, order target (row within the snapshot, -1 none), carrying, completed
  act (U,) int8: -1 not the player's unit, else one of ACTIONS
  act_target (U,) int16: the row within the snapshot the action targets (-1 none)
  act_dx, act_dy (U,) int16: for moves, where to, relative to the unit (mirrored like x, y)
  unit_tag (U,) uint32: the unit's ID (as in replay commands)
  snap_cx, snap_cy (S,) int32: the fight's center (map pixels); flip (2, 2) bool: per player,
         whether x and y are mirrored
  cmd_frame, cmd_player, cmd_kind (C,), cmd_x, cmd_y, cmd_target (C,), cmd_start (C+1,) into
         cmd_units: every unit command of both players (kind: fight_memory.HIST_KINDS)

Output: data/interim/fight/<VERSION>/<sha1[:2]>/<sha1>.npz and .../index.jsonl.

Usage:
  python ingest/fight_dataset.py --matchup TvZ --limit 100
  python ingest/fight_dataset.py --matchup TvZ
  python ingest/fight_dataset.py --report
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from inventory import data_root
from macro_dataset import RACES, TOWN_HALLS, resim_exe, select

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "resim"))
import scr_format  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from gary.policy.fight_memory import command_kind  # noqa: E402

VERSION = "v3"                  # v3: + unit tags, centers, commands; v2: own units within 768 px of the
                                # fight (v1: 384, like everyone)
FIGHTS_UNTIL_S = 600            # the first 10 minutes: early harass and skirmishes
LABEL_FRAMES = 12               # an order counts for a snapshot if it comes within half a second
ACTIONS = ["none", "move", "attack_move", "attack_unit", "gather", "own_unit", "stop", "hold",
           "stim", "return_cargo", "other"]
A = {name: i for i, name in enumerate(ACTIONS)}
REFINERIES = {110, 149, 157}


def out_dir() -> Path:
    return data_root() / "interim" / "fight" / VERSION


def run_resim(rel_path: str) -> list[dict]:
    data = (data_root() / "raw" / rel_path).read_bytes()
    args = [str(resim_exe()), "--data", str(data_root() / "gamedata" / "scr"), "--replay", "-",
            "--every", "0", "--fights", str(FIGHTS_UNTIL_S)]
    if scr_format.replay_format(data) != "legacy":
        args += ["--flat", "--unit-limit", str(scr_format.unit_limit(data))]
        data = scr_format.to_flat(data)
    proc = subprocess.run(args, input=data, capture_output=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()[:300])
    return [json.loads(line) for line in proc.stdout.decode("utf-8", "replace").splitlines()]


def label(cmd: dict, unit: list, rows: dict[int, int], snap_units: list) -> tuple[int, int, int, int]:
    """(action, target row, dx, dy) for one unit given the first command that included it."""
    c, target = cmd["cmd"], cmd["target"]
    row = rows.get(target, -1) if target else -1
    dx, dy = cmd["x"] - unit[3], cmd["y"] - unit[4]

    def on_unit(r: int) -> int:
        side, kind = snap_units[r][2], snap_units[r][1]
        if side == 0:
            return A["attack_unit"]
        if side == 2 or kind in REFINERIES:
            return A["gather"]
        return A["own_unit"]

    if c == "rclick":
        return (on_unit(row), row, 0, 0) if row >= 0 else (A["move"], -1, dx, dy)
    if c == "order":
        if cmd["order"] == 14:                       # attack-move
            return (A["attack_unit"], row, 0, 0) if row >= 0 and snap_units[row][2] == 0 else (A["attack_move"], -1, dx, dy)
        if cmd["order"] in (10, 11):                 # attack a unit
            return (A["attack_unit"], row, 0, 0) if row >= 0 else (A["attack_move"], -1, dx, dy)
        if cmd["order"] == 6:                        # move
            return (A["move"], -1, dx, dy)
        return (A["other"], row, 0, 0)
    return ({"stop": A["stop"], "hold": A["hold"], "stim": A["stim"], "return": A["return_cargo"]}.get(c, A["other"]),
            -1, 0, 0)


def build_game(rec: dict) -> dict:
    row = {"sha1": rec["sha1"], "rel_path": rec["rel_path"], "matchup": rec["matchup"]}
    t0 = time.time()
    try:
        lines = run_resim(rec["rel_path"])
    except Exception as e:  # noqa: BLE001 - one bad replay must not stop the batch
        row.update(ok=False, error=str(e)[:300], interrupted=type(e).__name__ == "ResimInterrupted")
        return row
    header = next(r for r in lines if r["type"] == "header")
    mw, mh = header.get("map_w", 4096), header.get("map_h", 4096)
    starts = {}
    for r in lines:
        if r["type"] == "event" and r["frame"] <= 1 and r.get("unit") in TOWN_HALLS and r["slot"] >= 0:
            starts.setdefault(r["slot"], (r["x"], r["y"]))
    slots = sorted(starts)
    if len(slots) != 2:
        row.update(ok=False, error="not two players with a start town hall")
        return row
    pidx = {s: i for i, s in enumerate(slots)}
    flip = {s: (starts[s][0] > mw / 2, starts[s][1] > mh / 2) for s in slots}
    cmds: dict[int, list[dict]] = {s: [] for s in slots}
    for r in lines:
        if r["type"] == "ucmd" and r["slot"] in cmds:
            cmds[r["slot"]].append(r)

    snap_frame, snap_player, snap_start = [], [], [0]
    snap_cx, snap_cy, unit_tag = [], [], []
    units, act, act_target, act_dx, act_dy = [], [], [], [], []
    counts = [0, 0]
    for r in lines:
        if r["type"] != "fight" or r["slot"] not in pidx:
            continue
        s, f = r["slot"], r["frame"]
        fx, fy = flip[s]
        snap = r["u"]
        rows = {u[0]: k for k, u in enumerate(snap)}
        upcoming = [c for c in cmds[s] if f < c["frame"] <= f + LABEL_FRAMES]
        snap_cx.append(r["cx"])
        snap_cy.append(r["cy"])
        for u in snap:
            tag, kind, side, x, y, hp, sh, cd, order, otarget, carry, done = u
            unit_tag.append(tag)
            rx, ry = x - r["cx"], y - r["cy"]
            units.append((kind, side, -rx if fx else rx, -ry if fy else ry, hp, sh, cd, order,
                          rows.get(otarget, -1) if otarget else -1, carry, done))
            if side != 1:
                act.append(-1), act_target.append(-1), act_dx.append(0), act_dy.append(0)
                continue
            c = next((c for c in upcoming if tag in c["units"]), None)
            a, t, dx, dy = label(c, u, rows, snap) if c else (A["none"], -1, 0, 0)
            act.append(a)
            act_target.append(t)
            act_dx.append(int(np.clip(-dx if fx else dx, -2048, 2047)))
            act_dy.append(int(np.clip(-dy if fy else dy, -2048, 2047)))
        snap_frame.append(f)
        snap_player.append(pidx[s])
        snap_start.append(len(units))
        counts[pidx[s]] += 1

    cmd_frame, cmd_player, cmd_kind, cmd_x, cmd_y, cmd_target, cmd_start, cmd_units = [], [], [], [], [], [], [0], []
    for r in lines:
        if r["type"] == "ucmd" and r["slot"] in pidx and r["frame"] <= FIGHTS_UNTIL_S * 1000 / 42 + 24:
            cmd_frame.append(r["frame"]); cmd_player.append(pidx[r["slot"]])
            cmd_kind.append(command_kind(r["cmd"], r.get("order", -1), r.get("target", 0)))
            cmd_x.append(max(0, r["x"])); cmd_y.append(max(0, r["y"])); cmd_target.append(r.get("target", 0))
            cmd_units.extend(r["units"]); cmd_start.append(len(cmd_units))
    path = out_dir() / rec["sha1"][:2] / f"{rec['sha1']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, snap_frame=np.array(snap_frame, np.int32), snap_player=np.array(snap_player, np.int8),
                        snap_start=np.array(snap_start, np.int32), units=np.clip(np.array(units, np.int32).reshape(-1, 11), -32768, 32767).astype(np.int16),
                        act=np.array(act, np.int8), act_target=np.array(act_target, np.int16),
                        act_dx=np.array(act_dx, np.int16), act_dy=np.array(act_dy, np.int16),
                        unit_tag=np.array(unit_tag, np.uint32), snap_cx=np.array(snap_cx, np.int32),
                        snap_cy=np.array(snap_cy, np.int32), flip=np.array([flip[s] for s in slots], bool),
                        cmd_frame=np.array(cmd_frame, np.int32), cmd_player=np.array(cmd_player, np.int8),
                        cmd_kind=np.array(cmd_kind, np.int8), cmd_x=np.array(cmd_x, np.int32),
                        cmd_y=np.array(cmd_y, np.int32), cmd_target=np.array(cmd_target, np.uint32),
                        cmd_start=np.array(cmd_start, np.int32), cmd_units=np.array(cmd_units, np.uint32))
    path.write_bytes(buf.getvalue())
    names = {p["slot"]: p for p in header["players"]}
    labeled = np.array(act, np.int8)
    row.update(ok=True, map=header.get("map"),
               players=[{"slot": s, "name": names.get(s, {}).get("name", ""),
                         "race": RACES.get(names.get(s, {}).get("race"), "?"), "snapshots": counts[pidx[s]]}
                        for s in slots],
               units=len(units), commanded=int((labeled > 0).sum()), bytes=path.stat().st_size,
               seconds=round(time.time() - t0, 2))
    return row


def load_index() -> dict[str, dict]:
    path = out_dir() / "index.jsonl"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["sha1"]: r for r in map(json.loads, f)}


def report() -> None:
    rows = list(load_index().values())
    ok = [r for r in rows if r.get("ok")]
    if not rows:
        print("no games yet")
        return
    print(f"games {len(ok)} ok, {len(rows) - len(ok)} failed")
    print(f"fight snapshots {sum(p['snapshots'] for r in ok for p in r['players']):,}, "
          f"unit rows {sum(r['units'] for r in ok):,}, commanded units {sum(r['commanded'] for r in ok):,}")
    print(f"disk {sum(r['bytes'] for r in ok) / 1e6:.0f} MB, resim {sum(r['seconds'] for r in ok) / max(1, len(ok)):.1f} s/game")
    for r in [r for r in rows if not r.get("ok")][:5]:
        print("  failed:", r["rel_path"], r.get("error"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--source", help="only games from this raw source folder (e.g. cwal), e.g. with --force after they were re-simulated")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    done = {} if args.force else load_index()
    todo = [r for r in select(args.matchup, args.source) if r["sha1"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} games to build ({len(done)} already done)", flush=True)
    out_dir().mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_dir() / "index.jsonl", "a", encoding="utf-8") as index, \
            ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(build_game, r) for r in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res.get("interrupted"):          # (killed, not failed: left out, so the next run builds it)
                continue
            index.write(json.dumps(res, ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(todo):
                index.flush()
                print(f"{n}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)
    report()


if __name__ == "__main__":
    main()
