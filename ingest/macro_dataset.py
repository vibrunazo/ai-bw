"""Training data for Gary's first learned model (macro: what to make next, and when).

Each clean, re-simulated game becomes one compressed file holding, for its two players, the game
state once per second *as that player knew it* and every production decision they made:

  frames  (T,)            game frame of each sample (every 24 frames = 1 s at Fastest)
  eco     (T, 2, 4)       minerals, gas, supply used x2, supply max x2 (BW counts half supply)
  units   (T, 2, 228, 2)  own units by type: all (incl. in production) / completed
  seen    (T, 2, 228)     other players' units this player can see right now (fog of war)
  cmds    (N, 6)          accepted production commands: frame, player (0/1), act, id, x, y
                          act: 0 train, 1 morph (larva), 2 building morph, 3 build,
                               4 research, 5 upgrade, 6 expand (a town hall away from the
                               player's existing ones; x, y = tile). Builds are the buildings
                               that really started, timed at the player's first order for
                               them; spam and failed orders are dropped.

A model samples (state at t, the player's next decision after t, and how long until it) from
these. Own state is complete (a player knows their own units); the opponent only through "seen",
so the model learns from what the player could actually know. The player's build style (the
taxonomy cluster, data/interim/taxonomy) is in the index, for conditioning ("play 2-hatch muta").

Only games that re-simulate cleanly are used (resim_index.jsonl: ok, no desync).

Output: data/interim/macro/v1/<sha1[:2]>/<sha1>.npz and data/interim/macro/v1/index.jsonl.

Usage:
  python ingest/macro_dataset.py --matchup TvZ --limit 200
  python ingest/macro_dataset.py --matchup TvZ                # everything (resumable)
  python ingest/macro_dataset.py --report
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "resim"))
import scr_format  # noqa: E402  (resim/scr_format.py)

VERSION = "v1"
EVERY = 24                       # one sample per second of game time
N_TYPES = 228
TOWN_HALLS = {106, 131, 132, 133, 154}
EXPAND_PX = 12 * 32              # a town hall this far from all of a player's others = expansion
BUILD_MATCH_S = 90               # orders for a building count up to this long before it started ...
BUILD_MATCH_PX = 6 * 32          # ... and this close to where it started
REPEAT_S = 10                    # repeats of a morph / research / upgrade order within this collapse
ACTS = {"train": 0, "morph": 1, "bmorph": 2, "build": 3, "research": 4, "upgrade": 5}
ACT_EXPAND = 6
RACES = {0: "Z", 1: "T", 2: "P"}


def out_dir() -> Path:
    return data_root() / "interim" / "macro" / VERSION


def resim_exe() -> Path:
    exe = REPO_ROOT / "resim" / "build" / ("gary_resim.exe" if os.name == "nt" else "gary_resim")
    if not exe.exists():
        sys.exit("gary_resim not built: run resim/build.bat (see resim/README.md)")
    return exe


def load_styles(matchup: str) -> dict[str, int]:
    """'rel_path|player name' -> build style, for every race in the matchup: a race's plan label
    (taxonomy/plans.py, every game has one) where there is one, else its tech cluster
    (taxonomy/discover.py)."""
    styles: dict[str, int] = {}
    folder = data_root() / "interim" / "taxonomy"
    for path in folder.glob(f"{matchup}_*_tech.json"):
        plan = folder / path.name.replace("_tech.json", "_plan.json")
        styles.update(json.loads((plan if plan.exists() else path).read_text(encoding="utf-8")).get("assignments", {}))
    return styles


def restyle(matchup: str) -> None:
    """Rewrite the index's player styles from the current taxonomy, without rebuilding the games."""
    styles = load_styles(matchup)
    path = out_dir() / "index.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for r in rows:
        if r.get("matchup") != matchup:
            continue
        for p in r.get("players", []):
            new = styles.get(f"{r['rel_path']}|{p['name']}")
            changed += new != p.get("style")
            p["style"] = new
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    print(f"restyled: {changed} player styles changed")


def run_resim(rel_path: str) -> list[dict]:
    data = (data_root() / "raw" / rel_path).read_bytes()
    args = [str(resim_exe()), "--data", str(data_root() / "gamedata" / "scr"), "--replay", "-",
            "--every", str(EVERY)]
    if scr_format.replay_format(data) != "legacy":
        args += ["--flat", "--unit-limit", str(scr_format.unit_limit(data))]
        data = scr_format.to_flat(data)
    proc = subprocess.run(args, input=data, capture_output=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()[:300])
    return [json.loads(line) for line in proc.stdout.decode("utf-8", "replace").splitlines()]


def production_decisions(lines: list[dict], pidx: dict[int, int]) -> list[tuple]:
    """(frame, player, act, id, x, y) for every production decision, in time order.

    Players spam-click and retry, so build orders are matched to the buildings that really
    started: each building absorbs the player's orders for that type near its spot in the
    BUILD_MATCH_S before it started, and the decision is the earliest of them (when the player
    decided). A building with no matching order still counts, when it started. Orders that never
    led to a building (failed, cancelled) are dropped. A town hall away from all of the player's
    others is an expansion. Repeated building morphs / research / upgrades collapse into the first.
    """
    out = []
    builds: list[dict] = []
    last_order: dict[tuple, int] = {}
    for r in lines:
        if r["type"] != "cmd" or r["slot"] not in pidx:
            continue
        act, f = ACTS[r["act"]], r["frame"]
        if act == ACTS["build"]:
            builds.append({"f": f, "slot": r["slot"], "id": r["id"],
                           "x": r["x"] * 32 + 32, "y": r["y"] * 32 + 32, "used": False})
            continue
        if act in (ACTS["bmorph"], ACTS["research"], ACTS["upgrade"]):
            key = (r["slot"], act, r["id"])
            if f - last_order.get(key, -10**9) < REPEAT_S * 24:
                continue
            last_order[key] = f
        out.append((f, pidx[r["slot"]], act, r["id"], r.get("x", -1), r.get("y", -1)))

    # town halls over time, to tell expansions from macro hatcheries
    halls: dict[int, dict] = {}                     # tag -> {slot, x, y, from, to}
    for r in lines:
        if r["type"] == "event" and r.get("unit") in TOWN_HALLS and r["ev"] in ("start", "done"):
            halls.setdefault(r["tag"], {"slot": r["slot"], "x": r["x"], "y": r["y"], "from": r["frame"], "to": 10**9})
        elif r["type"] == "event" and r["ev"] == "gone" and r["tag"] in halls:
            halls[r["tag"]]["to"] = r["frame"]

    built_ids = {(b["slot"], b["id"]) for b in builds}
    for e in lines:
        if not (e["type"] == "event" and e["ev"] == "start" and e["frame"] > 1
                and e["slot"] in pidx and (e["slot"], e["unit"]) in built_ids):
            continue
        near = [b for b in builds
                if not b["used"] and b["slot"] == e["slot"] and b["id"] == e["unit"]
                and e["frame"] - BUILD_MATCH_S * 24 <= b["f"] <= e["frame"]
                and (b["x"] - e["x"]) ** 2 + (b["y"] - e["y"]) ** 2 <= BUILD_MATCH_PX ** 2]
        for b in near:
            b["used"] = True
        f = min((b["f"] for b in near), default=e["frame"])
        act = ACTS["build"]
        if e["unit"] in TOWN_HALLS:
            others = [h for tag, h in halls.items() if tag != e["tag"] and h["slot"] == e["slot"]
                      and h["from"] < e["frame"] < h["to"]]
            if all((h["x"] - e["x"]) ** 2 + (h["y"] - e["y"]) ** 2 > EXPAND_PX ** 2 for h in others):
                act = ACT_EXPAND
        out.append((f, pidx[e["slot"]], act, e["unit"], e["x"] // 32, e["y"] // 32))
    out.sort()
    return out


def build_game(rec: dict, styles: dict[str, int]) -> dict:
    """Re-simulate one game and write its arrays. Returns its index row."""
    row = {"sha1": rec["sha1"], "rel_path": rec["rel_path"], "matchup": rec["matchup"]}
    t0 = time.time()
    try:
        lines = run_resim(rec["rel_path"])
    except Exception as e:  # noqa: BLE001 - one bad replay must not stop the batch
        row.update(ok=False, error=str(e)[:300])
        return row
    header = next(r for r in lines if r["type"] == "header")
    snaps = [r for r in lines if r["type"] == "snapshot"]
    # the two real players: observers send commands too, but the engine accepts almost none
    accepted: dict[int, int] = {}
    for s in snaps:
        for p in s["players"]:
            accepted[p["slot"]] = accepted.get(p["slot"], 0) + p["accepted"]
    slots = sorted(sorted(accepted, key=lambda s: -accepted[s])[:2])
    if len(slots) != 2 or not snaps:
        row.update(ok=False, error="not two players")
        return row
    pidx = {s: i for i, s in enumerate(slots)}
    names = {p["slot"]: p for p in header["players"]}

    T = len(snaps)
    frames = np.array([s["frame"] for s in snaps], dtype=np.int32)
    eco = np.zeros((T, 2, 4), dtype=np.int32)
    units = np.zeros((T, 2, N_TYPES, 2), dtype=np.uint16)
    seen = np.zeros((T, 2, N_TYPES), dtype=np.uint16)
    for t, s in enumerate(snaps):
        for p in s["players"]:
            i = pidx.get(p["slot"])
            if i is None:
                continue
            eco[t, i] = (p["minerals"], p["gas"], round(p["supply_used"] * 2), round(p["supply_max"] * 2))
            for k, (n_all, n_done) in p["units"].items():
                units[t, i, int(k)] = (n_all, n_done)
            for k, n in p.get("seen", {}).items():
                seen[t, i, int(k)] = n

    cmds = production_decisions(lines, pidx)
    cmds_arr = np.array(cmds, dtype=np.int32).reshape(-1, 6)

    path = out_dir() / rec["sha1"][:2] / f"{rec['sha1']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=frames, eco=eco, units=units, seen=seen, cmds=cmds_arr)
    path.write_bytes(buf.getvalue())

    players = []
    for s in slots:
        name = names.get(s, {}).get("name", "")
        players.append({"slot": s, "name": name, "race": RACES.get(names.get(s, {}).get("race"), "?"),
                        "style": styles.get(f"{rec['rel_path']}|{name}")})
    row.update(ok=True, map=header.get("map"), end_frame=header.get("end_frame"), samples=T,
               decisions=len(cmds), players=players, bytes=path.stat().st_size,
               seconds=round(time.time() - t0, 2))
    return row


def select(matchup: str, source: str | None = None) -> list[dict]:
    games = []
    with open(data_root() / "interim" / "resim_index.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if ((source is None or r.get("rel_path", "").split("/")[0] == source) and r.get("matchup") == matchup and r.get("ok") and r.get("desync_minute") is None
                    and r.get("frames_match") is not False):
                games.append(r)
    return games


def load_index() -> dict[str, dict]:
    path = out_dir() / "index.jsonl"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["sha1"]: r for r in map(json.loads, f)}


def report() -> None:
    rows = [r for r in load_index().values()]
    ok = [r for r in rows if r.get("ok")]
    if not rows:
        print("no games yet")
        return
    styled = sum(1 for r in ok for p in r["players"] if p["style"] is not None)
    print(f"games {len(ok)} ok, {len(rows) - len(ok)} failed")
    print(f"samples {sum(r['samples'] for r in ok):,} player-seconds x2, "
          f"decisions {sum(r['decisions'] for r in ok):,}")
    print(f"players with a build style {styled}/{2 * len(ok)}")
    print(f"disk {sum(r['bytes'] for r in ok) / 1e6:.0f} MB, "
          f"resim {sum(r['seconds'] for r in ok) / max(1, len(ok)):.1f} s/game")
    for r in [r for r in rows if not r.get("ok")][:5]:
        print("  failed:", r["rel_path"], r.get("error"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--force", action="store_true", help="rebuild games already in the index")
    ap.add_argument("--source", help="only games from this raw source folder (e.g. cwal), e.g. with --force after they were re-simulated")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--restyle", action="store_true", help="only rewrite the index's styles from the taxonomy")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.restyle:
        restyle(args.matchup)
        return
    done = {} if args.force else load_index()
    todo = [r for r in select(args.matchup, args.source) if r["sha1"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    styles = load_styles(args.matchup)
    print(f"{len(todo)} games to build ({len(done)} already done)", flush=True)
    out_dir().mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_dir() / "index.jsonl", "a", encoding="utf-8") as index, \
            ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(build_game, r, styles) for r in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            index.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(todo):
                index.flush()
                print(f"{n}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)
    report()


if __name__ == "__main__":
    main()
