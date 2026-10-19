"""Re-simulate many replays with gary_resim and summarize each run.

For every selected replay (from data/interim/inventory.jsonl):
  - run gary_resim (events plus a snapshot every 10 s of game time)
  - store its output compressed in data/interim/resim/<sha1[:2]>/<sha1>.jsonl.zz
  - append a summary to data/interim/resim_index.jsonl: whether it ran, frames simulated, the
    rejected-command rate per player per minute, and the first minute that looks desynced

Desync heuristic: human spam gives a low, steady rate of rejected commands. When the simulation
drifts from what actually happened, the player's commands stop making sense (selected units
don't exist, buildings can't be placed) and the rate jumps and stays high. Only the two real
players count (observers' commands are always rejected). A minute is "bad" if at least
DESYNC_MIN_ACTIONS commands were issued and more than DESYNC_RATE of them were rejected. The
suspected desync point is the first run of DESYNC_RUN bad minutes after which at least
DESYNC_STAYS of the remaining minutes are bad too: a desync never recovers, early-game spam does.

Usage:
  python resim/run_batch.py --matchup TvZ --limit 300
  python resim/run_batch.py --source tl --version-hint     # only replays with an old-patch hint
  python resim/run_batch.py --remastered                   # only Remastered-format replays
  python resim/run_batch.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))
from inventory import data_root, load_inventory  # noqa: E402

import scr_format  # noqa: E402  (same folder)
sys.path.insert(0, str(REPO_ROOT))
from eval import monitor, priority  # noqa: E402

SNAPSHOT_EVERY = 240          # frames (10 s)
FRAMES_PER_MIN = 24 * 60
DESYNC_RATE = 0.35
DESYNC_MIN_ACTIONS = 20
DESYNC_RUN = 2
DESYNC_STAYS = 0.7


def resim_exe() -> Path:
    exe = REPO_ROOT / "resim" / "build" / ("gary_resim.exe" if os.name == "nt" else "gary_resim")
    if not exe.exists():
        sys.exit("gary_resim not built: run resim/build.bat (see resim/README.md)")
    return exe


def summarize(lines: list[str]) -> dict:
    per_min: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    events = 0
    end_frame = None
    for line in lines:
        rec = json.loads(line)
        if rec["type"] == "snapshot":
            minute = (rec["frame"] - 1) // FRAMES_PER_MIN
            for p in rec["players"]:
                per_min[p["slot"]][minute][0] += p["accepted"]
                per_min[p["slot"]][minute][1] += p["rejected"]
        elif rec["type"] == "event":
            events += 1
        elif rec["type"] == "end":
            end_frame = rec["frame"]
    # the two real players: observers issue commands too, but the engine accepts almost none
    accepted = {slot: sum(a for a, _ in mins.values()) for slot, mins in per_min.items()}
    players = sorted(accepted, key=lambda s: -accepted[s])[:2]
    desync_minute = None
    rates = {}
    for slot in players:
        mins = per_min[slot]
        minutes = sorted(mins)
        series = [round(mins[m][1] / sum(mins[m]), 3) if sum(mins[m]) else 0.0 for m in minutes]
        bad = [sum(mins[m]) >= DESYNC_MIN_ACTIONS and r > DESYNC_RATE for m, r in zip(minutes, series)]
        for i in range(len(bad) - DESYNC_RUN + 1):
            if all(bad[i:i + DESYNC_RUN]) and sum(bad[i:]) >= DESYNC_STAYS * len(bad[i:]):
                m = minutes[i]
                desync_minute = m if desync_minute is None else min(desync_minute, m)
                break
        rates[str(slot)] = series
    return {"end_frame": end_frame, "events": events, "rejected_rate_by_minute": rates,
            "desync_minute": desync_minute}


def run_one(exe: Path, gamedata: Path, out_root: Path, rec: dict) -> dict:
    replay = data_root() / "raw" / rec["rel_path"]
    t0 = time.time()
    data = replay.read_bytes()
    row = {"sha1": rec["sha1"], "rel_path": rec["rel_path"], "matchup": rec.get("matchup"),
           "version_hint": rec.get("version_hint", "")}
    # the replay goes through stdin: its path may not survive the Windows ANSI command line
    args = [str(exe), "--data", str(gamedata), "--replay", "-", "--every", str(SNAPSHOT_EVERY)]
    try:
        row["format"] = scr_format.replay_format(data)
        if row["format"] != "legacy":  # Remastered-era: decode for OpenBW (resim/scr_format.py)
            row["unit_limit"] = scr_format.unit_limit(data)
            bfix = scr_format.remastered_sections(data).get("BFIX")
            if bfix:
                row["bfix"] = bfix.hex()
            data = scr_format.to_flat(data)
            args += ["--flat", "--unit-limit", str(row["unit_limit"])]
    except ValueError as e:
        row.update(ok=False, error=f"decode: {e}"[:300], seconds=0)
        return row
    proc = subprocess.run(args, input=data, capture_output=True, timeout=600)
    row["seconds"] = round(time.time() - t0, 2)
    if proc.returncode != 0:
        row.update(ok=False, error=proc.stderr.decode("utf-8", "replace").strip()[:300])
        return row
    out = proc.stdout
    path = out_root / rec["sha1"][:2] / f"{rec['sha1']}.jsonl.zz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(out, 6))
    row.update(ok=True, **summarize(out.decode("utf-8", "replace").splitlines()))
    row["frames_match"] = row["end_frame"] == rec.get("frames")
    return row


def select(args: argparse.Namespace) -> list[dict]:
    recs = load_inventory(data_root() / "interim" / "inventory.jsonl").values()
    games = [r for r in recs if r.get("ok") and not r.get("dup_of") and r.get("is_1v1_human")
             and not r.get("has_bot_suspect") and not r.get("labeled_bot_game")]
    if args.matchup:
        games = [r for r in games if r["matchup"] == args.matchup]
    if args.source:
        games = [r for r in games if r["source"] == args.source]
    if args.version_hint:
        games = [r for r in games if r.get("version_hint")]
    if args.remastered:
        games = [r for r in games if r.get("version", "").startswith("1.")]
    games.sort(key=lambda r: r["rel_path"])
    return games[: args.limit] if args.limit else games


def build(args: argparse.Namespace) -> None:
    exe = resim_exe()
    gamedata = Path(args.data) if args.data else data_root() / "gamedata" / "scr"
    out_root = data_root() / "interim" / "resim"
    index_path = data_root() / "interim" / "resim_index.jsonl"
    done = {}
    if index_path.exists():
        done = {r["sha1"]: r for r in map(json.loads, index_path.open(encoding="utf-8"))}
    games = [g for g in select(args) if args.force or g["sha1"] not in done
             or (args.retry_failed and not done[g["sha1"]]["ok"])]
    print(f"{len(games)} replays to simulate ({len(done)} already in the index)", flush=True)
    def save() -> None:
        tmp = index_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in sorted(done.values(), key=lambda r: r["rel_path"]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(index_path)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, exe, gamedata, out_root, g) for g in games]
        for n, fut in enumerate(monitor.track(as_completed(futures), len(futures), "replays re-simulated"), 1):
            row = fut.result()
            done[row["sha1"]] = row
            if n % 500 == 0 or n == len(games):
                save()  # an interrupted run keeps its progress; re-running skips finished replays
                print(f"  {n}/{len(games)}  ({time.time() - t0:.0f} s)", flush=True)
    save()
    print(f"wrote {index_path}")
    report(args)


def report(args: argparse.Namespace) -> None:
    index_path = data_root() / "interim" / "resim_index.jsonl"
    rows = [json.loads(line) for line in index_path.open(encoding="utf-8")]
    if args.matchup:
        rows = [r for r in rows if r.get("matchup") == args.matchup]
    ok = [r for r in rows if r["ok"]]
    print(f"\n{len(rows)} simulated, {len(ok)} ran to the end, {len(rows) - len(ok)} failed")
    if not ok:
        return
    groups = defaultdict(list)
    for r in ok:
        src = r["rel_path"].split("/")[0]
        if r.get("format", "legacy") != "legacy":
            src += f" (Remastered, {r.get('unit_limit', 1700)} units, bfix {r.get('bfix', '-')[:2]})"
        groups[f"{src} {'(old patch ' + r['version_hint'] + ')' if r['version_hint'] else ''}".strip()].append(r)
    print(f"\n{'group':<40} {'games':>6} {'desync':>8}  first desync minute (median)")
    for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ds = sorted(r["desync_minute"] for r in rs if r["desync_minute"] is not None)
        med = ds[len(ds) // 2] if ds else "-"
        print(f"{g:<40} {len(rs):>6} {len(ds) / len(rs):>7.1%}  {med}")
    rates = sorted(x for r in ok for s in r["rejected_rate_by_minute"].values() for x in s)
    q = lambda p: rates[int(p * (len(rates) - 1))]
    print(f"\nrejected-command rate per player-minute: p50 {q(.5):.1%}, p90 {q(.9):.1%}, p99 {q(.99):.1%}")
    secs = sum(r["seconds"] for r in ok)
    print(f"simulation time: {secs / len(ok):.2f} s per game on average")
    errs = Counter(r.get("error", "")[:80] for r in rows if not r["ok"])
    for e, n in errs.most_common(5):
        print(f"  failed {n}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--matchup")
    ap.add_argument("--source")
    ap.add_argument("--version-hint", action="store_true", help="only replays with an old-patch hint")
    ap.add_argument("--remastered", action="store_true", help="only Remastered-format (1.18+) replays")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="re-simulate replays already in the index")
    ap.add_argument("--retry-failed", action="store_true", help="re-simulate replays that failed before")
    ap.add_argument("--data", help="game data folder (default: data/gamedata/scr)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report(args) if args.report else build(args)


if __name__ == "__main__":
    main()
