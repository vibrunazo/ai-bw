"""Replay inventory: one row per replay file under data/raw/.

Walks data/raw/, hashes every .rep file, parses it with screp, and writes:
  data/interim/inventory.jsonl   one JSON record per replay (players nested)
  data/interim/inventory.csv     flat summary for quick viewing

Re-runs are incremental: replays whose hash is already in the inventory are not re-parsed.
Chat message text is never stored, only a count.

Usage:
  python ingest/inventory.py              # build or update the inventory
  python ingest/inventory.py --report     # print a summary of the existing inventory
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAME_MS = 42  # one game frame at Fastest speed
SCREP_TIMEOUT_S = 60


def data_root() -> Path:
    return Path(os.environ.get("GARY_DATA", REPO_ROOT / "data"))


def find_screp(explicit: str | None) -> str:
    candidates = [explicit] if explicit else [
        str(REPO_ROOT / "tools" / "bin" / ("screp.exe" if os.name == "nt" else "screp")),
        shutil.which("screp"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    sys.exit("screp not found: put it in tools/bin/ or on PATH, or pass --screp")


def source_of(rel: Path) -> tuple[str, str]:
    """Source id and pack from a path relative to data/raw/ (see data/sources.yaml)."""
    parts = rel.parts
    source, pack = parts[0], ""
    if source == "stardata" or len(parts) <= 2:
        return source, pack  # stardata subfolders are just numbered chunks
    pack = parts[1]
    if source == "tl" and pack == "progames" and len(parts) > 3:
        pack = f"progames/{parts[2]}"  # tl/progames/<subpack>/...
    return source, pack


VERSION_DIR = re.compile(r"^1(0\d|1[0-6])$")                     # "108" -> 1.08
VERSION_TEXT = re.compile(r"(?<![\d.])1\.(0\d|1[0-6])(?:\.\d)?(?:\s*-\s*1\.(0\d|1[0-6]))?(?![\d])")  # 1.00-1.16 only


def version_hint(rel: Path) -> str:
    """Patch version suggested by folder or file names (e.g. "108/", "1.09-1.11/", "1.15_x.rep").
    Replay headers don't store the exact pre-1.18 patch, so this is the only clue we have."""
    for part in rel.parts:
        if m := VERSION_DIR.match(part):
            return f"1.{m.group(1)}"
        if m := VERSION_TEXT.search(part):
            return f"1.{m.group(1)}" + (f"-1.{m.group(2)}" if m.group(2) else "")
    return ""


# Files that are never replays. Anything else under raw/ is tried, because old packs often
# have replays with mangled extensions ("Dreamt.xenarep", "Sea.FireFist", "re.p").
NOT_REPLAY_EXT = {
    ".rar", ".zip", ".7z", ".gz", ".tar", ".part", ".lnk", ".exe", ".dll", ".db", ".txt", ".nfo",
    ".json", ".list", ".tcr", ".jpg", ".png", ".gif", ".htm", ".html", ".url", ".ini",
}
NOT_REPLAY_NAMES = {"repcache", ".DS_Store", "Thumbs.db", "desktop.ini"}
MAX_REPLAY_BYTES = 20_000_000


def is_candidate(p: Path) -> bool:
    if p.suffix.lower() == ".rep":
        return True
    if p.name in NOT_REPLAY_NAMES or p.suffix.lower() in NOT_REPLAY_EXT:
        return False
    return 1_000 < p.stat().st_size < MAX_REPLAY_BYTES


def sha1_of(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def matchup_of(players: list[dict]) -> str:
    """Canonical matchup for 1v1 games, races sorted (PvT, PvZ, TvZ, TvT, ...)."""
    letters = sorted(p["race"] for p in players)
    return "v".join(letters) if len(letters) == 2 else ""


def parse_replay(screp: str, path: Path, raw_root: Path) -> dict:
    rel = path.relative_to(raw_root)
    source, pack = source_of(rel)
    rec: dict = {
        "sha1": sha1_of(path),
        "source": source,
        "pack": pack,
        "rel_path": rel.as_posix(),
        "size": path.stat().st_size,
        "ok": False,
        "error": "",
    }
    try:
        proc = subprocess.run(
            [screp, "-indent=false", "-mapDataHash", "sha1", str(path)],
            capture_output=True, timeout=SCREP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        rec["error"] = "timeout"
        return rec
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        err = re.sub(r"^\d{4}/\d\d/\d\d \d\d:\d\d:\d\d ", "", err)  # drop Go log timestamp
        rec["error"] = (err or f"exit {proc.returncode}")[:300]
        return rec
    try:
        d = json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as e:
        rec["error"] = f"bad json: {e}"[:300]
        return rec

    h = d.get("Header") or {}
    c = d.get("Computed") or {}
    descs = {pd["PlayerID"]: pd for pd in (c.get("PlayerDescs") or [])}

    players = []
    for p in h.get("Players") or []:
        if p.get("Observer"):
            continue
        pd = descs.get(p.get("ID"), {})
        start = pd.get("StartLocation") or {}
        players.append({
            "name": p.get("Name", ""),
            "race": chr((p.get("Race") or {}).get("Letter") or ord("?")),  # T, Z, P, R(andom)
            "team": p.get("Team"),
            "type": (p.get("Type") or {}).get("Name", ""),
            "apm": pd.get("APM"),
            "eapm": pd.get("EAPM"),
            "cmd_count": pd.get("CmdCount"),
            "start_x": start.get("X"),
            "start_y": start.get("Y"),
            "start_clock": pd.get("StartDirection"),  # 1-12, clock position of the start location
        })

    humans = [p for p in players if p["type"] == "Human"]
    frames = h.get("Frames") or 0
    spawn_dist = clock_diff = None
    if len(players) == 2 and all(p["start_x"] is not None for p in players):
        a, b = players
        spawn_dist = round(((a["start_x"] - b["start_x"]) ** 2 + (a["start_y"] - b["start_y"]) ** 2) ** 0.5)
        if a["start_clock"] and b["start_clock"]:
            diff = abs(a["start_clock"] - b["start_clock"]) % 12
            clock_diff = min(diff, 12 - diff)  # 6 = opposite (cross), 3 = adjacent on 4-player maps

    rec.update({
        "ok": True,
        "engine": (h.get("Engine") or {}).get("ShortName", ""),
        "version": h.get("Version", ""),
        "frames": frames,
        "duration_s": round(frames * FRAME_MS / 1000),
        "start_time": h.get("StartTime", ""),
        "map": (h.get("Map") or "").strip(),
        "map_w": h.get("MapWidth"),
        "map_h": h.get("MapHeight"),
        "map_hash": ((d.get("Custom") or {}).get("MapDataHash") or ""),
        "game_type": (h.get("Type") or {}).get("ShortName", ""),
        "speed": (h.get("Speed") or {}).get("Name", ""),
        "n_players": len(players),
        "n_humans": len(humans),
        "is_1v1_human": len(players) == 2 and len(humans) == 2,
        "matchup": matchup_of(players),
        "winner_team": c.get("WinnerTeam") or None,
        "spawn_dist_px": spawn_dist,
        "spawn_clock_diff": clock_diff,
        "chat_count": len(c.get("ChatCmds") or []),
        "players": players,
    })
    return rec


def load_inventory(path: Path) -> dict[str, dict]:
    """Existing records keyed by rel_path."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["rel_path"]: r for r in map(json.loads, f) if r.get("rel_path")}


# Bot detection (heuristic). APM alone isn't the signal: 400+ APM pros exist. The tell is the
# effective-APM ratio: humans spam (EAPM/APM ~0.77 median, 0.96 at p99), bots almost never do.
# Calibrated on STARDATA and on cwal games the site tags as Pluto (Pluto--AI plays at 320-1130 APM,
# always with a ratio of 0.99-1.00). Very high EAPM is flagged regardless of ratio, because some
# bots pad their APM with spam (one Pluto variant: 2083 APM, 1461 EAPM, ratio 0.70).
BOT_RATIO_MIN = 0.98
BOT_RATIO_EAPM_MIN = 300
BOT_EAPM_ANY = 800
KNOWN_BOT_NAMES = {
    "letabot", "martin rooijackers", "martin rooijackers bot", "tscmoo", "krasi0", "krasi1",
    "krasimir krystev", "ximp", "ualbertabot", "andrew smith", "marian devecka", "mariandevecka",
    "icecraft", "skynet", "aiur", "steamhammer", "purplewave", "mcrave",
    "pluto--ai", "pluto_promax", "plutopromax",  # cwal-tagged ladder accounts; not plain "pluto"
    "namepen99", "morokei1", "morokei2", "morokei3", "dreamdawn",  # 2026 ladder bots (flagged on their cwal.gg profiles, confirmed by watching)
}


def derive(rec: dict) -> None:
    """Fields computed from parsed data; recomputed on every run, so rules can change cheaply."""
    rec["version_hint"] = version_hint(Path(rec["rel_path"]))
    if not rec.get("ok"):
        return
    for p in rec["players"]:
        apm, eapm = p.get("apm") or 0, p.get("eapm") or 0
        name = p.get("name", "").strip().lower()
        by_apm = (eapm >= BOT_RATIO_EAPM_MIN and eapm / apm >= BOT_RATIO_MIN) or eapm >= BOT_EAPM_ANY
        by_name = name in KNOWN_BOT_NAMES
        p["bot_suspect"] = bool(by_apm or by_name or p.get("type") == "Computer")
    rec["has_bot_suspect"] = any(p["bot_suspect"] for p in rec["players"])
    rec["labeled_bot_game"] = "/pluto/" in f"/{rec['rel_path'].lower()}"  # games tagged as bots by the source


def game_key(r: dict) -> str:
    """Identifies a game regardless of which player saved the replay (different files, same game)."""
    if not r.get("ok") or not r.get("start_time"):
        return ""
    names = sorted(p["name"] for p in r["players"])
    return f"{r['start_time']}|{r['map_hash']}|{r['frames'] // 100}|{'|'.join(names)}"


def mark_duplicates(records: list[dict]) -> None:
    """dup_of: first file with identical bytes, or the same game saved by another player."""
    first_file: dict[str, str] = {}
    first_game: dict[str, str] = {}
    for r in sorted(records, key=lambda r: r["rel_path"]):
        orig = first_file.setdefault(r["sha1"], r["rel_path"])
        key = game_key(r)
        if orig == r["rel_path"] and key:
            orig = first_game.setdefault(key, r["rel_path"])
        r["dup_of"] = "" if orig == r["rel_path"] else orig


CSV_FIELDS = [
    "sha1", "source", "pack", "rel_path", "version_hint", "size", "ok", "error", "dup_of", "engine",
    "version",
    "frames", "duration_s", "start_time", "map", "map_w", "map_h", "map_hash", "game_type",
    "speed", "n_players", "n_humans", "is_1v1_human", "matchup", "winner_team", "spawn_dist_px",
    "spawn_clock_diff", "chat_count", "has_bot_suspect", "labeled_bot_game",
    "p1_name", "p1_race", "p1_apm", "p1_eapm", "p1_start_clock",
    "p2_name", "p2_race", "p2_apm", "p2_eapm", "p2_start_clock",
]


def write_outputs(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: r["rel_path"])
    with (out_dir / "inventory.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            for i, p in enumerate((r.get("players") or [])[:2], 1):
                for k in ("name", "race", "apm", "eapm", "start_clock"):
                    row[f"p{i}_{k}"] = p.get(k)
            w.writerow(row)


def build(args: argparse.Namespace) -> None:
    root = data_root()
    raw_root = root / "raw"
    out_dir = root / "interim"
    screp = find_screp(args.screp)

    files = sorted(p for p in raw_root.rglob("*") if p.is_file() and is_candidate(p))
    if args.source:
        files = [p for p in files if p.relative_to(raw_root).parts[0] == args.source]
    if args.limit:
        files = files[: args.limit]

    existing = load_inventory(out_dir / "inventory.jsonl")
    todo, records = [], []
    for p in files:
        rel = p.relative_to(raw_root).as_posix()
        old = existing.get(rel)
        if old and old.get("size") == p.stat().st_size and not args.force:
            records.append(old)
        else:
            todo.append(p)
    # keep records for files outside this run's filter
    seen = {r["rel_path"] for r in records} | {p.relative_to(raw_root).as_posix() for p in todo}
    if args.source or args.limit:
        records += [r for k, r in existing.items() if k not in seen and (raw_root / k).exists()]

    print(f"{len(files)} replays found, {len(todo)} to parse, {len(records)} cached", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(parse_replay, screp, p, raw_root): p for p in todo}
        for fut in as_completed(futures):
            try:
                records.append(fut.result())
            except Exception as e:  # never lose a whole run to one odd replay
                p = futures[fut]
                source, pack = source_of(p.relative_to(raw_root))
                records.append({"sha1": sha1_of(p), "source": source, "pack": pack,
                                "rel_path": p.relative_to(raw_root).as_posix(),
                                "size": p.stat().st_size, "ok": False, "error": f"inventory: {e!r}"[:300]})
            done += 1
            if done % 2000 == 0 or done == len(todo):
                print(f"  parsed {done}/{len(todo)}", flush=True)

    for r in records:
        derive(r)
    mark_duplicates(records)
    write_outputs(records, out_dir)
    print(f"wrote {out_dir / 'inventory.jsonl'} and inventory.csv ({len(records)} records)")


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def report(args: argparse.Namespace) -> None:
    records = list(load_inventory(data_root() / "interim" / "inventory.jsonl").values())
    if not records:
        sys.exit("no inventory yet: run without --report first")

    def table(title: str, counter: Counter, top: int = 15) -> None:
        print(f"\n{title}")
        for k, v in counter.most_common(top):
            print(f"  {v:>7}  {k}")

    ok = [r for r in records if r["ok"]]
    uniq = [r for r in ok if not r.get("dup_of")]
    print(f"records: {len(records)}   parsed: {len(ok)}   failed: {len(records) - len(ok)}   "
          f"duplicates: {sum(1 for r in ok if r.get('dup_of'))}   unique parsed: {len(uniq)}")

    table("by source/pack (unique parsed)", Counter(f"{r['source']}/{r['pack']}".rstrip("/") for r in uniq))
    table("by version", Counter(r["version"] for r in uniq))
    table("pre-1.16 version hints from folder/file names", Counter(r.get("version_hint") or "none" for r in uniq))
    table("by game type", Counter(r["game_type"] for r in uniq))
    bots = [r for r in uniq if r.get("has_bot_suspect")]
    print(f"\ngames with a suspected bot player: {len(bots)}")
    table("suspected bot names", Counter(p["name"] for r in bots for p in r["players"] if p["bot_suspect"]), top=10)
    labeled = [r for r in uniq if r.get("labeled_bot_game")]
    if labeled:
        print(f"source-labeled bot games: {len(labeled)}, caught by the heuristic: "
              f"{sum(1 for r in labeled if r.get('has_bot_suspect'))}")
    v1 = [r for r in uniq if r["is_1v1_human"] and not r.get("has_bot_suspect")
          and not r.get("labeled_bot_game")]
    print(f"\n1v1 human games (no suspected bots): {len(v1)}")
    table("1v1 matchups", Counter(r["matchup"] for r in v1))
    table("1v1 matchups by source", Counter(f"{r['source']:<9} {r['matchup']}" for r in v1), top=40)

    tvz = [r for r in v1 if r["matchup"] == "TvZ"]
    durs = [r["duration_s"] for r in tvz]
    print(f"\nTvZ: {len(tvz)} games; duration p10/p50/p90 = "
          f"{pct(durs, .1) // 60}m / {pct(durs, .5) // 60}m / {pct(durs, .9) // 60}m; "
          f"winner known: {sum(1 for r in tvz if r['winner_team'])}; "
          f"under 3 min: {sum(1 for d in durs if d < 180)}")
    table("TvZ top maps", Counter(r["map"] for r in tvz), top=10)

    fails = [r for r in records if not r["ok"]]
    if fails:
        table("parse errors", Counter(r["error"][:80] for r in fails), top=10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="summarize the existing inventory")
    ap.add_argument("--source", help="only scan this top-level source folder (e.g. stardata)")
    ap.add_argument("--limit", type=int, help="only scan the first N files (for testing)")
    ap.add_argument("--force", action="store_true", help="re-parse files already in the inventory")
    ap.add_argument("--screp", help="path to the screp binary")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # map/player names are often Korean
    report(args) if args.report else build(args)


if __name__ == "__main__":
    main()
