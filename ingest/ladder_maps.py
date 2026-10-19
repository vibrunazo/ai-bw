"""The current ladder maps as files Gary's headless games can load: for each map of the modern era with enough
clean games, one of its replays rewritten in the 1.16 replay format (resim/scr_format.py to_legacy), which
OpenBW (gary/env.py Game.new) loads with its map, and the original map's hash beside it (<name>.maphash, the
replays' MapDataHash: what the macro model's map input knows; the conversion changes the bytes). League
rounds play them with eval/league.py --maps ladder. The map files stay in data/ (gitignored).

    python ingest/ladder_maps.py              # every modern map with MIN_GAMES clean games
    python ingest/ladder_maps.py --report     # what is there
Output: data/maps/ladder/<name>.rep and <name>.maphash
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "resim"))
from inventory import data_root  # noqa: E402
import scr_format  # noqa: E402

MIN_GAMES = 50                  # clean ladder games (source cwal) for a map to count as on the ladder
ASCII_NAMES = {"투혼": "Fighting Spirit"}   # (the engine opens ASCII paths only: the Korean name is Fighting Spirit)


def clean_name(raw: str) -> str:
    """A map's display name without colour codes, as an ASCII file name: '\\x07KnockOut \\x051.4' -> 'KnockOut 1.4'."""
    name = re.sub(r"[\x00-\x1f]", "", raw)
    for k, v in ASCII_NAMES.items():
        name = name.replace(k, v)
    name = re.sub(r"[<>:\"/\\|?*]", "-", name.encode("ascii", "ignore").decode())
    return re.sub(r"\s+", " ", name).strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-games", type=int, default=MIN_GAMES)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    out = data_root() / "maps" / "ladder"
    if args.report:
        for p in sorted(out.glob("*.rep")):
            print(p.stem, p.with_suffix(".maphash").read_text().strip()[:12])
        return
    inv = {}
    with open(data_root() / "interim" / "inventory.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("source") == "cwal" and r.get("map_hash"):
                inv[r["sha1"]] = r
    clean = {}
    with open(data_root() / "interim" / "resim_index.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["sha1"] in inv:
                clean[r["sha1"]] = bool(r.get("ok")) and r.get("desync_minute") is None
    games = collections.defaultdict(list)
    for sha, r in inv.items():
        if clean.get(sha):
            games[r["map_hash"]].append(r)
    out.mkdir(parents=True, exist_ok=True)
    for h, rs in sorted(games.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < args.min_games:
            continue
        name = clean_name(collections.Counter(r["map"] for r in rs).most_common(1)[0][0])
        for r in sorted(rs, key=lambda r: r["sha1"]):            # (the first that converts and matches its hash)
            data = (data_root() / "raw" / r["rel_path"]).read_bytes()
            try:
                if hashlib.sha1(scr_format.map_data(data)).hexdigest() != h:
                    continue
                legacy = scr_format.to_legacy(data)
            except Exception:  # noqa: BLE001 - try the next replay of the map
                continue
            (out / f"{name}.rep").write_bytes(legacy)
            (out / f"{name}.maphash").write_text(h + "\n")
            print(f"{name}: {len(rs)} clean games, from {r['rel_path']}")
            break
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
