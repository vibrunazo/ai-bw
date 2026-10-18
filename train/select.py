"""Which pro games a trainer may learn from: era, source and bot filters joined from the inventory.

The datasets (macro, army, fight, ...) are keyed by replay sha1 and built for a matchup; they don't
record where a replay came from. The inventory does: its "version" says "-1.16" (StarData and the
old TL packs, 2000s metas and maps) or "1.21+" (Remastered-era replays, e.g. cwal.gg), its "source"
is the raw folder, and its bot flags mark games a bot played. A trainer composes its selection with
`GameFilter`, so the same datasets can train an old-era, a modern-era or a mixed model.

Usage in a trainer:
    ap.add_argument(...); GameFilter.add_args(ap)
    keep = GameFilter.from_args(args)
    ... if keep(sha1): ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ingest"))
from inventory import data_root  # noqa: E402

ERAS = {"old": "-1.16", "modern": "1.21+"}


class GameFilter:
    """sha1 -> bool. Everything passes when no filter is set, except games a bot is known to have
    played (the source labels them), which never do."""

    def __init__(self, era: str | None = None, source: str | None = None, bots: bool = False):
        self.era, self.source, self.bots = era, source, bots
        self.rows: dict[str, dict] = {}
        with open(data_root() / "interim" / "inventory.jsonl", encoding="utf-8") as f:
            for r in map(json.loads, f):
                if r.get("ok") and r.get("sha1"):
                    self.rows[r["sha1"]] = r

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--era", choices=sorted(ERAS), help="old: 1.16 and before (StarData, TL packs); "
                        "modern: Remastered 1.21+ (cwal.gg). Default: both")
        ap.add_argument("--source", help="only replays from this raw folder, e.g. stardata, tl, cwal")
        ap.add_argument("--bots", action="store_true", help="keep games the inventory suspects a bot played")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GameFilter":
        return cls(getattr(args, "era", None), getattr(args, "source", None), getattr(args, "bots", False))

    def __call__(self, sha1: str) -> bool:
        r = self.rows.get(sha1)
        if r is None:
            return False
        if r.get("labeled_bot_game"):
            return False
        if not self.bots and r.get("has_bot_suspect"):
            return False
        if self.era and r.get("version") != ERAS[self.era]:
            return False
        if self.source and r.get("source") != self.source:
            return False
        return True

    def describe(self) -> str:
        return f"era={self.era or 'all'} source={self.source or 'all'} bots={'kept' if self.bots else 'dropped'}"
