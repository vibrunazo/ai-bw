"""Build styles per matchup and race (issue #54): what a style is called, how often the modern pros play it, and which
style a game's build is. The catalogue is data/interim/taxonomy/v2/<matchup>_<race>.json, written by
taxonomy/label_all.py from every game of both eras:

  names           style id -> name (the ids the macro models were trained with: the TvZ Terran's tech clusters, the
                  TvZ Zerg's plans)
  shares          style id -> its share among the modern pros' games (on the ladder maps' era), and shares_all
  assignments     'rel_path|player' -> style id, every game (the Terran's by a nearest-neighbour classifier on the
                  clusters' features, the Zerg's by its plan rules: taxonomy/plans.py)

A style is chosen by id or by name (a case-insensitive part of it):

    python -m gary.play --race T --style "standard bio" ...
    python -m gary.styles TvZ T            # the catalogue: ids, names, shares
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

VERSION = "v2"


def _root() -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    from inventory import data_root
    return data_root() / "interim" / "taxonomy" / VERSION


@lru_cache(maxsize=None)
def catalog(matchup: str, race: str) -> dict | None:
    """The catalogue for a race in a matchup ({"names", "shares", "shares_all", "assignments"}), or None (no styles
    for it yet: TvT, ZvZ and the Protoss matchups)."""
    path = _root() / f"{matchup}_{race}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    d["names"] = {int(k): v for k, v in d["names"].items()}
    d["shares"] = {int(k): v for k, v in d["shares"].items()}
    d["shares_all"] = {int(k): v for k, v in d.get("shares_all", {}).items()}
    return d


def describe(matchup: str, race: str) -> str:
    c = catalog(matchup, race)
    if c is None:
        return f"no build styles for {race} in {matchup} yet (issue #54)"
    lines = [f"{race} in {matchup}: {len(c['names'])} styles (modern pros' share, all eras')"]
    for k, name in sorted(c["names"].items()):
        lines.append(f"  {k:3d}  {name:52s} {c['shares'].get(k, 0):5.1%}  {c['shares_all'].get(k, 0):5.1%}")
    return "\n".join(lines)


def resolve(matchup: str, race: str, style) -> int:
    """A style given by id or by name (a case-insensitive part of it) -> its id; ValueError naming the styles there
    are when it matches none or several."""
    if isinstance(style, int) or (isinstance(style, str) and style.strip().lstrip("-").isdigit()):
        k = int(style)
        c = catalog(matchup, race)
        if c is not None and k not in c["names"]:
            raise ValueError(f"no style {k} for {race} in {matchup}\n{describe(matchup, race)}")
        return k
    c = catalog(matchup, race)
    if c is None:
        raise ValueError(describe(matchup, race))
    want = str(style).strip().lower()
    hits = [k for k, n in c["names"].items() if want == n.lower()] or \
        [k for k, n in c["names"].items() if want in n.lower()]
    if len(hits) != 1:
        raise ValueError(f"style {style!r} matches {'no' if not hits else len(hits)} styles for {race} in {matchup}\n"
                         f"{describe(matchup, race)}")
    return hits[0]


if __name__ == "__main__":
    print(describe(sys.argv[1] if len(sys.argv) > 1 else "TvZ", sys.argv[2] if len(sys.argv) > 2 else "T"))
