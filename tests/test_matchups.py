"""Every matchup Gary can play, a short headless game each (eval/match.py) on a ladder map: both sides' Garys
build (their manifests, the models of their matchup and race), play 3 minutes, and make workers. A matchup
whose Gary can't be built is skipped with the reason (no executor for a race yet, a model kind missing for
the matchup), so the table printed is what plays (docs/REFACTOR.md "Tests across matchups"). Needs the game
data and the ladder maps (skips without them); a minute or two.

Run: python -m pytest tests/test_matchups.py   (or plain: python tests/test_matchups.py)
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAP = "KnockOut 1.4"
SEED = 7001
MINUTES = 3
TERRAN = "gt1"
ZERG = "gz1"
PROTOSS = "gp1"
# matchup -> (side A's manifest, side B's), or the reason it isn't played
MATCHUPS = {"TvZ": (TERRAN, ZERG), "ZvZ": (ZERG, ZERG), "TvT": (TERRAN, TERRAN),
            "PvT": (PROTOSS, TERRAN), "PvZ": (PROTOSS, ZERG), "PvP": (PROTOSS, PROTOSS)}


def _play(path, a: str, b: str) -> tuple[str, str]:
    """(outcome, detail): "ok", "skip" (a Gary can't be built: the reason) or "fail"."""
    from eval.match import play
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with contextlib.redirect_stdout(out):
                play(str(path), MINUTES, SEED, str(Path(tmp) / "m.rep"), a, b)
        except SystemExit as e:                  # (build_gary: a model kind missing for this matchup)
            return "skip", str(e)
    lines = [l for l in out.getvalue().splitlines() if "workers" in l]
    if not lines:
        return "fail", "no status line"
    m = re.search(r"workers \w (\d+) \w (\d+)", lines[-1])
    wa, wb = int(m.group(1)), int(m.group(2))
    if min(wa, wb) < 9:
        return "fail", f"workers at {MINUTES}:00: {wa} and {wb}"
    return "ok", f"workers at {MINUTES}:00: {wa} and {wb}"


def test_matchups():
    try:
        from eval.league import ladder_maps
        path = ladder_maps([MAP])[MAP]
    except Exception as e:                       # no game data or ladder maps here
        print(f"skip: {e}")
        return
    table, failed = [], []
    for name, sides in MATCHUPS.items():
        if isinstance(sides, str):
            table.append((name, "skip", sides))
            continue
        outcome, detail = _play(path, *sides)
        table.append((name, outcome, detail))
        if outcome == "fail":
            failed.append(name)
    for name, outcome, detail in table:
        print(f"{name}  {outcome:4s}  {detail}")
    assert not failed, f"matchups that failed: {failed}"


if __name__ == "__main__":
    test_matchups()
    print("ok")
