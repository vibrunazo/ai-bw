"""The one Gary playing Zerg (manifest gz1, docs/REFACTOR.md Phase 1) against Gary-Z, a 4-minute ZvZ mirror
on a ladder map: its opening's basics still work (drones made and mining, a Spawning Pool, an Extractor, overlords
ahead of the supply). eval/zerg_opening.py compares the openings over many games. Needs the game data and the
ladder maps (skips without them); about a minute.

Run: python -m pytest tests/test_zerg_smoke.py   (or plain: python tests/test_zerg_smoke.py)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAP = "KnockOut 1.4"
SEED = 5001


def test_zerg_opening_basics():
    try:
        from eval.league import ladder_maps
        path = ladder_maps([MAP])[MAP]
    except Exception as e:                      # no game data or ladder maps here
        print(f"skip: {e}")
        return
    from eval.zerg_opening import opening, play
    import eval.zerg_opening as zo
    with tempfile.TemporaryDirectory() as tmp:
        zo.OUT = Path(tmp)
        rep = play((path, SEED, ("gz1", "garyz"), 4))
        per = opening(rep, 4)
    gz = per[min(per)]                          # (side A, the lower slot: gz1)
    assert gz["drones"] >= 9, f"{gz['drones']} drones at 4:00"
    assert gz["pool"] is not None, "no Spawning Pool by 4:00"
    assert gz["extractor"] is not None, "no Extractor by 4:00"
    assert gz["blocked"] <= 90, f"supply-blocked {gz['blocked']} s of 4 minutes"


if __name__ == "__main__":
    test_zerg_opening_basics()
    print("ok")
